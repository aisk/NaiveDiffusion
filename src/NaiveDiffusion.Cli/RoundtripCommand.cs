using NaiveDiffusion.Dml;
using NaiveDiffusion.Models;
using NaiveDiffusion.Pipeline;

namespace NaiveDiffusion.Cli;

/// <summary>An image through the family's VAE encoder and decoder. The
/// decoded PSNR is the floor on any output, so a healthy value proves the
/// whole VAE — SDXL's from the checkpoint or the file given for it, Anima's
/// and Qwen-Image's from their VAE files. The latent on the model's scale
/// and the decoded image can be dumped raw (--dump-latent, --dump-decoded)
/// to set against a reference's.</summary>
internal static class RoundtripCommand
{
    public static readonly string[] ValueOptions = new[]
    {
        "--checkpoint", "--size", "--dump-latent", "--dump-decoded",
    }.Concat(CommandLine.ComponentOptions()).ToArray();
    public static readonly string[] FlagOptions = { "--tile-encode", "--tile-decode" };

    public static int Run(DmlDevice device, CommandLine line)
    {
        string path = line.Argument ?? CommandLine.Fail<string>("roundtrip needs an image file");
        int size = line.Int("--size", 512);

        // Only the VAE is read: an SDXL checkpoint stripped of its towers
        // still roundtrips, and Anima's text encoder is not asked for.
        IModelFamily family = line.Family(CheckpointInspector.CheckpointParts.Vae);
        var options = new GenerationOptions
        {
            CheckpointPath = line.Checkpoint(CheckpointInspector.CheckpointParts.Vae),
            Components = line.Components(family, PipelineParts.Codec),
            Width = size,
            Height = size,
            TileVae = false,
        };
        if (size % family.Codec.Latent.ScaleFactor != 0)
        {
            return CommandLine.Fail<int>($"--size {size}: a multiple of {family.Codec.Latent.ScaleFactor}");
        }
        ILatentCodec codec = family.Codec;
        float[] original = ImageFile.LoadSquare(path, size);

        // The encoder and the decoder are each scoped so the one is gone
        // before the other is built: they never need the card at once.
        Console.WriteLine($"Encoding {size}x{size}");
        float[] latent;
        ulong wholeEncodeScratch;
        using (ILatentEncoder encoder = codec.OpenEncoder(device, options))
        {
            latent = encoder.Encode(original);
            wholeEncodeScratch = encoder.TemporaryBytes;
        }
        Console.WriteLine($"  latent mean {latent.Average():+0.000;-0.000}, std {Statistics.Std(latent):0.000}");
        if (line.Value("--dump-latent") is string dumpLatent)
        {
            Dumps.Write(dumpLatent, latent);
        }

        // The tiled encoder against the whole one, on the latent it is
        // actually asked for. The decoded image is what the difference
        // shows up in, so that comparison comes after the decode below.
        float[]? tiledLatent = null;
        if (line.Flag("--tile-encode"))
        {
            Console.WriteLine("Encoding again in tiles");
            using ILatentEncoder tiled = codec.OpenEncoder(device, options with { TileVae = true });
            tiledLatent = tiled.Encode(original);
            Console.WriteLine($"  scratch {tiled.TemporaryBytes / (double)(1 << 30):0.00} GiB against " +
                              $"{wholeEncodeScratch / (double)(1 << 30):0.00} GiB whole");
            double rms = Math.Sqrt(latent.Zip(tiledLatent, (a, b) => (a - b) * (double)(a - b)).Average());
            Console.WriteLine($"  latent rms difference {rms:0.000} against a std of {Statistics.Std(latent):0.000}");
        }

        Console.WriteLine("Decoding");
        float[] decoded;
        ulong wholeScratch;
        using (ILatentDecoder decoder = codec.OpenDecoder(device, options))
        {
            decoded = decoder.Decode(latent);
            wholeScratch = decoder.TemporaryBytes;
            Console.WriteLine($"Reconstruction PSNR {Statistics.Psnr(original, decoded):0.00} dB");
            if (line.Value("--dump-decoded") is string dumpDecoded)
            {
                // In [0, 1] as a reference's image tensor is.
                Dumps.Write(dumpDecoded, decoded.Select(value => value * 0.5f + 0.5f).ToArray());
            }

            if (tiledLatent is not null)
            {
                float[] fromTiles = decoder.Decode(tiledLatent);
                Console.WriteLine($"  decoded from the tiled encode: against the whole-encode path " +
                                  $"{Statistics.Psnr(decoded, fromTiles):0.00} dB, " +
                                  $"against the original {Statistics.Psnr(original, fromTiles):0.00} dB");
            }
        }

        if (line.Flag("--tile-decode"))
        {
            Console.WriteLine("Decoding again in tiles");
            using ILatentDecoder tiled = codec.OpenDecoder(device, options with { TileVae = true });
            float[] inTiles = tiled.Decode(latent);
            Console.WriteLine($"  scratch {tiled.TemporaryBytes / (double)(1 << 30):0.00} GiB against " +
                              $"{wholeScratch / (double)(1 << 30):0.00} GiB whole");
            Console.WriteLine($"  against the whole-image decode {Statistics.Psnr(decoded, inTiles):0.00} dB, " +
                              $"against the original {Statistics.Psnr(original, inTiles):0.00} dB");
        }

        ImageFile.SavePlanar(decoded, size, size, "roundtrip.png");
        Console.WriteLine("Wrote roundtrip.png");
        return 0;
    }
}
