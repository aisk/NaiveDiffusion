using NaiveDiffusion.Dml;
using NaiveDiffusion.Graph;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Tensors;

namespace NaiveDiffusion.Vae;

/// <summary>A latent codec over any VAE that has an encoder graph and a
/// decoder graph: what every family's codec does the same way — build one
/// end for this run's size, whole or in tiles as the options say, scale the
/// latent to and from the diffusion model's space, and release the graph
/// when the stage is done. A family supplies the latent space, where its
/// VAE's tensors come from, the two graph builders and the weight size the
/// decode estimate starts from.</summary>
public abstract class VaeLatentCodec : ILatentCodec
{
    public abstract LatentSpace Latent { get; }

    /// <summary>The VAE's tensors for this run, and whatever they are
    /// windows onto — null when they are copies. The mapping is kept open
    /// until the graph has been built and the weights uploaded, and closed
    /// then.</summary>
    protected abstract (IDisposable? Mapping, Dictionary<string, HostTensor> Parameters) LoadVae(
        GenerationOptions options);

    /// <summary>The encoder graph for one [1, 3, height, width] image.</summary>
    protected abstract ModelBuilder BuildEncoder(DmlDevice device,
        IReadOnlyDictionary<string, HostTensor> parameters, int height, int width);

    /// <summary>The decoder graph for one [1, C, height/f, width/f] latent.</summary>
    protected abstract ModelBuilder BuildDecoder(DmlDevice device,
        IReadOnlyDictionary<string, HostTensor> parameters, int height, int width);

    /// <summary>The decoder's weights in video memory, for the estimate a
    /// cached denoiser is measured against.</summary>
    protected abstract ulong DecoderWeightBytes { get; }

    /// <summary>The scratch a whole-image decode needs at 1024², which grows
    /// with the pixel count: 5.80 GiB measured for SDXL's VAE, and taken as
    /// the same for a VAE not measured yet, which errs towards releasing a
    /// cached denoiser rather than keeping one that will not fit.</summary>
    protected virtual ulong DecoderScratchPerMegapixel => 5939UL << 20;

    /// <summary>The VAE encoder, taking the distribution's mean rather than a
    /// draw from it, then onto the scale the diffusion model was trained
    /// at. At the full image size unless tiled, so on a card that needs the
    /// tiled decoder this is the stage with the largest single allocation;
    /// the pipeline runs it before the denoiser is built, when the card is
    /// at its emptiest.</summary>
    public ILatentEncoder OpenEncoder(DmlDevice device, GenerationOptions options)
    {
        (IDisposable? mapping, Dictionary<string, HostTensor> parameters) = LoadVae(options);
        using (mapping)
        {
            if (options.TileVae)
            {
                var tiled = new TiledVaeEncoder(Latent,
                    side => BuildEncoder(device, parameters, side, side), options.Height, options.Width);
                parameters.Clear();
                return new Encoder(tiled, tiled.Run, () => tiled.TemporarySize, Latent);
            }
            ModelBuilder whole = BuildEncoder(device, parameters, options.Height, options.Width);
            parameters.Clear();
            return new Encoder(whole, pixels => whole.Run(HostTensor.FromFloats(pixels,
                    1, 3, options.Height, options.Width))[0].ToFloats(),
                () => whole.TemporarySize, Latent);
        }
    }

    public ILatentDecoder OpenDecoder(DmlDevice device, GenerationOptions options)
    {
        (IDisposable? mapping, Dictionary<string, HostTensor> parameters) = LoadVae(options);
        using (mapping)
        {
            int latentHeight = options.Height / Latent.ScaleFactor;
            int latentWidth = options.Width / Latent.ScaleFactor;
            if (options.TileVae)
            {
                var tiled = new TiledVaeDecoder(Latent,
                    side => BuildDecoder(device, parameters, side, side), options.Height, options.Width);
                parameters.Clear();
                return new Decoder(tiled, tiled.Run, () => tiled.TemporarySize, Latent);
            }
            ModelBuilder whole = BuildDecoder(device, parameters, options.Height, options.Width);
            parameters.Clear();
            return new Decoder(whole, latent => whole.Run(HostTensor.FromFloats(latent,
                    1, Latent.Channels, latentHeight, latentWidth))[0].ToFloats(),
                () => whole.TemporarySize, Latent);
        }
    }

    /// <summary>The decoder's weights plus scratch scaled by the area. A
    /// tiled decode does one tile at a time however large the image, so its
    /// scratch stops at what one tile's worth of image needs — and an image
    /// of one tile or less decodes the same either way, which is why this is
    /// a floor on the area rather than a separate case.</summary>
    public ulong EstimateDecodeBytes(GenerationOptions options)
    {
        double megapixels = (double)options.Width * options.Height / (1024.0 * 1024.0);
        if (options.TileVae)
        {
            int side = TiledVae.TileLatent * Latent.ScaleFactor;
            megapixels = Math.Min(megapixels, (double)side * side / (1024.0 * 1024.0));
        }
        return DecoderWeightBytes + (ulong)(DecoderScratchPerMegapixel * megapixels);
    }

    private sealed class Encoder : ILatentEncoder
    {
        private readonly IDisposable _model;
        private readonly Func<float[], float[]> _encode;
        private readonly Func<ulong> _temporary;
        private readonly LatentSpace _latent;

        public Encoder(IDisposable model, Func<float[], float[]> encode, Func<ulong> temporary,
            LatentSpace latent)
        {
            _model = model;
            _encode = encode;
            _temporary = temporary;
            _latent = latent;
        }

        public float[] Encode(float[] image)
        {
            float[] latent = _encode(image);
            _latent.ToModelScale(latent);
            return latent;
        }

        public ulong TemporaryBytes => _temporary();

        public void Dispose() => _model.Dispose();
    }

    private sealed class Decoder : ILatentDecoder
    {
        private readonly IDisposable _model;
        private readonly Func<float[], float[]> _decode;
        private readonly Func<ulong> _temporary;
        private readonly LatentSpace _latent;

        public Decoder(IDisposable model, Func<float[], float[]> decode, Func<ulong> temporary,
            LatentSpace latent)
        {
            _model = model;
            _decode = decode;
            _temporary = temporary;
            _latent = latent;
        }

        public float[] Decode(float[] latent) => _decode(_latent.FromModelScale(latent));

        public ulong TemporaryBytes => _temporary();

        public void Dispose() => _model.Dispose();
    }
}
