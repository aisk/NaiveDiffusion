using System.Globalization;
using NaiveDiffusion.Dml;
using NaiveDiffusion.Models;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Sampling;
using NaiveDiffusion.Tensors;

namespace NaiveDiffusion.Cli;

/// <summary>One forward pass of the diffusion model on one latent at a
/// given noise level, under the prompt's conditioning: what a reference
/// implementation can be made to compute exactly, and so the check that the
/// model itself — not the sampler around it — is right. The latent comes
/// from a raw float32 file or a seed; the prediction goes to another. For
/// a family whose conditioning is a plain sequence of rows
/// (<see cref="IConditioner.TakesContextRows"/>) the rows can come from a
/// file too (--context), so a small transformer of the same shape runs
/// against the reference with no text encoder at all — and then no other
/// file is asked for, since nothing else is read. LoRAs (--lora) are folded
/// in as generate folds them, so a merge can be checked the same way.</summary>
internal static class DenoiseCommand
{
    public static readonly string[] ValueOptions = new[]
    {
        "--checkpoint", "--size", "--width", "--height", "--sigma", "--latent", "--seed", "--dump",
        "--unet-vram", "--context", "--lora",
    }.Concat(CommandLine.ComponentOptions()).ToArray();

    public static readonly string[] FlagOptions = { "--int8", "--fp16-compute" };

    /// <summary>What one pass is: the family, the options the pipeline has
    /// passed for the parts that will run, the level, and the conditioning
    /// read from the file of context rows in place of the text encoder, if
    /// one was given. No device is opened, so every refusal lands before
    /// anything loads.</summary>
    public static (IModelFamily Family, GenerationOptions Options, ModelSampling Sampling, float Sigma,
        Conditioning? Context) Read(CommandLine line)
    {
        IModelFamily family = line.Family();
        (int width, int height) = line.Size(256);
        string? contextPath = line.Value("--context");
        if (contextPath is not null && !family.Conditioner.TakesContextRows)
        {
            CommandLine.Fail($"--context: {family.Name}'s conditioning is not a file of rows");
        }
        if (contextPath is not null && !File.Exists(contextPath))
        {
            CommandLine.Fail($"--context {contextPath}: no such file");
        }
        // The text encoder runs only without --context; the VAE never does.
        PipelineParts parts = contextPath is null
            ? PipelineParts.Conditioner | PipelineParts.Denoiser
            : PipelineParts.Denoiser;
        var options = new GenerationOptions
        {
            Prompt = line.Argument ?? "an astronaut riding a horse on mars",
            // One pass on the prompt's branch: a family whose guidance is
            // off at 1 then encodes no negative prompt.
            Guidance = 1f,
            Width = width,
            Height = height,
            CheckpointPath = line.Checkpoint(),
            Components = line.Components(family, parts),
        };
        options = line.DenoiserMemory(options, allowAuto: false);
        options = line.Loras(options);
        float sigma = line.Float("--sigma", 0.7f);
        if (!(sigma > 0))
        {
            CommandLine.Fail($"--sigma {sigma}: above 0");
        }
        ModelSampling sampling = CommandLine.Prepare(ModelFamilies.PipelineFor(family), options, parts);
        Conditioning? context = null;
        if (contextPath is not null)
        {
            float[] rows = Dumps.Read("--context", contextPath, 1, "floats");
            try
            {
                context = family.Conditioner.FromContextRows(options, rows);
            }
            catch (ArgumentException exception)
            {
                CommandLine.Fail($"--context {contextPath}: {exception.Message}");
            }
            Console.WriteLine($"context of {rows.Length} values from {contextPath}");
        }
        return (family, options, sampling, sigma, context);
    }

    public static int Run(DmlDevice device, CommandLine line)
    {
        (IModelFamily family, GenerationOptions options, ModelSampling sampling, float sigma,
            Conditioning? context) = Read(line);
        GenerationPipeline pipeline = ModelFamilies.PipelineFor(family);

        int channels = family.Codec.Latent.Channels;
        int latentHeight = options.Height / family.Codec.Latent.ScaleFactor;
        int latentWidth = options.Width / family.Codec.Latent.ScaleFactor;
        int length = channels * latentHeight * latentWidth;
        float[] latent;
        if (line.Value("--latent") is string latentPath)
        {
            latent = Dumps.Read("--latent", latentPath, length,
                $"[{channels}, {latentHeight}, {latentWidth}] latents");
            if (latent.Length != length)
            {
                return CommandLine.Fail<int>($"--latent {latentPath}: {latent.Length} values, " +
                                             $"expected {length} for [{channels}, {latentHeight}, {latentWidth}]");
            }
        }
        else
        {
            latent = new GaussianGenerator(line.Seed()).Fill(length);
        }

        Conditioning conditioning;
        if (context is null)
        {
            Console.WriteLine("Encoding prompt");
            int branches = family.Guidance.Branches(options.Guidance);
            conditioning = family.Conditioner.Encode(options, branches).CheckedFor(branches);
        }
        else
        {
            conditioning = context;
        }
        double timestep = sampling.Levels.TimestepAt(sigma);
        Console.WriteLine($"sigma {sigma.ToString(CultureInfo.InvariantCulture)} is training timestep {timestep:0.###}");

        Console.WriteLine("Building the model");
        using IDenoiser denoiser = pipeline.BuildDenoiser(device, options, conditioning, default);
        Console.WriteLine($"{denoiser.GraphCount} graphs, {denoiser.PersistentBytes >> 20} MiB of weights, " +
                          $"{denoiser.StreamedBytes >> 20} MiB streamed, {denoiser.TemporaryBytes >> 20} MiB scratch");

        float[] input = latent;
        float scale = sampling.Prediction.ModelInputScale(sigma);
        if (scale != 1f)
        {
            input = latent.Select(value => value * scale).ToArray();
        }
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        // The last branch is the prompt's own; the first is the negative's.
        float[] prediction = denoiser.Predict(
            HostTensor.FromFloats(input, 1, channels, latentHeight, latentWidth),
            timestep, sigma, conditioning.Branch(conditioning.Branches - 1)).ToFloats();
        Console.WriteLine($"Predicted in {stopwatch.Elapsed.TotalSeconds:0.00} s");
        Console.WriteLine($"input [{channels}, {latentHeight}, {latentWidth}]: mean {latent.Average():+0.0000;-0.0000} " +
                          $"std {Statistics.Std(latent):0.0000} first {latent[0]:0.0000}");
        Console.WriteLine($"output: mean {prediction.Average():+0.0000;-0.0000} " +
                          $"std {Statistics.Std(prediction):0.0000} first {prediction[0]:0.0000}");
        if (Array.Exists(prediction, value => !float.IsFinite(value)))
        {
            Console.WriteLine("warning: the output holds NaN or infinity");
        }

        if (line.Value("--dump") is string dump)
        {
            Dumps.Write(dump, prediction);
        }
        return 0;
    }
}
