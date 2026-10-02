using NaiveDiffusion.Dml;
using NaiveDiffusion.Graph;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Tensors;
using NaiveDiffusion.Vae;

namespace NaiveDiffusion.Models.Sdxl;

/// <summary>SDXL's VAE as the pipeline sees it: diffusers' AutoencoderKL,
/// built from the checkpoint — or from the VAE file given in its place — at
/// each end of a run.</summary>
public sealed class SdxlLatentCodec : VaeLatentCodec
{
    public override LatentSpace Latent => SdxlVae.Latent;

    /// <summary>The checkpoint stays mapped until the graph is built: the
    /// tensors are windows onto it when the file already stores them at
    /// single precision.</summary>
    protected override (IDisposable? Mapping, Dictionary<string, HostTensor> Parameters) LoadVae(
        GenerationOptions options)
    {
        var checkpoint = new SdxlWeights(options.CheckpointPath,
            options.ComponentPath(SdxlFamily.VaeComponent.Id));
        try
        {
            return (checkpoint, checkpoint.LoadVae());
        }
        catch
        {
            checkpoint.Dispose();
            throw;
        }
    }

    protected override ModelBuilder BuildEncoder(DmlDevice device,
        IReadOnlyDictionary<string, HostTensor> parameters, int height, int width) =>
        AutoencoderKl.Encoder(device, parameters, SdxlVae.Config, height, width);

    protected override ModelBuilder BuildDecoder(DmlDevice device,
        IReadOnlyDictionary<string, HostTensor> parameters, int height, int width) =>
        AutoencoderKl.Decoder(device, parameters, SdxlVae.Config, height, width);

    /// <summary>188 MiB measured at 1024², with the scratch the base class
    /// assumes measured on this very VAE.</summary>
    protected override ulong DecoderWeightBytes => 188UL << 20;
}
