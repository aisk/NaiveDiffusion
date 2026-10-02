using NaiveDiffusion.Dml;
using NaiveDiffusion.Graph;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Tensors;
using NaiveDiffusion.Vae;

namespace NaiveDiffusion.Models.Anima;

/// <summary>Anima's latent space as the pipeline sees it: the Wan-family VAE
/// from the file given for it, on one frame, with the per-channel
/// standardization the transformer was trained in on either side.</summary>
public sealed class AnimaLatentCodec : VaeLatentCodec
{
    public override LatentSpace Latent => WanVae.Latent;

    /// <summary>The single-frame graph wants copies at single precision, so
    /// the mapping closes here.</summary>
    protected override (IDisposable? Mapping, Dictionary<string, HostTensor> Parameters) LoadVae(
        GenerationOptions options)
    {
        using var weights = new AnimaWeights(options.CheckpointPath,
            vaePath: options.ComponentPath(AnimaFamily.VaeComponent.Id)
                ?? throw new ArgumentException("Anima needs its VAE file"));
        return (null, weights.LoadVae());
    }

    protected override ModelBuilder BuildEncoder(DmlDevice device,
        IReadOnlyDictionary<string, HostTensor> parameters, int height, int width) =>
        WanVae.Encoder(device, parameters, height, width);

    protected override ModelBuilder BuildDecoder(DmlDevice device,
        IReadOnlyDictionary<string, HostTensor> parameters, int height, int width) =>
        WanVae.Decoder(device, parameters, height, width);

    /// <summary>The Wan VAE is narrower than SDXL's (96 base channels
    /// against 128) but has more blocks; 500 MiB is a round figure above
    /// what its weights take, until measured.</summary>
    protected override ulong DecoderWeightBytes => 500UL << 20;
}
