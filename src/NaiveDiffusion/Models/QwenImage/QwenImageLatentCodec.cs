using NaiveDiffusion.Dml;
using NaiveDiffusion.Graph;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Tensors;
using NaiveDiffusion.Vae;

namespace NaiveDiffusion.Models.QwenImage;

/// <summary>Qwen-Image 2.1's latent space as the pipeline sees it: the
/// Wan 2.2-layout VAE from the file given for it, on one frame, with the
/// per-channel standardization the transformer was trained in on either
/// side.</summary>
public sealed class QwenImageLatentCodec : VaeLatentCodec
{
    public override LatentSpace Latent => Wan22Vae.Latent;

    /// <summary>The single-frame graph wants copies at single precision, so
    /// the mapping closes here.</summary>
    protected override (IDisposable? Mapping, Dictionary<string, HostTensor> Parameters) LoadVae(
        GenerationOptions options)
    {
        using var weights = new QwenImageWeights(options.CheckpointPath,
            vaePath: options.ComponentPath(QwenImageFamily.VaeComponent.Id)
                ?? throw new ArgumentException("Qwen-Image needs its VAE file"));
        return (null, weights.LoadVae());
    }

    protected override ModelBuilder BuildEncoder(DmlDevice device,
        IReadOnlyDictionary<string, HostTensor> parameters, int height, int width) =>
        Wan22Vae.Encoder(device, parameters, height, width);

    protected override ModelBuilder BuildDecoder(DmlDevice device,
        IReadOnlyDictionary<string, HostTensor> parameters, int height, int width) =>
        Wan22Vae.Decoder(device, parameters, height, width);

    /// <summary>The file is 675 MB of bfloat16, decoder and encoder alike,
    /// widened to single precision for the graph; 900 MiB is a round figure
    /// above the decoder's share, until measured.</summary>
    protected override ulong DecoderWeightBytes => 900UL << 20;
}
