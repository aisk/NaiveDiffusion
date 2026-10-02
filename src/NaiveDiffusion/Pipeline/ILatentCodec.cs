using NaiveDiffusion.Dml;
using NaiveDiffusion.Images;
using NaiveDiffusion.Vae;

namespace NaiveDiffusion.Pipeline;

/// <summary>The two ends of the latent space: a reference image into the
/// latent a run starts from, and a sampled latent into pixels. Whatever the
/// family's VAE is, it is built here and released when the decoder is.</summary>
public interface ILatentCodec
{
    LatentSpace Latent { get; }

    /// <summary>Build the encoder for this run's size, whole or tiled as the
    /// options say; released by the caller once the reference is encoded.</summary>
    ILatentEncoder OpenEncoder(DmlDevice device, GenerationOptions options);

    /// <summary>Build the decoder for this run's size; it decodes every
    /// image of the run and is disposed after the last.</summary>
    ILatentDecoder OpenDecoder(DmlDevice device, GenerationOptions options);

    /// <summary>Video memory the decode stage adds on top of whatever is
    /// already resident: the decoder's weights plus the scratch one dispatch
    /// needs, at this run's size.</summary>
    ulong EstimateDecodeBytes(GenerationOptions options);
}

public static class LatentCodecs
{
    /// <summary>The reference image as the latent a sampling run would have
    /// reached had it drawn this image, already on the scale the diffusion
    /// model was trained at. The encoder is built and released inside.</summary>
    public static float[] EncodeReference(this ILatentCodec codec, DmlDevice device,
        GenerationOptions options, ImageResult image)
    {
        using ILatentEncoder encoder = codec.OpenEncoder(device, options);
        return encoder.Encode(Pixels.FromRgb24(image.Rgb24, image.Height, image.Width));
    }
}

/// <summary>A built encoder: one [1, 3, H, W] image in [-1, 1] in, the
/// latent it stands for out — the distribution's mean rather than a draw
/// from it, on the diffusion model's scale.</summary>
public interface ILatentEncoder : IDisposable
{
    float[] Encode(float[] image);

    /// <summary>Scratch one dispatch needs, for the size reports.</summary>
    ulong TemporaryBytes { get; }
}

/// <summary>A built decoder: one latent on the diffusion model's scale in,
/// one [1, 3, H, W] image in [-1, 1] out.</summary>
public interface ILatentDecoder : IDisposable
{
    float[] Decode(float[] latent);

    /// <summary>Scratch one dispatch needs, for the size reports.</summary>
    ulong TemporaryBytes { get; }
}
