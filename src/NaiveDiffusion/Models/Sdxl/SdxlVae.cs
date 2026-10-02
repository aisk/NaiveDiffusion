using NaiveDiffusion.Vae;

namespace NaiveDiffusion.Models.Sdxl;

/// <summary>SDXL's VAE as diffusers' sdxl-vae config, and its latent space.</summary>
public static class SdxlVae
{
    /// <summary>Four channels, eight pixels a cell, the published scaling and
    /// no shift. The preview constants are ComfyUI's, fitted against the SDXL
    /// VAE.</summary>
    public static readonly LatentSpace Latent = new(
        Channels: 4,
        ScaleFactor: 8,
        ScalingFactor: 0.13025f,
        ShiftFactor: 0f,
        PreviewColors: new[]
        {
            new[] { 0.3651f, 0.4232f, 0.4341f },
            new[] { -0.2533f, -0.0042f, 0.1068f },
            new[] { 0.1076f, 0.1111f, -0.0362f },
            new[] { -0.3165f, -0.2492f, -0.2188f },
        },
        PreviewBias: new[] { 0.1084f, -0.0175f, -0.0011f });

    public static readonly VaeConfig Config = new(
        BlockOutChannels: new[] { 128, 256, 512, 512 },
        LayersPerBlock: 2,
        NormGroups: 32,
        NormEpsilon: 1e-6f,
        Latent: Latent);
}
