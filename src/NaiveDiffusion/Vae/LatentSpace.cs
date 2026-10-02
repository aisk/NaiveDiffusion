using NaiveDiffusion.Images;

namespace NaiveDiffusion.Vae;

/// <summary>What the diffusion model's latent is: how many channels, how many
/// pixels each cell stands for, how the VAE's output is scaled into the
/// space the model was trained in, and how to glance at one without running
/// the decoder.</summary>
/// <param name="Channels">Latent channels — 4 for SDXL, 16 for the newer VAEs.</param>
/// <param name="ScaleFactor">Pixels per latent cell along each axis.</param>
/// <param name="ScalingFactor">Latents carry roughly unit variance only after
/// this scaling; the diffusion model is trained in the scaled space.</param>
/// <param name="ShiftFactor">Subtracted before the scaling, on the VAEs whose
/// latents are not centred; 0 for SDXL.</param>
/// <param name="PreviewColors">One row per latent channel, red, green and blue
/// per row: the projection <see cref="PreviewRgb24"/> draws with.</param>
/// <param name="PreviewBias">The projection's offset, red, green and blue.</param>
public record LatentSpace(int Channels, int ScaleFactor, float ScalingFactor,
    float ShiftFactor, float[][] PreviewColors, float[] PreviewBias)
{
    /// <summary>The VAE's latent, in place, onto the scale the diffusion
    /// model was trained at.</summary>
    public virtual void ToModelScale(float[] latent)
    {
        if (ShiftFactor == 0f)
        {
            for (int i = 0; i < latent.Length; i++)
            {
                latent[i] *= ScalingFactor;
            }
            return;
        }
        for (int i = 0; i < latent.Length; i++)
        {
            latent[i] = (latent[i] - ShiftFactor) * ScalingFactor;
        }
    }

    /// <summary>A sampled latent back onto the VAE's own scale, as a copy.</summary>
    public virtual float[] FromModelScale(float[] latent)
    {
        var unscaled = new float[latent.Length];
        if (ShiftFactor == 0f)
        {
            for (int i = 0; i < latent.Length; i++)
            {
                unscaled[i] = latent[i] / ScalingFactor;
            }
            return unscaled;
        }
        for (int i = 0; i < latent.Length; i++)
        {
            unscaled[i] = latent[i] / ScalingFactor + ShiftFactor;
        }
        return unscaled;
    }

    /// <summary>A picture of a latent without running the VAE. Each latent
    /// channel turns out to contribute close to a fixed colour, so one small
    /// matrix over the channels lands near enough to the decoded image to show
    /// composition, colour and framing — at a fraction of the resolution and
    /// at roughly a ten-thousandth of the decoder's cost, which is what makes
    /// it affordable on every step of a run. The latent is in the sampler's
    /// own scaling; the projection lands in the same [-1, 1] range the decoder
    /// outputs, so it is mapped to bytes the same way. Returns packed RGB24
    /// rows.</summary>
    public byte[] PreviewRgb24(float[] latent, int height, int width)
    {
        int plane = height * width;
        var rgb = new byte[plane * 3];
        for (int pixel = 0; pixel < plane; pixel++)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                float value = PreviewBias[channel];
                for (int source = 0; source < PreviewColors.Length; source++)
                {
                    value += latent[source * plane + pixel] * PreviewColors[source][channel];
                }
                rgb[pixel * 3 + channel] = Pixels.ToByte(value);
            }
        }
        return rgb;
    }
}

/// <summary>A latent space standardized channel by channel: the model was
/// trained on (z − mean_c) / std_c, with one mean and one deviation per
/// channel, which is how the 16-channel video VAEs — Wan 2.1's, and the
/// Qwen-Image VAE that shares its weights — are used. The scalar factors of
/// the base record are 1 and 0 here and never read.</summary>
/// <param name="Means">One per channel, the VAE's own scale.</param>
/// <param name="Deviations">One per channel, the VAE's own scale.</param>
public sealed record StandardizedLatentSpace(int Channels, int ScaleFactor,
    float[] Means, float[] Deviations, float[][] PreviewColors, float[] PreviewBias)
    : LatentSpace(Channels, ScaleFactor, ScalingFactor: 1f, ShiftFactor: 0f,
        PreviewColors, PreviewBias)
{
    public override void ToModelScale(float[] latent)
    {
        int plane = latent.Length / Channels;
        for (int channel = 0; channel < Channels; channel++)
        {
            float mean = Means[channel];
            float deviation = Deviations[channel];
            for (int i = channel * plane; i < (channel + 1) * plane; i++)
            {
                latent[i] = (latent[i] - mean) / deviation;
            }
        }
    }

    public override float[] FromModelScale(float[] latent)
    {
        var unscaled = new float[latent.Length];
        int plane = latent.Length / Channels;
        for (int channel = 0; channel < Channels; channel++)
        {
            float mean = Means[channel];
            float deviation = Deviations[channel];
            for (int i = channel * plane; i < (channel + 1) * plane; i++)
            {
                unscaled[i] = latent[i] * deviation + mean;
            }
        }
        return unscaled;
    }
}
