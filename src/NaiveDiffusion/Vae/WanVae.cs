using NaiveDiffusion.Dml;
using NaiveDiffusion.Graph;
using NaiveDiffusion.Tensors;
using Vortice.DirectML;

namespace NaiveDiffusion.Vae;

/// <summary>The Wan 2.1 video VAE — the one the Qwen-Image VAE and Anima
/// decode with — on a single frame, as a 2-D DirectML graph under the file's
/// own names.
///
/// The model is a causal 3-D convolutional autoencoder: every convolution
/// looks at the current frame and the two before it, padded with zeros where
/// there are none. On one frame that padding is all the past there is, so
/// only the last temporal tap of each kernel ever meets data and the 3-D
/// convolution is exactly a 2-D one with that slice of the kernel — which is
/// what <see cref="SingleFrame"/> cuts out, and what ComfyUI's own fast path
/// does for a one-frame chunk. The temporal resampling convolutions are
/// skipped on the first frame by the reference too, so they are not built.
/// The norm is an RMS norm over the channels rather than a GroupNorm, the
/// activation SiLU, the middle attention a single head over every pixel.
/// The blocks are shared with <see cref="Wan22Vae"/>, the wider layout
/// Qwen-Image 2.1 decodes with.</summary>
public static class WanVae
{
    public const int LatentChannels = 16;
    private static readonly int[] WidthMultipliers = { 1, 2, 4, 4 };
    private const int ResnetsPerLevel = 2;

    /// <summary>Wan 2.1's latent as ComfyUI's latent_formats.Wan21 has it:
    /// sixteen channels, eight pixels a cell, standardized channel by channel,
    /// with the RGB projection fitted for previews.</summary>
    public static readonly LatentSpace Latent = new StandardizedLatentSpace(
        Channels: LatentChannels,
        ScaleFactor: 8,
        Means: new[]
        {
            -0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
            0.4134f, -0.0715f, 0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f,
        },
        Deviations: new[]
        {
            2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
            3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f,
        },
        PreviewColors: new[]
        {
            new[] { -0.1299f, -0.1692f, 0.2932f }, new[] { 0.0671f, 0.0406f, 0.0442f },
            new[] { 0.3568f, 0.2548f, 0.1747f }, new[] { 0.0372f, 0.2344f, 0.1420f },
            new[] { 0.0313f, 0.0189f, -0.0328f }, new[] { 0.0296f, -0.0956f, -0.0665f },
            new[] { -0.3477f, -0.4059f, -0.2925f }, new[] { 0.0166f, 0.1902f, 0.1975f },
            new[] { -0.0412f, 0.0267f, -0.1364f }, new[] { -0.1293f, 0.0740f, 0.1636f },
            new[] { 0.0680f, 0.3019f, 0.1128f }, new[] { 0.0032f, 0.0581f, 0.0639f },
            new[] { -0.1251f, 0.0927f, 0.1699f }, new[] { 0.0060f, -0.0633f, 0.0005f },
            new[] { 0.3477f, 0.2275f, 0.2950f }, new[] { 0.1984f, 0.0913f, 0.1861f },
        },
        PreviewBias: new[] { -0.1835f, -0.0868f, -0.3360f });

    /// <summary>The file's tensors as the single-frame graph reads them, at
    /// float32: every 3-D kernel cut to its last temporal tap, the norm
    /// scales flattened to one value per channel, and the temporal
    /// resampling convolutions — never run on a first frame — left out.</summary>
    public static Dictionary<string, HostTensor> SingleFrame(
        IReadOnlyDictionary<string, HostTensor> raw)
    {
        var tensors = new Dictionary<string, HostTensor>(raw.Count);
        foreach ((string name, HostTensor tensor) in raw)
        {
            if (name.Contains(".time_conv.", StringComparison.Ordinal))
            {
                continue;
            }
            if (name.EndsWith(".gamma", StringComparison.Ordinal))
            {
                tensors[name] = HostTensor.FromFloats(tensor.ToFloats(), tensor.Shape[0]);
            }
            else if (tensor.Shape.Length == 5)
            {
                tensors[name] = LastTemporalTap(tensor);
            }
            else
            {
                tensors[name] = tensor.ConvertTo(HostDataType.Float32);
            }
        }
        return tensors;
    }

    /// <summary>[out, in, kt, kh, kw] to [out, in, kh, kw] at the last kt.</summary>
    private static HostTensor LastTemporalTap(HostTensor kernel)
    {
        int[] s = kernel.Shape;
        (int outChannels, int inChannels, int taps, int kh, int kw) = (s[0], s[1], s[2], s[3], s[4]);
        float[] wide = kernel.ToFloats();
        int plane = kh * kw;
        var sliced = new float[outChannels * inChannels * plane];
        for (int o = 0; o < outChannels; o++)
        {
            for (int i = 0; i < inChannels; i++)
            {
                Array.Copy(wide, ((o * inChannels + i) * taps + taps - 1) * plane,
                    sliced, (o * inChannels + i) * plane, plane);
            }
        }
        return HostTensor.FromFloats(sliced, outChannels, inChannels, kh, kw);
    }

    /// <summary>Wan's RMS_norm: each pixel's channel vector scaled to length
    /// sqrt(C), then per channel by gamma — F.normalize over the channel axis
    /// times sqrt(dim) times the weight.</summary>
    internal static DmlExpression ChannelRmsNorm(ModelBuilder model, DmlExpression x, HostTensor gamma)
    {
        int channels = (int)x.Shape[1];
        // x / |x| · sqrt(C) is x / sqrt(mean(x²)): RMS normalization over the
        // channel axis. F.normalize divides by max(norm, 1e-12); the floor,
        // squared and taken per channel, keeps a zero vector finite and is
        // otherwise nothing.
        DmlExpression scale = model.Constant(HostTensor.FromFloats(gamma.ToFloats(), 1, channels, 1, 1));
        return DmlOps.MeanVarianceNormalization2(x, new[] { 1 }, false, 1e-24f / channels, scale);
    }

    internal static DmlExpression Conv(ModelBuilder model, DmlExpression x,
        IReadOnlyDictionary<string, HostTensor> parameters, string name, int stride = 1,
        int padding = 1, int? endPadding = null) =>
        Layers.Conv2d(model, x, parameters[$"{name}.weight"], parameters[$"{name}.bias"],
            stride, padding, endPadding);

    /// <summary>norm, SiLU, 3×3 conv, norm, SiLU, 3×3 conv, plus the input
    /// through a 1×1 conv where the width changes.</summary>
    internal static DmlExpression ResidualBlock(ModelBuilder model, DmlExpression x,
        IReadOnlyDictionary<string, HostTensor> parameters, string prefix)
    {
        DmlExpression h = Conv(model, Layers.Silu(ChannelRmsNorm(model, x,
            parameters[$"{prefix}.residual.0.gamma"])), parameters, $"{prefix}.residual.2");
        h = Conv(model, Layers.Silu(ChannelRmsNorm(model, h,
            parameters[$"{prefix}.residual.3.gamma"])), parameters, $"{prefix}.residual.6");
        if (parameters.ContainsKey($"{prefix}.shortcut.weight"))
        {
            x = Conv(model, x, parameters, $"{prefix}.shortcut", padding: 0);
        }
        return x + h;
    }

    /// <summary>Single-head attention over every pixel with a residual, the
    /// projections as 1×1 convolutions, as the reference writes them.</summary>
    internal static DmlExpression AttentionBlock(ModelBuilder model, DmlExpression x,
        IReadOnlyDictionary<string, HostTensor> parameters, string prefix)
    {
        uint[] shape = x.Shape;
        (int channels, uint height, uint width) = ((int)shape[1], shape[2], shape[3]);
        DmlExpression qkv = Conv(model, ChannelRmsNorm(model, x, parameters[$"{prefix}.norm.gamma"]),
            parameters, $"{prefix}.to_qkv", padding: 0);

        DmlExpression Part(int index) => Layers.ToTokens(DmlOps.Slice(qkv,
            offsets: new[] { 0, index * channels, 0, 0 },
            sizes: new[] { 1, channels, (int)height, (int)width },
            strides: new[] { 1, 1, 1, 1 }));

        DmlExpression scores = DmlOps.Gemm(Part(0), Part(1),
            transB: MatrixTransform.Transpose, alpha: 1.0f / MathF.Sqrt(channels));
        DmlExpression attended = DmlOps.Gemm(DmlOps.ActivationSoftmax(scores, new[] { 3 }), Part(2));
        return Conv(model, Layers.ToImage(attended, height, width), parameters, $"{prefix}.proj",
            padding: 0) + x;
    }

    internal static DmlExpression Middle(ModelBuilder model, DmlExpression x,
        IReadOnlyDictionary<string, HostTensor> parameters, string prefix)
    {
        x = ResidualBlock(model, x, parameters, $"{prefix}.middle.0");
        x = AttentionBlock(model, x, parameters, $"{prefix}.middle.1");
        return ResidualBlock(model, x, parameters, $"{prefix}.middle.2");
    }

    /// <summary>Compile a decoder for a height by width image: one latent on
    /// the VAE's own scale in, one [1, 3, H, W] image out.</summary>
    public static ModelBuilder Decoder(DmlDevice device,
        IReadOnlyDictionary<string, HostTensor> parameters, int height, int width)
    {
        var model = new ModelBuilder(device);
        DmlExpression z = model.Placeholder(new[]
        {
            1, LatentChannels, height / Latent.ScaleFactor, width / Latent.ScaleFactor,
        });
        DmlExpression h = Conv(model, z, parameters, "conv2", padding: 0);
        h = Conv(model, h, parameters, "decoder.conv1");
        h = Middle(model, h, parameters, "decoder");

        // Three residual blocks per level then an upsample on all but the
        // last, numbered straight through as the reference's Sequential is.
        int index = 0;
        int levels = WidthMultipliers.Length;
        for (int level = 0; level < levels; level++)
        {
            for (int i = 0; i < ResnetsPerLevel + 1; i++)
            {
                h = ResidualBlock(model, h, parameters, $"decoder.upsamples.{index++}");
            }
            if (level != levels - 1)
            {
                h = Conv(model, Layers.UpsampleNearest(h), parameters,
                    $"decoder.upsamples.{index++}.resample.1");
            }
        }

        h = Layers.Silu(ChannelRmsNorm(model, h, parameters["decoder.head.0.gamma"]));
        DmlExpression image = Conv(model, h, parameters, "decoder.head.2");
        return model.Compile(new[] { image });
    }

    /// <summary>Compile an encoder for a height by width image: one [1, 3, H,
    /// W] image in, the latent distribution's mean out, on the VAE's own scale.</summary>
    public static ModelBuilder Encoder(DmlDevice device,
        IReadOnlyDictionary<string, HostTensor> parameters, int height, int width)
    {
        var model = new ModelBuilder(device);
        DmlExpression image = model.Placeholder(new[] { 1, 3, height, width });
        DmlExpression h = Conv(model, image, parameters, "encoder.conv1");

        int index = 0;
        int levels = WidthMultipliers.Length;
        for (int level = 0; level < levels; level++)
        {
            for (int i = 0; i < ResnetsPerLevel; i++)
            {
                h = ResidualBlock(model, h, parameters, $"encoder.downsamples.{index++}");
            }
            if (level != levels - 1)
            {
                // ZeroPad2d(0, 1, 0, 1) then a stride-2 convolution with no
                // padding of its own: the right and bottom edges only.
                h = Conv(model, h, parameters, $"encoder.downsamples.{index++}.resample.1",
                    stride: 2, padding: 0, endPadding: 1);
            }
        }

        h = Middle(model, h, parameters, "encoder");
        h = Layers.Silu(ChannelRmsNorm(model, h, parameters["encoder.head.0.gamma"]));
        h = Conv(model, h, parameters, "encoder.head.2");
        DmlExpression moments = Conv(model, h, parameters, "conv1", padding: 0);

        uint[] shape = moments.Shape;
        DmlExpression mean = DmlOps.Slice(moments,
            offsets: new[] { 0, 0, 0, 0 },
            sizes: new[] { 1, LatentChannels, (int)shape[2], (int)shape[3] },
            strides: new[] { 1, 1, 1, 1 });
        return model.Compile(new[] { mean });
    }
}
