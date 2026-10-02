using NaiveDiffusion.Dml;
using NaiveDiffusion.Graph;
using NaiveDiffusion.Tensors;

namespace NaiveDiffusion.Vae;

/// <summary>The Wan 2.2 video VAE layout on a single frame, as Qwen-Image 2.1
/// ships it: sixteen pixels a latent cell over five levels instead of eight
/// over four, sixty-four latent channels, four image channels (RGBA), and
/// a residual shortcut beside every level — an average-pool on the way
/// down, a duplicate-and-shuffle on the way up — that the 2.1 layout does
/// not have. The blocks themselves are Wan's (<see cref="WanVae"/>): the
/// channel RMS norm, SiLU, the causal 3-D convolutions cut to one frame,
/// the single-head attention in the middle.
///
/// This file's convolutions have a temporal kernel of one, so the one-frame
/// cut is the whole kernel, and its temporal resampling convolutions are
/// skipped on a first frame as Wan's are. What the temporal design leaves
/// behind is in the shortcuts, which still reshape time: on the way down
/// the levels that halve time pad a zero frame in front and average it in,
/// so every other output channel of the shortcut is zero; on the way up the
/// levels that double time keep the second of the two frames they make,
/// which for a level that halves the width is the odd input channels. Both
/// are what the reference computes for one frame and both are spelled out
/// in <see cref="Shortcut"/> and <see cref="UpShortcut"/>.
///
/// The decoder returns the three colour channels; the alpha the model also
/// makes is cut in the graph until the image path carries four.</summary>
public static class Wan22Vae
{
    public const int LatentChannels = 64;
    public const int ImageChannels = 4;
    private static readonly int[] WidthMultipliers = { 1, 2, 4, 8, 8 };
    private const int ResnetsPerLevel = 2;

    /// <summary>Which levels, counted from the image, also halved time in
    /// the video model. The decoder mirrors it from the latent.</summary>
    private static readonly bool[] TemporalLevels = { false, true, true, true };

    /// <summary>Qwen-Image 2.1's latent as ComfyUI's latent_formats.QwenImage21
    /// has it: sixty-four channels, sixteen pixels a cell, standardized
    /// channel by channel, with the RGB projection fitted for previews.</summary>
    public static readonly LatentSpace Latent = new StandardizedLatentSpace(
        Channels: LatentChannels,
        ScaleFactor: 16,
        Means: new[]
        {
            0.5126f, 0.7721f, -0.0631f, 1.3506f, -0.7855f, -2.1025f, -0.3458f, 1.3722f,
            1.8873f, -1.7177f, -0.6510f, 0.2732f, 0.7562f, -0.6163f, -1.0277f, 3.8363f,
            2.0210f, 0.0472f, 0.9320f, 2.0087f, 2.4954f, -0.1391f, -1.4249f, 1.8464f,
            -0.5236f, 1.2826f, 3.7046f, -1.3035f, 2.7286f, -1.4518f, -1.9036f, -1.9955f,
            -0.0342f, -1.0265f, -0.7636f, 3.0555f, 0.0746f, -3.0751f, -0.1076f, 1.7376f,
            -1.0914f, -1.9435f, -0.2784f, -1.3680f, 0.4809f, -0.4433f, 0.3764f, 0.5729f,
            -2.0595f, 1.0960f, -1.3260f, -2.0211f, -5.0179f, 0.5275f, 4.0162f, 1.8505f,
            0.3026f, 1.9373f, 1.4937f, 0.2632f, 0.5547f, -1.7121f, -0.1562f, 0.0304f,
        },
        Deviations: new[]
        {
            3.2001f, 3.2936f, 3.4321f, 3.0091f, 3.1061f, 4.0379f, 4.0705f, 3.7910f,
            3.0785f, 3.6500f, 3.9308f, 3.0904f, 2.8778f, 3.7675f, 3.7320f, 5.0756f,
            3.2864f, 4.0397f, 3.1317f, 4.0443f, 2.9249f, 3.9454f, 3.0988f, 4.2489f,
            3.4896f, 3.8513f, 3.9323f, 3.4719f, 3.7498f, 4.2830f, 3.5694f, 4.2467f,
            3.9037f, 3.2947f, 5.0770f, 3.5075f, 3.2700f, 3.4767f, 2.8063f, 5.1125f,
            3.5327f, 4.7833f, 3.1286f, 4.1819f, 3.8527f, 3.8312f, 3.5605f, 4.3875f,
            3.9624f, 4.0168f, 3.5643f, 4.0550f, 5.5614f, 4.2963f, 4.4080f, 3.4959f,
            3.8747f, 3.7608f, 3.5735f, 3.1490f, 3.7662f, 3.6746f, 3.4563f, 3.8161f,
        },
        PreviewColors: new[]
        {
            new[] { -0.0158f, -0.0115f, -0.0174f }, new[] { 0.0030f, 0.0120f, 0.0027f },
            new[] { 0.0637f, 0.0470f, -0.0127f }, new[] { 0.0360f, 0.0661f, -0.0030f },
            new[] { 0.0159f, 0.0181f, 0.0082f }, new[] { 0.0132f, 0.0326f, 0.0169f },
            new[] { 0.0191f, 0.0261f, 0.0136f }, new[] { -0.0146f, -0.0276f, -0.0361f },
            new[] { 0.0187f, -0.0024f, -0.0072f }, new[] { -0.1059f, -0.0090f, 0.0350f },
            new[] { -0.0195f, -0.0226f, -0.0138f }, new[] { -0.0295f, 0.0024f, -0.0215f },
            new[] { 0.0191f, -0.0393f, -0.0001f }, new[] { -0.0144f, -0.0166f, -0.0272f },
            new[] { 0.0389f, 0.0430f, 0.0445f }, new[] { -0.0153f, -0.0336f, 0.0031f },
            new[] { 0.0339f, 0.0122f, 0.0220f }, new[] { -0.0136f, -0.0078f, -0.0120f },
            new[] { -0.0340f, -0.0282f, -0.0245f }, new[] { -0.0133f, -0.0176f, -0.0133f },
            new[] { 0.0109f, -0.0087f, 0.0096f }, new[] { -0.0010f, 0.0044f, 0.0016f },
            new[] { 0.0301f, 0.0053f, 0.0361f }, new[] { -0.0281f, -0.0205f, -0.0032f },
            new[] { -0.0725f, 0.0002f, 0.0160f }, new[] { -0.0036f, 0.0158f, 0.0807f },
            new[] { 0.0087f, 0.0040f, -0.0053f }, new[] { -0.0260f, 0.0183f, -0.0077f },
            new[] { -0.0039f, -0.0035f, -0.0107f }, new[] { -0.0026f, 0.0172f, 0.0237f },
            new[] { 0.0088f, 0.0078f, 0.0078f }, new[] { -0.0087f, -0.0310f, -0.0122f },
            new[] { -0.0027f, 0.0018f, 0.0094f }, new[] { -0.0064f, 0.0292f, -0.0256f },
            new[] { 0.0594f, 0.1049f, 0.1180f }, new[] { 0.0103f, -0.0103f, -0.0026f },
            new[] { -0.0091f, 0.0025f, -0.0015f }, new[] { 0.0178f, 0.0243f, 0.0292f },
            new[] { -0.0063f, -0.0012f, 0.0202f }, new[] { 0.0452f, 0.0246f, 0.0143f },
            new[] { 0.0149f, 0.0270f, 0.0052f }, new[] { 0.1484f, 0.0801f, 0.0804f },
            new[] { -0.0120f, 0.0040f, 0.0010f }, new[] { 0.0181f, 0.0051f, -0.0021f },
            new[] { 0.0132f, 0.0050f, 0.0019f }, new[] { 0.0291f, 0.0020f, 0.0092f },
            new[] { 0.0066f, -0.0410f, -0.1314f }, new[] { -0.1153f, -0.0629f, -0.0802f },
            new[] { 0.0258f, 0.0378f, 0.0298f }, new[] { 0.0375f, 0.1139f, 0.0468f },
            new[] { -0.0142f, -0.0126f, -0.0276f }, new[] { 0.0339f, 0.0153f, 0.0138f },
            new[] { 0.0346f, 0.0211f, 0.0267f }, new[] { 0.0369f, -0.0431f, -0.0993f },
            new[] { -0.0052f, -0.0092f, 0.0056f }, new[] { -0.0279f, 0.0410f, -0.0357f },
            new[] { 0.0036f, 0.0017f, -0.0083f }, new[] { -0.0441f, -0.0367f, -0.0454f },
            new[] { -0.0001f, -0.0092f, -0.0001f }, new[] { -0.0222f, -0.0183f, -0.0051f },
            new[] { 0.0039f, 0.0053f, -0.0184f }, new[] { -0.0094f, -0.0075f, -0.0143f },
            new[] { -0.0066f, -0.0088f, -0.0063f }, new[] { 0.0220f, 0.0074f, 0.0100f },
        },
        PreviewBias: new[] { -0.1228f, -0.1869f, -0.3083f });

    private static int Channels(DmlExpression x) => (int)x.Shape[1];

    // --- The shortcuts -------------------------------------------------------

    /// <summary>The down level's shortcut, AvgDown3D on one frame: a 2×2
    /// average pool, then — on a level that halved time — interleaved with
    /// the zero frame it was padded with, so the output's even channels
    /// are zero and its odd ones the pool. A level that does not halve
    /// space is its own shortcut.</summary>
    private static DmlExpression Shortcut(DmlExpression x, int outChannels, bool spatial, bool temporal)
    {
        int channels = Channels(x);
        if (!spatial)
        {
            if (channels != outChannels || temporal)
            {
                throw new InvalidDataException($"a level that keeps {channels} channels cannot make {outChannels}");
            }
            return x;
        }
        DmlExpression pooled = DmlOps.AveragePooling(x, 2);
        if (!temporal)
        {
            if (channels != outChannels)
            {
                throw new InvalidDataException($"a spatial level cannot pool {channels} channels into {outChannels}");
            }
            return pooled;
        }
        if (outChannels != 2 * channels)
        {
            throw new InvalidDataException($"a temporal level makes twice its channels, not {outChannels} from {channels}");
        }
        uint[] shape = pooled.Shape;
        uint[] framed = { shape[0], shape[1], 1, shape[2], shape[3] };
        DmlExpression zeros = DmlOps.Identity(pooled, scale: 0f);
        DmlExpression joined = DmlOps.Join(new[]
        {
            DmlOps.Reinterpret(zeros, framed), DmlOps.Reinterpret(pooled, framed),
        }, axis: 2);
        return DmlOps.Reinterpret(joined, new[] { shape[0], shape[1] * 2, shape[2], shape[3] });
    }

    /// <summary>The up level's shortcut, DupUp3D on one frame kept as its
    /// second output frame. With the width unchanged it is a nearest
    /// upsample; with the width halved it is a nearest upsample of the odd
    /// input channels on a level that doubled time, and on one that did
    /// not, each output channel's even rows from one input channel and its
    /// odd rows from the next.</summary>
    private static DmlExpression UpShortcut(DmlExpression x, int outChannels, bool temporal)
    {
        int channels = Channels(x);
        uint[] shape = x.Shape;
        (uint height, uint width) = (shape[2], shape[3]);
        DmlExpression EveryOther(int first) => DmlOps.Slice(x,
            offsets: new[] { 0, first, 0, 0 },
            sizes: new[] { 1, channels - first, (int)height, (int)width },
            strides: new[] { 1, 2, 1, 1 });
        if (channels == outChannels)
        {
            return Layers.UpsampleNearest(x);
        }
        if (channels != 2 * outChannels)
        {
            throw new InvalidDataException($"an up level halves its channels at most, not {channels} to {outChannels}");
        }
        if (temporal)
        {
            return Layers.UpsampleNearest(EveryOther(1));
        }
        // Each half doubled across, then the two interleaved down the rows.
        uint[] rows = { 1, (uint)outChannels, height, 1, width * 2 };
        DmlExpression even = DmlOps.Reinterpret(DmlOps.Upsample2D(EveryOther(0), 2, 1,
            Vortice.DirectML.InterpolationMode.NearestNeighbor), rows);
        DmlExpression odd = DmlOps.Reinterpret(DmlOps.Upsample2D(EveryOther(1), 2, 1,
            Vortice.DirectML.InterpolationMode.NearestNeighbor), rows);
        DmlExpression joined = DmlOps.Join(new[] { even, odd }, axis: 3);
        return DmlOps.Reinterpret(joined, new[] { 1u, (uint)outChannels, height * 2, width * 2 });
    }

    // --- The graphs --------------------------------------------------------

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
        DmlExpression h = WanVae.Conv(model, z, parameters, "conv2", padding: 0);
        h = WanVae.Conv(model, h, parameters, "decoder.conv1");
        h = WanVae.Middle(model, h, parameters, "decoder");

        // Three residual blocks per level then an upsample on all but the
        // last, with the level's input through the shortcut beside them.
        int levels = WidthMultipliers.Length;
        for (int level = 0; level < levels; level++)
        {
            string prefix = $"decoder.upsamples.{level}.upsamples";
            DmlExpression input = h;
            int index = 0;
            for (int i = 0; i < ResnetsPerLevel + 1; i++)
            {
                h = WanVae.ResidualBlock(model, h, parameters, $"{prefix}.{index++}");
            }
            if (level != levels - 1)
            {
                h = WanVae.Conv(model, Layers.UpsampleNearest(h), parameters,
                    $"{prefix}.{index}.resample.1");
                // The levels count from the latent here, so time is halved
                // where the encoder's mirror level halved it.
                bool temporal = TemporalLevels[levels - 2 - level];
                h += UpShortcut(input, Channels(h), temporal);
            }
        }

        h = Layers.Silu(WanVae.ChannelRmsNorm(model, h, parameters["decoder.head.0.gamma"]));
        DmlExpression image = WanVae.Conv(model, h, parameters, "decoder.head.2");
        if (Channels(image) != ImageChannels)
        {
            throw new InvalidDataException($"the decoder makes {Channels(image)} channels, not {ImageChannels}");
        }
        DmlExpression rgb = DmlOps.Slice(image,
            offsets: new[] { 0, 0, 0, 0 },
            sizes: new[] { 1, 3, height, width },
            strides: new[] { 1, 1, 1, 1 });
        return model.Compile(new[] { rgb });
    }

    /// <summary>Compile an encoder for a height by width image: one [1, 3,
    /// H, W] image in, given to the model over an opaque alpha, the latent
    /// distribution's mean out, on the VAE's own scale.</summary>
    public static ModelBuilder Encoder(DmlDevice device,
        IReadOnlyDictionary<string, HostTensor> parameters, int height, int width)
    {
        var model = new ModelBuilder(device);
        DmlExpression rgb = model.Placeholder(new[] { 1, 3, height, width });
        DmlExpression alpha = DmlOps.Identity(DmlOps.Slice(rgb,
            offsets: new[] { 0, 0, 0, 0 },
            sizes: new[] { 1, 1, height, width },
            strides: new[] { 1, 1, 1, 1 }), scale: 0f, bias: 1f);
        DmlExpression image = DmlOps.Join(new[] { rgb, alpha }, axis: 1);
        DmlExpression h = WanVae.Conv(model, image, parameters, "encoder.conv1");

        int levels = WidthMultipliers.Length;
        for (int level = 0; level < levels; level++)
        {
            string prefix = $"encoder.downsamples.{level}.downsamples";
            DmlExpression input = h;
            int index = 0;
            for (int i = 0; i < ResnetsPerLevel; i++)
            {
                h = WanVae.ResidualBlock(model, h, parameters, $"{prefix}.{index++}");
            }
            bool spatial = level != levels - 1;
            if (spatial)
            {
                // ZeroPad2d(0, 1, 0, 1) then a stride-2 convolution with no
                // padding of its own: the right and bottom edges only.
                h = WanVae.Conv(model, h, parameters, $"{prefix}.{index}.resample.1",
                    stride: 2, padding: 0, endPadding: 1);
            }
            bool temporal = level < TemporalLevels.Length && TemporalLevels[level];
            h += Shortcut(input, Channels(h), spatial, temporal);
        }

        h = WanVae.Middle(model, h, parameters, "encoder");
        h = Layers.Silu(WanVae.ChannelRmsNorm(model, h, parameters["encoder.head.0.gamma"]));
        h = WanVae.Conv(model, h, parameters, "encoder.head.2");
        DmlExpression moments = WanVae.Conv(model, h, parameters, "conv1", padding: 0);

        uint[] shape = moments.Shape;
        DmlExpression mean = DmlOps.Slice(moments,
            offsets: new[] { 0, 0, 0, 0 },
            sizes: new[] { 1, LatentChannels, (int)shape[2], (int)shape[3] },
            strides: new[] { 1, 1, 1, 1 });
        return model.Compile(new[] { mean });
    }
}
