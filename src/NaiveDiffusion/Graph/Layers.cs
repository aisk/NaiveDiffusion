using NaiveDiffusion.Dml;
using NaiveDiffusion.Tensors;
using Vortice.DirectML;

namespace NaiveDiffusion.Graph;

/// <summary>The handful of neural-network layers the models are built out of,
/// each a few DirectML operators. Broadcasting and transposing are stride tricks: a
/// stride of 0 repeats an axis, and reading NCHW through token-shaped strides
/// is the reshape attention needs, with no copy either way.</summary>
public static class Layers
{
    /// <summary>View <paramref name="expression"/> as <paramref name="shape"/>,
    /// repeating any axis whose extent is 1.</summary>
    public static DmlExpression Broadcast(DmlExpression expression, uint[] shape)
    {
        uint[] source = expression.Shape;
        if (source.SequenceEqual(shape))
        {
            return expression;
        }
        if (source.Length != shape.Length)
        {
            throw new ArgumentException(
                $"cannot broadcast [{string.Join(", ", source)}] to [{string.Join(", ", shape)}]: rank differs");
        }

        var packed = new uint[source.Length];
        packed[^1] = 1;
        for (int i = source.Length - 2; i >= 0; i--)
        {
            packed[i] = packed[i + 1] * source[i + 1];
        }

        var strides = new uint[source.Length];
        for (int i = 0; i < source.Length; i++)
        {
            if (source[i] == shape[i])
            {
                strides[i] = packed[i];
            }
            else if (source[i] == 1)
            {
                strides[i] = 0;
            }
            else
            {
                throw new ArgumentException(
                    $"cannot broadcast [{string.Join(", ", source)}] to [{string.Join(", ", shape)}]");
            }
        }
        return DmlOps.Reinterpret(expression, shape, strides);
    }

    /// <summary>View [1, C, H, W] as [1, 1, H*W, C] — a transpose, not a copy.</summary>
    public static DmlExpression ToTokens(DmlExpression expression)
    {
        uint[] s = expression.Shape;
        (uint n, uint c, uint h, uint w) = (s[0], s[1], s[2], s[3]);
        return DmlOps.Reinterpret(expression,
            new[] { n, 1u, h * w, c },
            new[] { c * h * w, c * h * w, 1u, h * w });
    }

    /// <summary>The inverse of <see cref="ToTokens"/>.</summary>
    public static DmlExpression ToImage(DmlExpression expression, uint height, uint width)
    {
        uint[] s = expression.Shape;
        (uint n, uint tokens, uint c) = (s[0], s[2], s[3]);
        if (tokens != height * width)
        {
            throw new ArgumentException($"{tokens} tokens do not fill {height}x{width}");
        }
        return DmlOps.Reinterpret(expression,
            new[] { n, c, height, width },
            new[] { c * tokens, 1u, width * c, c });
    }

    /// <summary>View [1, 1, 1, C] as [1, C, 1, 1], ready to broadcast over an image.</summary>
    public static DmlExpression ToChannels(DmlExpression expression)
    {
        uint[] s = expression.Shape;
        return DmlOps.Reinterpret(expression, new[] { s[0], s[3], 1u, 1u });
    }

    /// <summary>x * sigmoid(x), the activation the VAE calls "swish".</summary>
    public static DmlExpression Silu(DmlExpression x) => DmlOps.ActivationSwish(x);

    /// <summary>The precision the layers' constants take: the activation's.
    /// A model stores its weights at one width and computes at another only
    /// when it says so (<see cref="Linear"/> widens in the graph), but the
    /// small affine constants — biases, norm scales — are always made at the
    /// width of the tensor they meet, so a single-precision activation in a
    /// half-precision builder gets single-precision constants and the
    /// elementwise operators never see two widths.</summary>
    private static HostDataType ConstantType(DmlExpression x) =>
        DmlTensorDesc.ToHostDataType(x.Desc.DataType);

    /// <summary>A [1, C, 1, 1] constant for a per-channel affine over an image.</summary>
    private static DmlExpression ChannelConstant(ModelBuilder model, HostTensor values,
        DmlExpression x) =>
        model.Constant(values, new[] { 1, values.Shape[0], 1, 1 }, ConstantType(x));

    /// <summary>A [1, …, 1, C] constant for a per-feature affine over tokens.</summary>
    private static DmlExpression FeatureConstant(ModelBuilder model, HostTensor values,
        DmlExpression x)
    {
        var shape = new int[x.Shape.Length];
        Array.Fill(shape, 1);
        shape[^1] = (int)x.Shape[^1];
        return model.Constant(values, shape, ConstantType(x));
    }

    /// <summary>A 2-D convolution from torch-ordered [out, in, kh, kw] weights.</summary>
    public static DmlExpression Conv2d(ModelBuilder model, DmlExpression x,
        HostTensor weight, HostTensor bias, int stride = 1, int padding = 1,
        int? endPadding = null)
    {
        DmlExpression filters = DmlOps.Cast(
            model.Weight(weight, weight.Shape, rows: weight.Shape[0]), x.Desc.DataType);
        DmlExpression biases = ChannelConstant(model, bias, x);
        int end = endPadding ?? padding;
        return DmlOps.Convolution(x, filters, biases,
            strides: new[] { stride, stride },
            startPadding: new[] { padding, padding },
            endPadding: new[] { end, end });
    }

    /// <summary>A dense layer over the last axis, from torch-ordered [out, in]
    /// weights. Attention projections carry no bias, so it is optional.</summary>
    public static DmlExpression Linear(ModelBuilder model, DmlExpression x,
        HostTensor weight, HostTensor? bias = null)
    {
        int outFeatures = weight.Shape[0];
        int inFeatures = weight.Shape[1];
        DmlExpression weights = model.Weight(weight, new[] { 1, 1, outFeatures, inFeatures },
            rows: outFeatures);
        // The product runs at the activation's precision: weights stored
        // narrower than the activations are widened in the graph.
        weights = DmlOps.Cast(weights, x.Desc.DataType);

        // A gemm wants both operands to agree on the batch axes, and a weight
        // has none of its own, so it is repeated across the batch at stride 0.
        uint batch = x.Shape[0];
        if (batch != 1)
        {
            weights = Broadcast(weights, new[] { batch, 1u, (uint)outFeatures, (uint)inFeatures });
        }
        if (bias is null)
        {
            return DmlOps.Gemm(x, weights, transB: MatrixTransform.Transpose);
        }

        DmlExpression biases = model.Constant(bias, new[] { 1, 1, 1, outFeatures },
            ConstantType(x));
        uint[] shape = x.Shape.ToArray();
        shape[^1] = (uint)outFeatures;
        return DmlOps.Gemm(x, weights, Broadcast(biases, shape),
            transB: MatrixTransform.Transpose);
    }

    /// <summary>GroupNorm over [1, C, H, W]: mean-variance normalization over a
    /// regrouped view, with the per-channel affine applied separately — DirectML
    /// wants scale and bias to be 1 along every normalized axis, and the channel
    /// axis is normalized here.</summary>
    public static DmlExpression GroupNorm(ModelBuilder model, DmlExpression x,
        HostTensor weight, HostTensor bias, int groups, float epsilon)
    {
        uint[] shape = x.Shape;
        (uint n, uint c, uint h, uint w) = (shape[0], shape[1], shape[2], shape[3]);
        if (c % groups != 0)
        {
            throw new ArgumentException($"{c} channels do not divide into {groups} groups");
        }

        DmlExpression grouped = DmlOps.Reinterpret(x,
            new[] { n, (uint)groups, c / (uint)groups, h * w });
        DmlExpression normalized = DmlOps.MeanVarianceNormalization(grouped,
            new[] { 2, 3 }, epsilon);
        normalized = DmlOps.Reinterpret(normalized, shape);

        DmlExpression scale = Broadcast(ChannelConstant(model, weight, x), shape);
        DmlExpression shift = Broadcast(ChannelConstant(model, bias, x), shape);
        return normalized * scale + shift;
    }

    /// <summary>LayerNorm over the last axis of a token tensor, with a
    /// per-feature affine.</summary>
    public static DmlExpression LayerNorm(ModelBuilder model, DmlExpression x,
        HostTensor weight, HostTensor bias, float epsilon)
    {
        uint[] shape = x.Shape;
        DmlExpression normalized = LayerNorm(x, epsilon);
        DmlExpression scale = Broadcast(FeatureConstant(model, weight, x), shape);
        DmlExpression shift = Broadcast(FeatureConstant(model, bias, x), shape);
        return normalized * scale + shift;
    }

    /// <summary>RMSNorm over the last axis: x / sqrt(mean(x²) + ε), times a
    /// per-feature weight when one is given — one mean-variance normalization
    /// that leaves the mean out, the weight as its scale. At half precision
    /// the weight is multiplied in after the operator instead: with the
    /// scale inside, the first dispatch a process makes of that shader came
    /// back rounded differently from every later one on an AMD driver — a
    /// quarter of the values a last bit lower — so the first forward pass
    /// after start-up was not the pass the same inputs gave from then on.
    /// The product outside is those later values to the bit, first time
    /// included.</summary>
    public static DmlExpression RmsNorm(ModelBuilder model, DmlExpression x,
        HostTensor? weight, float epsilon)
    {
        int[] axes = { x.Shape.Length - 1 };
        if (weight is null)
        {
            return DmlOps.MeanVarianceNormalization2(x, axes, false, epsilon);
        }
        DmlExpression scale = FeatureConstant(model, weight, x);
        return x.Desc.DataType == TensorDataType.Float16
            ? DmlOps.MeanVarianceNormalization2(x, axes, false, epsilon) * Broadcast(scale, x.Shape)
            : DmlOps.MeanVarianceNormalization2(x, axes, false, epsilon, scale);
    }

    /// <summary>LayerNorm over the last axis without an affine: what a DiT
    /// modulates with its own shift and scale.</summary>
    public static DmlExpression LayerNorm(DmlExpression x, float epsilon) =>
        DmlOps.MeanVarianceNormalization(x, new[] { x.Shape.Length - 1 }, epsilon);

    /// <summary>The same values at another width — a residual stream kept at
    /// single precision beside half-precision weights, and back.</summary>
    public static DmlExpression Cast(DmlExpression x, HostDataType dataType) =>
        DmlOps.Cast(x, DmlTensorDesc.ToDataType(dataType));

    /// <summary>Nearest-neighbour upsampling, the only interpolation the VAEs use.</summary>
    public static DmlExpression UpsampleNearest(DmlExpression x, uint scale = 2) =>
        DmlOps.Upsample2D(x, scale, scale, InterpolationMode.NearestNeighbor);

    /// <summary>Trim an image tensor to height by width, from the top left.
    /// Cropping a nearest-neighbour 2x upsample this way picks the same source
    /// pixel as a nearest-neighbour resize to the target would.</summary>
    public static DmlExpression CropTo(DmlExpression x, uint height, uint width)
    {
        uint[] s = x.Shape;
        if (s[2] == height && s[3] == width)
        {
            return x;
        }
        return DmlOps.Slice(x,
            offsets: new[] { 0, 0, 0, 0 },
            sizes: new[] { (int)s[0], (int)s[1], (int)height, (int)width },
            strides: new[] { 1, 1, 1, 1 });
    }

    /// <summary>Scaled dot-product attention over already-projected token
    /// tensors, as one operator — the scores never land in memory. The UNet
    /// and Anima attend unmasked; the text towers, with their causal mask,
    /// run on the CPU (see <see cref="Tensors.CpuMath"/>). A single-stream
    /// transformer that carries its text tokens beside the image's masks in
    /// the graph, through <see cref="Attend(DmlExpression, DmlExpression,
    /// DmlExpression, int, DmlExpression, MultiheadAttentionMaskType)"/>.</summary>
    public static DmlExpression Attend(DmlExpression query, DmlExpression key,
        DmlExpression value, int heads)
    {
        int dim = (int)(query.Shape[^1] / (uint)heads);
        return DmlOps.MultiheadAttention(query, key, value, heads, 1.0f / MathF.Sqrt(dim));
    }

    /// <summary>Attention under a mask: an int32 [1, heads, tokens, keys] of
    /// ones where a query may read a key for
    /// <see cref="MultiheadAttentionMaskType.Boolean"/> (a [1, 1, tokens,
    /// keys] table <see cref="Broadcast"/> over the heads), or an int32
    /// [1, 1, 1, 1] count of the real leading keys for
    /// <see cref="MultiheadAttentionMaskType.KeySequenceLength"/>.</summary>
    public static DmlExpression Attend(DmlExpression query, DmlExpression key,
        DmlExpression value, int heads, DmlExpression mask, MultiheadAttentionMaskType maskType)
    {
        int dim = (int)(query.Shape[^1] / (uint)heads);
        return DmlOps.MultiheadAttention(query, key, value, heads, 1.0f / MathF.Sqrt(dim),
            mask, maskType);
    }
}
