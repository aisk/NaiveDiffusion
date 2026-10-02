using NaiveDiffusion.Dml;
using NaiveDiffusion.Tensors;

namespace NaiveDiffusion.Graph;

/// <summary>The pieces a diffusion transformer's blocks are made of that
/// no family owns: the sinusoid of the noise level, the per-head norm on
/// queries and keys, rotary positions in the rotate-half layout, and the
/// adaLN modulation of a LayerNorm. Each takes the head layout as numbers
/// so that a family reads them off its weights rather than fixing them;
/// each computes at the precision it is told, since the blocks run wide
/// (see <see cref="Models.Anima.AnimaDit"/>) while the weights stay
/// narrow.</summary>
public static class Transformer
{
    /// <summary>The sinusoidal embedding of a time: half cosines then half
    /// sines of t × 10000^(−i/half), i from 0, computed in float as the
    /// references do. <paramref name="time"/> is the model's own time —
    /// the level itself for Cosmos, a thousand times it for Flux.</summary>
    public static float[] Sinusoid(float time, int width)
    {
        int half = width / 2;
        var embedding = new float[width];
        for (int i = 0; i < half; i++)
        {
            float frequency = MathF.Exp(-MathF.Log(10000f) * i / half);
            float angle = time * frequency;
            embedding[i] = MathF.Cos(angle);
            embedding[half + i] = MathF.Sin(angle);
        }
        return embedding;
    }

    /// <summary>Each head's query or key normalized on its own: the packed
    /// [1, 1, tokens, width] viewed as [1, tokens, heads, dim], which the
    /// result stays.</summary>
    public static DmlExpression HeadNorm(ModelBuilder model, DmlExpression x, HostTensor weight,
        int heads, int headDim, float epsilon)
    {
        uint tokens = x.Shape[2];
        DmlExpression view = DmlOps.Reinterpret(x, new[] { 1u, tokens, (uint)heads, (uint)headDim });
        return Layers.RmsNorm(model, view, weight, epsilon);
    }

    /// <summary>Rotary positions over [1, tokens, heads, dim] in the
    /// rotate-half layout: the first half of each head's features turned
    /// against the second by the [1, tokens, 1, dim/2] tables, at single
    /// precision as the references apply them, then back at
    /// <paramref name="compute"/>.</summary>
    public static DmlExpression Rotate(DmlExpression heads, DmlExpression cos, DmlExpression sin,
        int headCount, int headDim, HostDataType compute)
    {
        uint[] shape = heads.Shape;
        int half = headDim / 2;
        DmlExpression wide = Layers.Cast(heads, HostDataType.Float32);
        DmlExpression Half(int offset) => DmlOps.Slice(wide, new[] { 0, 0, 0, offset },
            new[] { 1, (int)shape[1], headCount, half }, new[] { 1, 1, 1, 1 });
        uint[] halfShape = { 1, shape[1], (uint)headCount, (uint)half };
        DmlExpression c = Layers.Broadcast(cos, halfShape);
        DmlExpression s = Layers.Broadcast(sin, halfShape);
        DmlExpression low = Half(0), high = Half(half);
        DmlExpression rotated = DmlOps.Join(new[] { low * c - high * s, high * c + low * s }, axis: 3);
        return Layers.Cast(rotated, compute);
    }

    /// <summary>[1, tokens, heads, dim] back to the packed [1, 1, tokens, width].</summary>
    public static DmlExpression Flat(DmlExpression heads, int width) =>
        DmlOps.Reinterpret(heads, new[] { 1u, 1u, heads.Shape[1], (uint)width });

    /// <summary>LayerNorm without an affine, then (1 + scale), at the
    /// precision the block computes at.</summary>
    public static DmlExpression Modulate(DmlExpression x, DmlExpression scale, float epsilon,
        HostDataType compute) =>
        Layers.Cast(Layers.LayerNorm(x, epsilon) * (scale + 1.0f), compute);

    /// <summary>LayerNorm without an affine, then (1 + scale) and shift, at
    /// the precision the block computes at.</summary>
    public static DmlExpression Modulate(DmlExpression x, DmlExpression shift, DmlExpression scale,
        float epsilon, HostDataType compute) =>
        Layers.Cast(Layers.LayerNorm(x, epsilon) * (scale + 1.0f) + shift, compute);
}
