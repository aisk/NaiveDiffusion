using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace NaiveDiffusion.Tensors;

/// <summary>The handful of operators a transformer needs, on the CPU.
///
/// This exists for the text towers — SDXL's two CLIP towers, Anima's Qwen3
/// and its adapter — which run here rather than on the GPU: their weights
/// are gigabytes but their arithmetic is a fraction of a second, so putting
/// them on the GPU spent video memory the diffusion model needs to buy back
/// almost no time. See <see cref="Text.Clip.ClipTextEncoder"/> and
/// <see cref="Text.Qwen.Qwen3TextModel"/>.
///
/// Everything is row-major and every matrix multiply has the same shape:
/// nn.Linear stores its weight as [out, in], so both operands of the inner loop
/// are contiguous rows and the kernel is a plain dot product. That is the whole
/// reason this is fast enough to be worth doing — on an 8-core Ryzen the linear
/// layers run at 130-180 GFLOPS with nothing cleverer than one thread per output
/// column.
///
/// <b>Matrices stay at the width the checkpoint stored them</b> and are widened
/// one row at a time, inside the loop that is about to use that row. Widening a
/// whole tower up front would double what it costs to hold — 1.52 GiB of half
/// precision becomes 3.05 GiB of float — while the conversion itself is free in
/// practice: a row is converted once and then reused by every token of the
/// prompt, so it amortizes to about one percent, and float32 represents
/// every float16 and bfloat16 value exactly, so the result is bit-identical
/// either way (<see cref="HostTensor.WidenRow"/>).</summary>
public static class CpuMath
{
    /// <summary>y[m, o] = dot(x[m, :], weight[o, :]) + bias[o].</summary>
    public static void Linear(float[] x, int rows, int inner, HostTensor weight, float[]? bias,
        float[] y, int outer)
    {
        if (weight.DataType == HostDataType.Float32)
        {
            Parallel.For(0, outer, o =>
            {
                ReadOnlySpan<float> column =
                    MemoryMarshal.Cast<byte, float>(weight.Data.Span.Slice(o * inner * 4, inner * 4));
                Accumulate(x, rows, inner, column, bias is null ? 0 : bias[o], y, outer, o);
            });
            return;
        }

        // One scratch row per worker thread, not per column.
        Parallel.For(0, outer, () => new float[inner], (o, _, column) =>
        {
            weight.WidenRow(o, inner, column);
            Accumulate(x, rows, inner, column, bias is null ? 0 : bias[o], y, outer, o);
            return column;
        }, _ => { });
    }

    private static void Accumulate(float[] x, int rows, int inner, ReadOnlySpan<float> column,
        float shift, float[] y, int outer, int o)
    {
        for (int m = 0; m < rows; m++)
        {
            y[m * outer + o] = Dot(x.AsSpan(m * inner, inner), column) + shift;
        }
    }

    /// <summary>Copy one row of a packed matrix into <paramref name="y"/>,
    /// widening it — an embedding lookup.</summary>
    public static void CopyRow(HostTensor matrix, long row, int width, float[] y, int offset)
    {
        matrix.WidenRow(row, width, y.AsSpan(offset, width));
    }

    /// <summary>Add one row of a packed matrix to <paramref name="y"/>.</summary>
    public static void AddRow(HostTensor matrix, long row, int width, float[] y, int offset)
    {
        // On the stack for the widths the towers have; a wider row goes to
        // the heap rather than risk the stack.
        Span<float> scratch = stackalloc float[width <= 4096 ? width : 0];
        float[]? heap = width <= 4096 ? null : new float[width];
        Span<float> widened = heap ?? scratch;

        matrix.WidenRow(row, width, widened);
        Span<float> target = y.AsSpan(offset, width);
        for (int i = 0; i < width; i++)
        {
            target[i] += widened[i];
        }
    }

    /// <summary>Normalize each row to zero mean and unit variance, then scale
    /// and shift per feature.</summary>
    public static void LayerNorm(float[] x, int rows, int width, float[] weight, float[] bias,
        float[] y, float epsilon)
    {
        Parallel.For(0, rows, m =>
        {
            Span<float> source = x.AsSpan(m * width, width);
            Span<float> target = y.AsSpan(m * width, width);

            // Two passes, both in double: the variance of a nearly-constant row
            // is a small difference of large numbers, and float32 loses it.
            double sum = 0;
            for (int i = 0; i < width; i++)
            {
                sum += source[i];
            }
            double mean = sum / width;

            double square = 0;
            for (int i = 0; i < width; i++)
            {
                double centred = source[i] - mean;
                square += centred * centred;
            }
            double scale = 1.0 / Math.Sqrt(square / width + epsilon);

            for (int i = 0; i < width; i++)
            {
                target[i] = (float)((source[i] - mean) * scale) * weight[i] + bias[i];
            }
        });
    }

    /// <summary>Scale each row to unit root-mean-square, then per feature —
    /// RMSNorm, the norm every Llama-shaped model and every DiT uses in place
    /// of LayerNorm. Rows may be heads: a [tokens, heads, dim] tensor is
    /// normalized per head by calling this with tokens × heads rows of dim.</summary>
    public static void RmsNorm(float[] x, int rows, int width, float[] weight, float[] y,
        float epsilon)
    {
        Parallel.For(0, rows, m =>
        {
            Span<float> source = x.AsSpan(m * width, width);
            Span<float> target = y.AsSpan(m * width, width);
            double square = 0;
            for (int i = 0; i < width; i++)
            {
                square += (double)source[i] * source[i];
            }
            float scale = (float)(1.0 / Math.Sqrt(square / width + epsilon));
            for (int i = 0; i < width; i++)
            {
                target[i] = source[i] * scale * weight[i];
            }
        });
    }

    /// <summary>Rotary position embedding in the rotate-half layout — the one
    /// Llama, Qwen and the T5-style adapters share: feature i of each head
    /// pairs with feature i + dim/2, and the pair turns by the angle
    /// position × frequency_i. <paramref name="cos"/> and <paramref name="sin"/>
    /// hold one row per token of dim/2 angles each. In place, over
    /// [rows, heads, dim].</summary>
    public static void Rope(float[] x, int rows, int heads, int dim, float[] cos, float[] sin)
    {
        int half = dim / 2;
        Parallel.For(0, rows, m =>
        {
            ReadOnlySpan<float> c = cos.AsSpan(m * half, half);
            ReadOnlySpan<float> s = sin.AsSpan(m * half, half);
            for (int head = 0; head < heads; head++)
            {
                Span<float> v = x.AsSpan((m * heads + head) * dim, dim);
                for (int i = 0; i < half; i++)
                {
                    float low = v[i];
                    float high = v[i + half];
                    v[i] = low * c[i] - high * s[i];
                    v[i + half] = high * c[i] + low * s[i];
                }
            }
        });
    }

    /// <summary>The angle tables <see cref="Rope"/> reads: for positions 0
    /// to <paramref name="rows"/> − 1, frequency i is θ^(−2i/dim).</summary>
    public static (float[] Cos, float[] Sin) RopeTables(int rows, int dim, double theta)
    {
        int half = dim / 2;
        var cos = new float[rows * half];
        var sin = new float[rows * half];
        for (int i = 0; i < half; i++)
        {
            // In float, as torch computes the inverse frequencies.
            float frequency = 1.0f / MathF.Pow((float)theta, 2.0f * i / dim);
            for (int m = 0; m < rows; m++)
            {
                float angle = m * frequency;
                cos[m * half + i] = MathF.Cos(angle);
                sin[m * half + i] = MathF.Sin(angle);
            }
        }
        return (cos, sin);
    }

    /// <summary>Multi-head attention between projected sequences: the query
    /// rows attend over the key rows, with <paramref name="causal"/> letting
    /// row i see keys 0 to i only — for that the two sequences are the same
    /// one. Fewer key heads than query heads is grouped-query attention:
    /// query head h reads key head h / (queryHeads / keyHeads). Query is
    /// laid out [rows, queryHeads × dim], key and value [keyRows, keyHeads ×
    /// dim], the output like the query.</summary>
    public static void Attention(float[] query, float[] key, float[] value, int rows,
        int keyRows, int queryHeads, int keyHeads, int dim, bool causal, float[] y)
    {
        int queryWidth = queryHeads * dim;
        int keyWidth = keyHeads * dim;
        int group = queryHeads / keyHeads;
        float scale = 1.0f / MathF.Sqrt(dim);

        Parallel.For(0, queryHeads * rows, work =>
        {
            int head = work / rows;
            int i = work % rows;
            int queryOffset = head * dim;
            int keyOffset = head / group * dim;
            int visible = causal ? Math.Min(i + 1, keyRows) : keyRows;
            Span<float> scores = visible <= 4096 ? stackalloc float[visible] : new float[visible];

            ReadOnlySpan<float> q = query.AsSpan(i * queryWidth + queryOffset, dim);
            float largest = float.NegativeInfinity;
            for (int j = 0; j < visible; j++)
            {
                float score = Dot(q, key.AsSpan(j * keyWidth + keyOffset, dim)) * scale;
                scores[j] = score;
                largest = MathF.Max(largest, score);
            }
            float total = 0;
            for (int j = 0; j < visible; j++)
            {
                scores[j] = MathF.Exp(scores[j] - largest);
                total += scores[j];
            }
            float normalize = 1.0f / total;

            Span<float> target = y.AsSpan(i * queryWidth + queryOffset, dim);
            target.Clear();
            for (int j = 0; j < visible; j++)
            {
                float weight = scores[j] * normalize;
                ReadOnlySpan<float> v = value.AsSpan(j * keyWidth + keyOffset, dim);
                for (int d = 0; d < dim; d++)
                {
                    target[d] += weight * v[d];
                }
            }
        });
    }

    /// <summary>x *= y, elementwise.</summary>
    public static void Multiply(float[] x, float[] y, int count)
    {
        for (int i = 0; i < count; i++)
        {
            x[i] *= y[i];
        }
    }

    /// <summary>x * sigmoid(x), in place.</summary>
    public static void Silu(float[] x, int count)
    {
        for (int i = 0; i < count; i++)
        {
            x[i] *= 1.0f / (1.0f + MathF.Exp(-x[i]));
        }
    }

    /// <summary>x += y, elementwise.</summary>
    public static void Add(float[] x, float[] y, int count)
    {
        for (int i = 0; i < count; i++)
        {
            x[i] += y[i];
        }
    }

    /// <summary>CLIP ViT-L's activation: cheaper than the real thing and what
    /// the model was trained with.</summary>
    public static void QuickGelu(float[] x, int count)
    {
        for (int i = 0; i < count; i++)
        {
            x[i] *= 1.0f / (1.0f + MathF.Exp(-1.702f * x[i]));
        }
    }

    /// <summary>The exact GELU, which is what OpenCLIP bigG and DirectML's
    /// ACTIVATION_GELU both mean by the name.</summary>
    public static void Gelu(float[] x, int count)
    {
        for (int i = 0; i < count; i++)
        {
            x[i] = 0.5f * x[i] * (1.0f + Erf(x[i] * 0.70710678f));
        }
    }

    /// <summary>Abramowitz and Stegun 7.1.26 — maximum error 1.5e-7, which is
    /// float32's own resolution.</summary>
    private static float Erf(float x)
    {
        float sign = MathF.Sign(x);
        x = MathF.Abs(x);
        float t = 1.0f / (1.0f + 0.3275911f * x);
        float series = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t
            - 0.284496736f) * t + 0.254829592f) * t;
        return sign * (1.0f - series * MathF.Exp(-x * x));
    }

    /// <summary>The inner loop everything above is built from. Four accumulators
    /// because a single dependency chain stalls on the multiply-add latency.</summary>
    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int width = Vector<float>.Count;
        Vector<float> s0 = Vector<float>.Zero, s1 = Vector<float>.Zero;
        Vector<float> s2 = Vector<float>.Zero, s3 = Vector<float>.Zero;

        int i = 0;
        for (; i + 4 * width <= a.Length; i += 4 * width)
        {
            s0 += new Vector<float>(a.Slice(i, width)) * new Vector<float>(b.Slice(i, width));
            s1 += new Vector<float>(a.Slice(i + width, width)) *
                  new Vector<float>(b.Slice(i + width, width));
            s2 += new Vector<float>(a.Slice(i + 2 * width, width)) *
                  new Vector<float>(b.Slice(i + 2 * width, width));
            s3 += new Vector<float>(a.Slice(i + 3 * width, width)) *
                  new Vector<float>(b.Slice(i + 3 * width, width));
        }

        float total = Vector.Sum(s0 + s1 + s2 + s3);
        for (; i < a.Length; i++)
        {
            total += a[i] * b[i];
        }
        return total;
    }
}
