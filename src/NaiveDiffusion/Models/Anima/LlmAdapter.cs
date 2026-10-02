using NaiveDiffusion.Tensors;
using NaiveDiffusion.Text;

namespace NaiveDiffusion.Models.Anima;

/// <summary>Anima's bridge from the language model to the transformer: a
/// six-block decoder that embeds the prompt's T5 ids, attends over them, and
/// cross-attends to the Qwen3 hidden states, so that the transformer — whose
/// cross-attention was trained on T5-shaped context — reads a Qwen3 through
/// it. Each block is self-attention, cross-attention and a GELU MLP, all
/// pre-normed with RMSNorm, every query and key head normed and given rotary
/// positions (the keys of the cross-attention by their position in the Qwen
/// sequence); the output goes through a projection and a final norm. Run on
/// the CPU, as the text encoder it follows is, and with the same kernels.</summary>
public static class LlmAdapter
{
    public const int Width = 1024;
    public const int Heads = 16;
    public const int HeadDim = Width / Heads;
    private const int MlpWidth = 4096;
    private const int Blocks = 6;
    private const float NormEpsilon = 1e-6f;
    private const double RopeTheta = 10000;

    /// <summary>The adapter's output, [ids.Length, Width], before any
    /// weighting or padding.</summary>
    /// <param name="ids">The T5 ids, end marker included.</param>
    /// <param name="context">The Qwen3 hidden states, [contextRows, Width].</param>
    public static float[] Forward(TowerWeights parameters, int[] ids, float[] context,
        int contextRows)
    {
        int rows = ids.Length;
        HostTensor table = parameters.Matrix("embed.weight");
        var x = new float[rows * Width];
        for (int m = 0; m < rows; m++)
        {
            CpuMath.CopyRow(table, ids[m], Width, x, m * Width);
        }

        (float[] cos, float[] sin) = CpuMath.RopeTables(rows, HeadDim, RopeTheta);
        (float[] contextCos, float[] contextSin) = CpuMath.RopeTables(contextRows, HeadDim, RopeTheta);
        var normed = new float[rows * Width];
        var query = new float[rows * Width];
        var key = new float[Math.Max(rows, contextRows) * Width];
        var value = new float[Math.Max(rows, contextRows) * Width];
        var attended = new float[rows * Width];
        var mlp = new float[rows * MlpWidth];

        for (int block = 0; block < Blocks; block++)
        {
            string prefix = $"blocks.{block}";
            HostTensor Weight(string name) => parameters.Matrix($"{prefix}.{name}.weight");
            float[] Vector(string name) => parameters.Vector($"{prefix}.{name}");

            // Self-attention over the T5 sequence, unmasked.
            CpuMath.RmsNorm(x, rows, Width, Vector("norm_self_attn.weight"), normed, NormEpsilon);
            Attention(normed, rows, normed, rows, cos, sin, cos, sin, Weight, Vector, "self_attn",
                query, key, value, attended);
            CpuMath.Linear(attended, rows, Width, Weight("self_attn.o_proj"), null, normed, Width);
            CpuMath.Add(x, normed, x.Length);

            // Cross-attention into the Qwen sequence, its keys placed by
            // their own positions.
            CpuMath.RmsNorm(x, rows, Width, Vector("norm_cross_attn.weight"), normed, NormEpsilon);
            Attention(normed, rows, context, contextRows, cos, sin, contextCos, contextSin, Weight,
                Vector, "cross_attn", query, key, value, attended);
            CpuMath.Linear(attended, rows, Width, Weight("cross_attn.o_proj"), null, normed, Width);
            CpuMath.Add(x, normed, x.Length);

            CpuMath.RmsNorm(x, rows, Width, Vector("norm_mlp.weight"), normed, NormEpsilon);
            CpuMath.Linear(normed, rows, Width, Weight("mlp.0"), Vector("mlp.0.bias"), mlp, MlpWidth);
            CpuMath.Gelu(mlp, mlp.Length);
            CpuMath.Linear(mlp, rows, MlpWidth, Weight("mlp.2"), Vector("mlp.2.bias"), normed, Width);
            CpuMath.Add(x, normed, x.Length);
        }

        var projected = new float[rows * Width];
        CpuMath.Linear(x, rows, Width, parameters.Matrix("out_proj.weight"),
            parameters.Vector("out_proj.bias"), projected, Width);
        CpuMath.RmsNorm(projected, rows, Width, parameters.Vector("norm.weight"), projected, NormEpsilon);
        return projected;
    }

    /// <summary>Project, norm each head, rotate, attend: the shared middle of
    /// both attentions, differing in where the keys and values come from and
    /// which positions they carry.</summary>
    private static void Attention(float[] queries, int rows, float[] source, int sourceRows,
        float[] cos, float[] sin, float[] sourceCos, float[] sourceSin,
        Func<string, HostTensor> weight, Func<string, float[]> vector, string prefix,
        float[] query, float[] key, float[] value, float[] attended)
    {
        CpuMath.Linear(queries, rows, Width, weight($"{prefix}.q_proj"), null, query, Width);
        CpuMath.Linear(source, sourceRows, Width, weight($"{prefix}.k_proj"), null, key, Width);
        CpuMath.Linear(source, sourceRows, Width, weight($"{prefix}.v_proj"), null, value, Width);
        CpuMath.RmsNorm(query, rows * Heads, HeadDim, vector($"{prefix}.q_norm.weight"), query, NormEpsilon);
        CpuMath.RmsNorm(key, sourceRows * Heads, HeadDim, vector($"{prefix}.k_norm.weight"), key, NormEpsilon);
        CpuMath.Rope(query, rows, Heads, HeadDim, cos, sin);
        CpuMath.Rope(key, sourceRows, Heads, HeadDim, sourceCos, sourceSin);
        CpuMath.Attention(query, key, value, rows, sourceRows, Heads, Heads, HeadDim,
            causal: false, attended);
    }
}
