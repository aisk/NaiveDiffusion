using NaiveDiffusion.Tensors;
using NaiveDiffusion.Text;

namespace NaiveDiffusion.Text.Qwen;

/// <summary>A Qwen3 decoder run as a text encoder, on the CPU: the token ids
/// in, the last layer's hidden states out — after the final norm, which is
/// what ComfyUI hands Anima's adapter from qwen3_06b, or before it, which
/// is what Qwen-Image reads from Qwen3-VL. The architecture is Llama's:
/// pre-norm blocks of grouped-query attention with a norm on each query and
/// key head and rotary positions, then a SwiGLU feed-forward; causal, so
/// every token sees only what came before it.
///
/// On the CPU for the reason the CLIP towers are (see
/// <see cref="ClipTextEncoder"/>): the weights are read straight off the
/// mapped file a row at a time — 1.2 GiB of bfloat16 for the 0.6B, 16 GiB
/// for the 8B — the arithmetic for a prompt's worth of tokens is a second
/// for the small model and some five seconds for the large one, and a
/// graph compiled for one sequence length would have to be rebuilt for the
/// next prompt.</summary>
public static class Qwen3TextModel
{
    /// <summary>The widths of one Qwen3 size. 0.6B is the one Anima reads
    /// with; the 8B is the language model inside Qwen3-VL-8B, which
    /// Qwen-Image 2.1 reads with — on text alone its three-axis rotary
    /// positions all count the same tokens, which is one axis, so it is
    /// this model at a longer rotary base.</summary>
    public sealed record Config(int Width, int LayerCount, int Heads, int KeyValueHeads,
        int HeadDim, int MlpWidth, float NormEpsilon, double RopeTheta)
    {
        public static readonly Config Qwen3_0_6B = new(Width: 1024, LayerCount: 28, Heads: 16,
            KeyValueHeads: 8, HeadDim: 128, MlpWidth: 3072, NormEpsilon: 1e-6f, RopeTheta: 1e6);

        public static readonly Config Qwen3Vl_8B = new(Width: 4096, LayerCount: 36, Heads: 32,
            KeyValueHeads: 8, HeadDim: 128, MlpWidth: 12288, NormEpsilon: 1e-6f, RopeTheta: 5e6);
    }

    /// <summary>The hidden state per token, [ids.Length, Width]: the last
    /// block's output through the final norm, or, with
    /// <paramref name="finalNorm"/> false, as it left the block —
    /// transformers' hidden_states[-1].</summary>
    public static float[] Forward(TowerWeights parameters, int[] ids, Config config,
        bool finalNorm = true)
    {
        int rows = ids.Length;
        int width = config.Width;
        int queryWidth = config.Heads * config.HeadDim;
        int keyWidth = config.KeyValueHeads * config.HeadDim;

        HostTensor table = parameters.Matrix("model.embed_tokens.weight");
        var x = new float[rows * width];
        for (int m = 0; m < rows; m++)
        {
            CpuMath.CopyRow(table, ids[m], width, x, m * width);
        }

        (float[] cos, float[] sin) = CpuMath.RopeTables(rows, config.HeadDim, config.RopeTheta);
        var normed = new float[rows * width];
        var query = new float[rows * queryWidth];
        var key = new float[rows * keyWidth];
        var value = new float[rows * keyWidth];
        var attended = new float[rows * queryWidth];
        var gate = new float[rows * config.MlpWidth];
        var up = new float[rows * config.MlpWidth];

        for (int layer = 0; layer < config.LayerCount; layer++)
        {
            string prefix = $"model.layers.{layer}";
            HostTensor Weight(string name) => parameters.Matrix($"{prefix}.{name}.weight");
            float[] Norm(string name) => parameters.Vector($"{prefix}.{name}.weight");

            CpuMath.RmsNorm(x, rows, width, Norm("input_layernorm"), normed, config.NormEpsilon);
            CpuMath.Linear(normed, rows, width, Weight("self_attn.q_proj"), null, query, queryWidth);
            CpuMath.Linear(normed, rows, width, Weight("self_attn.k_proj"), null, key, keyWidth);
            CpuMath.Linear(normed, rows, width, Weight("self_attn.v_proj"), null, value, keyWidth);
            // Each head normalized on its own before the positions go on.
            CpuMath.RmsNorm(query, rows * config.Heads, config.HeadDim, Norm("self_attn.q_norm"),
                query, config.NormEpsilon);
            CpuMath.RmsNorm(key, rows * config.KeyValueHeads, config.HeadDim, Norm("self_attn.k_norm"),
                key, config.NormEpsilon);
            CpuMath.Rope(query, rows, config.Heads, config.HeadDim, cos, sin);
            CpuMath.Rope(key, rows, config.KeyValueHeads, config.HeadDim, cos, sin);
            CpuMath.Attention(query, key, value, rows, rows, config.Heads, config.KeyValueHeads,
                config.HeadDim, causal: true, attended);
            CpuMath.Linear(attended, rows, queryWidth, Weight("self_attn.o_proj"), null, normed, width);
            CpuMath.Add(x, normed, x.Length);

            CpuMath.RmsNorm(x, rows, width, Norm("post_attention_layernorm"), normed, config.NormEpsilon);
            CpuMath.Linear(normed, rows, width, Weight("mlp.gate_proj"), null, gate, config.MlpWidth);
            CpuMath.Linear(normed, rows, width, Weight("mlp.up_proj"), null, up, config.MlpWidth);
            CpuMath.Silu(gate, gate.Length);
            CpuMath.Multiply(gate, up, gate.Length);
            CpuMath.Linear(gate, rows, config.MlpWidth, Weight("mlp.down_proj"), null, normed, width);
            CpuMath.Add(x, normed, x.Length);
        }

        if (!finalNorm)
        {
            return x;
        }
        CpuMath.RmsNorm(x, rows, width, parameters.Vector("model.norm.weight"), normed,
            config.NormEpsilon);
        return normed;
    }
}
