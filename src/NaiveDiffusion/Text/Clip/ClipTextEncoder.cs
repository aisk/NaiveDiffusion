using NaiveDiffusion.Tensors;
using NaiveDiffusion.Text;

namespace NaiveDiffusion.Text.Clip;

/// <summary>One CLIP text tower — ViT-L/14 (768 wide, QuickGELU) or OpenCLIP
/// ViT-bigG/14 (1280 wide, GELU, with the projection SDXL pools from) — run
/// over one chunk of tokens. The embeddings SDXL uses are the penultimate
/// layer's output without the final layer norm — or another layer's, when
/// the clip skip says so; the pooled vector comes from the last layer with
/// it either way.
///
/// <b>These run on the CPU, unlike the diffusion model and the VAE.</b> The
/// arithmetic is tiny — a 77-token chunk through bigG is about 97 GFLOP, which
/// takes well under a second on eight cores — but the weights are 3.05 GiB,
/// and on the GPU they cost far more than that: DirectML keeps a repacked
/// second copy of every gemm weight whose row count is not a multiple of 64,
/// and CLIP's sequence length is fixed at 77, so the towers always pay it.
/// Measured, the GPU path peaked at 8601 MiB to save 0.05 seconds of compute.
/// See <see cref="Tensors.CpuMath"/> for the kernels.</summary>
public static class ClipTextEncoder
{
    public const int MaxTokens = ClipTokenizer.MaxTokens;
    private const float LayerNormEpsilon = 1e-5f;

    public sealed record Config(string Name, int Width, int LayerCount, int Heads,
        bool UsesQuickGelu, bool WantsPooled, string PadToken);

    public static readonly Config[] Configs =
    {
        new("text_encoder", 768, 12, 12, UsesQuickGelu: true, WantsPooled: false,
            PadToken: "<|endoftext|>"),
        new("text_encoder_2", 1280, 32, 20, UsesQuickGelu: false, WantsPooled: true,
            PadToken: "!"),
    };

    /// <summary>The deepest clip skip: with 1 meaning the last layer, this
    /// is the first layer of the 12-layer tower, and A1111's upper bound.
    /// The 32-layer tower takes the same count from its own end.</summary>
    public const int MaxClipSkip = 12;

    /// <summary>Run one tower over one chunk of token ids.
    ///
    /// Returns the hidden state <paramref name="clipSkip"/> layers from the
    /// end — 2, the penultimate, is what SDXL cross-attends to — without the
    /// final layer norm, and, for the tower that carries the projection, the
    /// last layer's hidden state after that norm, which is what pooling
    /// reads whatever the skip. The tower without a projection stops at the
    /// skipped-to layer and returns null for the rest: nothing downstream
    /// would look at it.</summary>
    public static (float[] Hidden, float[]? Final) Forward(
        TowerWeights parameters, uint[] ids, Config config, int clipSkip = 2)
    {
        if (clipSkip < 1 || clipSkip > MaxClipSkip)
        {
            throw new ArgumentOutOfRangeException(nameof(clipSkip), clipSkip,
                $"between 1 and {MaxClipSkip}");
        }
        int rows = MaxTokens;
        int width = config.Width;
        int size = rows * width;
        int mlpWidth = (int)(parameters
            .Matrix("text_model.encoder.layers.0.mlp.fc1.weight").ElementCount / width);

        // Token embeddings, looked up by id, plus the learned position vectors.
        // Both tables stay packed and are widened a row at a time.
        HostTensor table = parameters.Matrix("text_model.embeddings.token_embedding.weight");
        HostTensor positions =
            parameters.Matrix("text_model.embeddings.position_embedding.weight");
        var x = new float[size];
        for (int m = 0; m < rows; m++)
        {
            CpuMath.CopyRow(table, ids[m], width, x, m * width);
            CpuMath.AddRow(positions, m, width, x, m * width);
        }

        // One set of scratch buffers for the whole tower, reused every layer.
        var scratch = new Scratch(new float[size], new float[size], new float[size],
            new float[size], new float[size], new float[rows * mlpWidth]);

        // The sequence comes from clipSkip layers before the end — the
        // penultimate at 2 — so stop there and keep a copy before the layers
        // after it overwrite it.
        int kept = config.LayerCount - clipSkip + 1;
        for (int i = 0; i < kept; i++)
        {
            EncoderLayer(x, parameters, $"text_model.encoder.layers.{i}", config, scratch,
                rows, mlpWidth);
        }
        float[] hidden = (float[])x.Clone();

        // The layers after it only earn their keep when the pooled vector is
        // wanted.
        if (!config.WantsPooled)
        {
            return (hidden, null);
        }

        for (int i = kept; i < config.LayerCount; i++)
        {
            EncoderLayer(x, parameters, $"text_model.encoder.layers.{i}", config, scratch,
                rows, mlpWidth);
        }

        var final = new float[size];
        CpuMath.LayerNorm(x, rows, width, parameters.Vector("text_model.final_layer_norm.weight"),
            parameters.Vector("text_model.final_layer_norm.bias"), final, LayerNormEpsilon);
        return (hidden, final);
    }

    /// <summary>Working buffers, so a 32-layer tower allocates once rather than
    /// once per layer. Every one is written before it is read.</summary>
    private readonly record struct Scratch(float[] Normed, float[] Query, float[] Key,
        float[] Value, float[] Attended, float[] Mlp);

    /// <summary>Attention then feed-forward, each pre-normed and residual —
    /// CLIP's encoder block. <paramref name="x"/> is updated in place.</summary>
    private static void EncoderLayer(float[] x, TowerWeights parameters,
        string prefix, Config config, Scratch scratch, int rows, int mlpWidth)
    {
        int width = config.Width;
        int size = rows * width;

        HostTensor Weight(string name) => parameters.Matrix($"{prefix}.{name}.weight");
        float[] Norm(string name) => parameters.Vector($"{prefix}.{name}.weight");
        float[] Bias(string name) => parameters.Vector($"{prefix}.{name}.bias");

        CpuMath.LayerNorm(x, rows, width, Norm("layer_norm1"), Bias("layer_norm1"),
            scratch.Normed, LayerNormEpsilon);

        CpuMath.Linear(scratch.Normed, rows, width, Weight("self_attn.q_proj"),
            Bias("self_attn.q_proj"), scratch.Query, width);
        CpuMath.Linear(scratch.Normed, rows, width, Weight("self_attn.k_proj"),
            Bias("self_attn.k_proj"), scratch.Key, width);
        CpuMath.Linear(scratch.Normed, rows, width, Weight("self_attn.v_proj"),
            Bias("self_attn.v_proj"), scratch.Value, width);
        CpuMath.Attention(scratch.Query, scratch.Key, scratch.Value, rows, rows,
            config.Heads, config.Heads, width / config.Heads, causal: true, scratch.Attended);

        // Normed is free again: it was the input to the projections, not to this.
        CpuMath.Linear(scratch.Attended, rows, width, Weight("self_attn.out_proj"),
            Bias("self_attn.out_proj"), scratch.Normed, width);
        CpuMath.Add(x, scratch.Normed, size);

        CpuMath.LayerNorm(x, rows, width, Norm("layer_norm2"), Bias("layer_norm2"),
            scratch.Normed, LayerNormEpsilon);
        CpuMath.Linear(scratch.Normed, rows, width, Weight("mlp.fc1"), Bias("mlp.fc1"),
            scratch.Mlp, mlpWidth);
        if (config.UsesQuickGelu)
        {
            CpuMath.QuickGelu(scratch.Mlp, rows * mlpWidth);
        }
        else
        {
            CpuMath.Gelu(scratch.Mlp, rows * mlpWidth);
        }
        CpuMath.Linear(scratch.Mlp, rows, mlpWidth, Weight("mlp.fc2"), Bias("mlp.fc2"),
            scratch.Normed, width);
        CpuMath.Add(x, scratch.Normed, size);
    }
}
