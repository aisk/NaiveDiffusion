using NaiveDiffusion.Dml;
using NaiveDiffusion.Graph;
using NaiveDiffusion.Tensors;
using NaiveDiffusion.Vae;

namespace NaiveDiffusion.Models.Anima;

/// <summary>Anima's diffusion transformer — NVIDIA's Cosmos-Predict2 2B
/// architecture, trained on a single frame: a noisy latent cut into 2×2
/// patches, the noise level as a sinusoid, and the adapter's context in; the
/// flow the sampler steps along out. Twenty-eight identical blocks, each
/// self-attention with 3-D rotary positions, cross-attention into the
/// context and a GELU feed-forward, every one gated and modulated from the
/// time embedding through a low-rank adaLN. Built as a chain of graphs, one
/// per block, with the weights stored at half precision and everything
/// computed at single precision: the residual stream because its values run
/// past what half precision holds, which is why the reference keeps it wide
/// too, and the blocks because DirectML's half-precision attention and
/// products drift far enough here to leave a hatching over a whole image
/// (see <see cref="_compute"/>). The width, the head size and the block
/// count are read off the weights, so a random small model of the same
/// shape runs through the same graphs.
///
/// The patching runs on the CPU at both ends: a latent is a megabyte, the
/// rearrangement is a copy, and it saves the graph a strided view of six
/// dimensions. The padding-mask channel the architecture concatenates is
/// all zeros on an unpadded image, so its 4 of the 68 embedding columns are
/// left out rather than multiplied by nothing.</summary>
public sealed class AnimaDit : ChainDenoiser
{
    public const int LatentChannels = WanVae.LatentChannels;
    public const int Patch = 2;
    private const float NormEpsilon = 1e-6f;
    private const double RopeTheta = 10000;

    // The names values travel under between links.
    private const string Patches = "patches";
    private const string Time = "time";
    private const string Context = "context";
    private const string Cos = "cos";
    private const string Sin = "sin";
    private const string X = "x";
    private const string Temb = "temb";
    private const string Lora = "lora";
    private const string Out = "out";

    /// <summary>The width the weights are stored at. The checkpoint is
    /// bfloat16, which half precision holds exactly, so this costs nothing
    /// but the bytes it saves; <see cref="Layers.Linear"/> widens each weight
    /// in the graph to the precision of the activation it meets.</summary>
    private const HostDataType Narrow = HostDataType.Float16;

    /// <summary>The width the attention, the MLP and the modulation run at.
    /// At single precision a forward pass lands within 1e-6 of the reference
    /// at 256² and 1024² alike. At half precision it is 0.7% off at σ=0.7 and
    /// 1.8% at σ=1 — several times what torch's own half-precision run drifts
    /// by — and twenty steps of that at 1024² put a two-pixel diagonal
    /// hatching over the whole image (an isolated spectral peak at Nyquist
    /// some three hundred times the background) and change the composition
    /// under the same seed. Single precision costs a fifth more time per step
    /// and 770 MiB more scratch; the resident weights stay the same size.</summary>
    private const HostDataType Wide = HostDataType.Float32;

    /// <summary>What the blocks compute at: <see cref="Wide"/> unless the
    /// half-precision figure was asked for
    /// (<see cref="Pipeline.GenerationOptions.DenoiserHalfCompute"/>), and
    /// then the products and the attention run narrow while the residual
    /// stream, the norms, the rotary positions, the embeddings and the final
    /// layer stay wide.</summary>
    private readonly HostDataType _compute;
    private readonly int _latentHeight;
    private readonly int _latentWidth;
    private readonly int _blocks;
    private readonly int _width;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly HostTensor _cos;
    private readonly HostTensor _sin;

    public AnimaDit(DmlDevice device, IReadOnlyDictionary<string, HostTensor> parameters,
        int height, int width, int contextRows, ulong? residentBudget = null,
        bool int8Weights = false, bool halfCompute = false)
    {
        _compute = halfCompute ? HostDataType.Float16 : Wide;
        _latentHeight = height / WanVae.Latent.ScaleFactor;
        _latentWidth = width / WanVae.Latent.ScaleFactor;
        if (_latentHeight % Patch != 0 || _latentWidth % Patch != 0)
        {
            throw new ArgumentException($"{width}x{height} does not divide into 16-pixel patches");
        }

        // The widths from the weights themselves. The 2B is 2048 wide with
        // sixteen heads of 128.
        int[] embedding = parameters["x_embedder.proj.1.weight"].Shape;
        if (embedding.Length != 2 || embedding[1] != (LatentChannels + 1) * Patch * Patch)
        {
            throw new InvalidDataException(
                $"x_embedder.proj.1.weight is [{string.Join(", ", embedding)}], not [width, {(LatentChannels + 1) * Patch * Patch}]");
        }
        _width = embedding[0];
        _headDim = parameters["blocks.0.self_attn.q_norm.weight"].Shape[0];
        if (_headDim % 2 != 0 || _width % _headDim != 0)
        {
            throw new InvalidDataException($"a head of {_headDim} features does not divide the width of {_width}");
        }
        _heads = _width / _headDim;
        _blocks = 0;
        while (parameters.ContainsKey($"blocks.{_blocks}.mlp.layer1.weight"))
        {
            _blocks++;
        }
        (_cos, _sin) = RopeTables(_latentHeight / Patch, _latentWidth / Patch, _headDim);

        Build(device, Narrow, residentBudget, int8Weights, chain => Build(chain, parameters, contextRows));
    }

    public int Tokens => _latentHeight / Patch * (_latentWidth / Patch);

    /// <summary>The flow at <paramref name="latent"/> for noise level
    /// <paramref name="sigma"/>, under one branch of <see cref="AnimaConditioning"/>.
    /// The training-table index is not this model's time; the level is,
    /// and Cosmos embeds it as it is.</summary>
    public override HostTensor Predict(HostTensor latent, double timestep, double sigma,
        IReadOnlyDictionary<string, HostTensor> conditioning)
    {
        HostTensor output = Chain.Run(new Dictionary<string, HostTensor>
        {
            [Patches] = Patchify(latent.ToFloats()),
            [Time] = HostTensor.FromFloats(Transformer.Sinusoid((float)sigma, _width), 1, 1, 1, _width),
            [Context] = conditioning[AnimaConditioning.ContextName],
            [Cos] = _cos,
            [Sin] = _sin,
        });
        return Unpatchify(output.ToFloats());
    }

    // --- CPU-side conditioning ---------------------------------------------

    /// <summary>Cosmos's 3-D rotary tables for a grid of patches, one row per
    /// patch in row-major order and one column per pair of head features:
    /// a head of 128 features makes 64 pairs, the first 22 turning with time
    /// (a single frame sits at 0, so they do not turn at all), the next 21
    /// with the row and the last 21 with the column, each set at its own
    /// frequencies. The spatial base is 10000 raised by the NTK factor of
    /// the extrapolation ratio ComfyUI configures for the 16-channel model,
    /// 4 — the 2B was trained on longer sides than its rotary base was set
    /// for, and the factor is how it is read at them. Returned as
    /// [1, tokens, 1, dim/2] to broadcast over heads.</summary>
    public static (HostTensor Cos, HostTensor Sin) RopeTables(int rows, int columns, int headDim)
    {
        int spatial = headDim / 6 * 2;          // 42 features, 21 pairs, for each of h and w
        int temporal = headDim - 2 * spatial;    // 44 features, 22 pairs
        int pairs = headDim / 2;
        int tokens = rows * columns;
        var cos = new float[tokens * pairs];
        var sin = new float[tokens * pairs];
        const double extrapolation = 4.0;
        double ntkFactor = Math.Pow(extrapolation, spatial / (double)(spatial - 2));
        float theta = (float)(RopeTheta * ntkFactor);
        var spatialFrequencies = new float[spatial / 2];
        for (int k = 0; k < spatialFrequencies.Length; k++)
        {
            spatialFrequencies[k] = 1.0f / MathF.Pow(theta, 2.0f * k / spatial);
        }
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                int token = i * columns + j;
                int offset = token * pairs;
                for (int d = 0; d < temporal / 2; d++)
                {
                    cos[offset + d] = 1f;
                }
                for (int k = 0; k < spatial / 2; k++)
                {
                    float rowAngle = i * spatialFrequencies[k];
                    float columnAngle = j * spatialFrequencies[k];
                    cos[offset + temporal / 2 + k] = MathF.Cos(rowAngle);
                    sin[offset + temporal / 2 + k] = MathF.Sin(rowAngle);
                    cos[offset + temporal / 2 + spatial / 2 + k] = MathF.Cos(columnAngle);
                    sin[offset + temporal / 2 + spatial / 2 + k] = MathF.Sin(columnAngle);
                }
            }
        }
        return (HostTensor.FromFloats(cos, 1, tokens, 1, pairs),
            HostTensor.FromFloats(sin, 1, tokens, 1, pairs));
    }

    /// <summary>[16, H, W] to [tokens, 64]: patch (i, j)'s feature c·4 + m·2
    /// + n is channel c at row 2i + m, column 2j + n — the order the
    /// embedding's columns were trained in.</summary>
    private HostTensor Patchify(float[] latent)
    {
        int rows = _latentHeight / Patch, columns = _latentWidth / Patch;
        int features = LatentChannels * Patch * Patch;
        var patches = new float[rows * columns * features];
        int plane = _latentHeight * _latentWidth;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                int token = (i * columns + j) * features;
                for (int c = 0; c < LatentChannels; c++)
                {
                    for (int m = 0; m < Patch; m++)
                    {
                        for (int n = 0; n < Patch; n++)
                        {
                            patches[token + c * Patch * Patch + m * Patch + n] =
                                latent[c * plane + (i * Patch + m) * _latentWidth + j * Patch + n];
                        }
                    }
                }
            }
        }
        return HostTensor.FromFloats(patches, 1, 1, rows * columns, features);
    }

    /// <summary>[tokens, 64] back to [1, 16, H, W]: the final layer's feature
    /// p1·32 + p2·16 + c is channel c at row 2i + p1, column 2j + p2.</summary>
    private HostTensor Unpatchify(float[] output)
    {
        int rows = _latentHeight / Patch, columns = _latentWidth / Patch;
        int features = LatentChannels * Patch * Patch;
        int plane = _latentHeight * _latentWidth;
        var latent = new float[LatentChannels * plane];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                int token = (i * columns + j) * features;
                for (int p1 = 0; p1 < Patch; p1++)
                {
                    for (int p2 = 0; p2 < Patch; p2++)
                    {
                        for (int c = 0; c < LatentChannels; c++)
                        {
                            latent[c * plane + (i * Patch + p1) * _latentWidth + j * Patch + p2] =
                                output[token + (p1 * Patch + p2) * LatentChannels + c];
                        }
                    }
                }
            }
        }
        return HostTensor.FromFloats(latent, 1, LatentChannels, _latentHeight, _latentWidth);
    }

    /// <summary>The embedding's columns for the sixteen real channels; the
    /// last four met the zero padding-mask channel.</summary>
    private static HostTensor RealChannelColumns(HostTensor embedding)
    {
        int outputs = embedding.Shape[0], inputs = embedding.Shape[1];
        int kept = LatentChannels * Patch * Patch;
        float[] wide = embedding.ToFloats();
        var narrow = new float[outputs * kept];
        for (int o = 0; o < outputs; o++)
        {
            Array.Copy(wide, o * inputs, narrow, o * kept, kept);
        }
        return HostTensor.FromFloats(narrow, outputs, kept);
    }

    // --- Graph construction ------------------------------------------------

    /// <summary>The low-rank adaLN: SiLU of the time embedding through two
    /// linears, plus the shared term from the time embedder, cut into
    /// <paramref name="chunks"/> vectors of the model width, each widened to
    /// single precision and spread over the tokens.</summary>
    private DmlExpression[] Modulation(ModelBuilder model, DmlExpression temb,
        DmlExpression lora, IReadOnlyDictionary<string, HostTensor> parameters, string prefix,
        int chunks, uint tokens)
    {
        DmlExpression modulation = Layers.Linear(model,
            Layers.Linear(model, Layers.Silu(temb), parameters[$"{prefix}.1.weight"]),
            parameters[$"{prefix}.2.weight"]);
        // The final layer's two chunks take the first two of the embedder's three.
        modulation += chunks == 3
            ? lora
            : DmlOps.Slice(lora, new[] { 0, 0, 0, 0 }, new[] { 1, 1, 1, chunks * _width },
                new[] { 1, 1, 1, 1 });
        var parts = new DmlExpression[chunks];
        for (int i = 0; i < chunks; i++)
        {
            DmlExpression part = DmlOps.Slice(modulation, new[] { 0, 0, 0, i * _width },
                new[] { 1, 1, 1, _width }, new[] { 1, 1, 1, 1 });
            parts[i] = Layers.Broadcast(Layers.Cast(part, HostDataType.Float32),
                new[] { 1u, 1u, tokens, (uint)_width });
        }
        return parts;
    }

    /// <summary>Attention with the query and key heads normed, rotated when
    /// positions are given, as one operator over the packed heads.</summary>
    private DmlExpression Attention(ModelBuilder model, DmlExpression x, DmlExpression source,
        IReadOnlyDictionary<string, HostTensor> parameters, string prefix,
        DmlExpression? cos, DmlExpression? sin)
    {
        // The per-head norms at the precision the projections come out at:
        // the mean-variance normalization accumulates a head's sum of
        // squares wider than half, so no cast is needed around it.
        DmlExpression query = Transformer.HeadNorm(model,
            Layers.Linear(model, x, parameters[$"{prefix}.q_proj.weight"]),
            parameters[$"{prefix}.q_norm.weight"], _heads, _headDim, NormEpsilon);
        DmlExpression key = Transformer.HeadNorm(model,
            Layers.Linear(model, source, parameters[$"{prefix}.k_proj.weight"]),
            parameters[$"{prefix}.k_norm.weight"], _heads, _headDim, NormEpsilon);
        DmlExpression value = Layers.Cast(
            Layers.Linear(model, source, parameters[$"{prefix}.v_proj.weight"]), _compute);
        if (cos is not null && sin is not null)
        {
            query = Transformer.Rotate(query, cos, sin, _heads, _headDim, _compute);
            key = Transformer.Rotate(key, cos, sin, _heads, _headDim, _compute);
        }
        else
        {
            query = Layers.Cast(query, _compute);
            key = Layers.Cast(key, _compute);
        }
        DmlExpression attended = Layers.Attend(Transformer.Flat(query, _width),
            Transformer.Flat(key, _width), value, _heads);
        return Layers.Linear(model, attended, parameters[$"{prefix}.output_proj.weight"]);
    }

    private void Block(GraphChain chain, IReadOnlyDictionary<string, HostTensor> parameters,
        string prefix)
    {
        ModelBuilder model = chain.Current;
        DmlExpression x = chain.Get(X);
        DmlExpression temb = chain.Get(Temb);
        DmlExpression lora = chain.Get(Lora);
        uint tokens = x.Shape[2];

        DmlExpression[] self = Modulation(model, temb, lora, parameters,
            $"{prefix}.adaln_modulation_self_attn", 3, tokens);
        DmlExpression[] cross = Modulation(model, temb, lora, parameters,
            $"{prefix}.adaln_modulation_cross_attn", 3, tokens);
        DmlExpression[] mlp = Modulation(model, temb, lora, parameters,
            $"{prefix}.adaln_modulation_mlp", 3, tokens);

        DmlExpression normed = Transformer.Modulate(x, self[0], self[1], NormEpsilon, _compute);
        DmlExpression result = Attention(model, normed, normed, parameters, $"{prefix}.self_attn",
            chain.Get(Cos), chain.Get(Sin));
        x += self[2] * Layers.Cast(result, Wide);

        normed = Transformer.Modulate(x, cross[0], cross[1], NormEpsilon, _compute);
        result = Attention(model, normed, chain.Get(Context), parameters, $"{prefix}.cross_attn",
            null, null);
        x += cross[2] * Layers.Cast(result, Wide);

        normed = Transformer.Modulate(x, mlp[0], mlp[1], NormEpsilon, _compute);
        result = Layers.Linear(model,
            DmlOps.ActivationGelu(Layers.Linear(model, normed, parameters[$"{prefix}.mlp.layer1.weight"])),
            parameters[$"{prefix}.mlp.layer2.weight"]);
        x += mlp[2] * Layers.Cast(result, Wide);

        chain.Set(X, x);
    }

    private void Build(GraphChain chain, IReadOnlyDictionary<string, HostTensor> parameters,
        int contextRows)
    {
        int tokens = Tokens;
        int features = LatentChannels * Patch * Patch;
        chain.Input(Patches, new[] { 1, 1, tokens, features }, Wide);
        chain.Input(Time, new[] { 1, 1, 1, _width }, Wide);
        chain.Input(Context, new[] { 1, 1, contextRows, LlmAdapter.Width }, Wide);
        chain.Input(Cos, new[] { 1, tokens, 1, _headDim / 2 }, HostDataType.Float32);
        chain.Input(Sin, new[] { 1, tokens, 1, _headDim / 2 }, HostDataType.Float32);

        // The embeddings: patches to the model width, kept wide from here on;
        // the time to its norm and, through the embedder, the shared adaLN term.
        ModelBuilder model = chain.Current;
        chain.Set(X, Layers.Cast(Layers.Linear(model, chain.Get(Patches),
            RealChannelColumns(parameters["x_embedder.proj.1.weight"])), HostDataType.Float32));
        chain.Drop(Patches);
        DmlExpression time = chain.Get(Time);
        chain.Set(Temb, Layers.RmsNorm(model, time, parameters["t_embedding_norm.weight"], NormEpsilon));
        chain.Set(Lora, Layers.Linear(model,
            Layers.Silu(Layers.Linear(model, time, parameters["t_embedder.1.linear_1.weight"])),
            parameters["t_embedder.1.linear_2.weight"]));
        chain.Drop(Time);
        chain.Cut();

        for (int block = 0; block < _blocks; block++)
        {
            Block(chain, parameters, $"blocks.{block}");
            chain.Cut();
        }

        model = chain.Current;
        DmlExpression x = chain.Get(X);
        DmlExpression[] final = Modulation(model, chain.Get(Temb), chain.Get(Lora), parameters,
            "final_layer.adaln_modulation", 2, (uint)tokens);
        chain.Set(Out, Layers.Linear(model, Transformer.Modulate(x, final[0], final[1], NormEpsilon, Wide),
            parameters["final_layer.linear.weight"]));
        chain.Drop(X);
        chain.Drop(Temb);
        chain.Drop(Lora);
        chain.Drop(Context);
        chain.Drop(Cos);
        chain.Drop(Sin);
        chain.Finish(Out);
    }
}
