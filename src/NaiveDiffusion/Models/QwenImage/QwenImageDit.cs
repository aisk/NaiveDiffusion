using NaiveDiffusion.Dml;
using NaiveDiffusion.Graph;
using NaiveDiffusion.Tensors;
using NaiveDiffusion.Vae;
using Vortice.DirectML;

namespace NaiveDiffusion.Models.QwenImage;

/// <summary>Qwen-Image 2.1's diffusion transformer: a single stream of
/// tokens — the latent's cells, one token each, and the text encoder's
/// hidden states projected to the model width — through thirty-two
/// identical blocks of self-attention with three-axis rotary positions and
/// a SwiGLU feed-forward, each gated and scaled by one modulation the whole
/// model shares, computed from the noise level; the flow the sampler steps
/// along out, read from the image tokens.
///
/// The text tokens attend causally among themselves and see nothing of the
/// image; the image tokens see everything. That is the block-causal mask
/// of the reference, and here it is two attentions per block: the image
/// rows' queries over every key, and the text rows' queries over the text
/// keys under a causal table. The text rows are also modulated as if at
/// time zero, so their path through the blocks never depends on the noise
/// level — which is why the reference caches their keys and values across
/// the steps, and why this could; for now they are
/// recomputed every step, some five percent of the work at 1024².
///
/// The sequence is laid out image first, then the text with its padding
/// after it, so the padding a shorter branch carries (both branches of the
/// guidance go through the same compiled graphs) is masked by a count of
/// real keys. Everything a branch differs in — the projected context, its
/// two masks, and the rotary tables, whose image rows sit at the branch's
/// own text length — comes in with the conditioning
/// (<see cref="QwenImageConditioning"/>).
///
/// Built as a chain of graphs — several per block, since a compiled
/// graph's scratch is the sum of its intermediates and one block at 1024²
/// came to 9 GiB — with the weights stored at half precision and everything
/// computed at single precision, for the reason <see cref="Anima.AnimaDit"/>
/// gives: DirectML's half-precision
/// attention drifts enough over a run to leave a hatching over the image.
/// The blocks can be asked to compute at half precision instead
/// (<see cref="Pipeline.GenerationOptions.DenoiserCompute"/>), for the
/// timing: then the products and the attention run narrow while the
/// residual stream, the norms, the rotary positions, the embeddings and the
/// final layer stay wide.
/// The rotary positions are the reference's, which turns adjacent feature
/// pairs; the query and key projections and their norms are permuted once
/// on the host so the rotate-half form the graph has does the same thing
/// (<see cref="PairsToHalves"/>).</summary>
public sealed class QwenImageDit : ChainDenoiser
{
    public const int LatentChannels = Wan22Vae.LatentChannels;
    public const int Patch = 1;
    private const float NormEpsilon = 1e-6f;
    private const double RopeTheta = 10000;

    /// <summary>How many image rows one attention operator takes at a
    /// time (see <see cref="Attention"/>).</summary>
    private const int QueryChunk = 1024;

    /// <summary>What the sequence must be a multiple of when the blocks
    /// compute at half precision. A half-precision GEMM whose weight is a
    /// constant of the graph keeps a second, row-aligned copy of it in the
    /// persistent resource when the rows are not a multiple of 64 — 4121
    /// rows at 1024² double every block's weights and exhaust video
    /// memory — while at single precision the weight passes through a
    /// Cast first and the GEMM never sees a constant. The conditioning
    /// pads the text rows to make the total up
    /// (<see cref="QwenImageConditioning"/>).</summary>
    public const int HalfComputeAlignment = 64;

    /// <summary>How many features of a head turn with each rotary axis:
    /// the text position and the image's frame, the row, the column.</summary>
    private static readonly int[] RopeAxes = { 16, 56, 56 };

    // The names values travel under between links.
    private const string Latent = "latent";
    private const string Time = "time";
    private const string X = "x";
    private const string Temb = "temb";
    private const string Modulation = "modulation";
    private const string Query = "query";
    private const string Key = "key";
    private const string Value = "value";
    private const string Out = "out";

    private const HostDataType Narrow = HostDataType.Float16;
    private const HostDataType Wide = HostDataType.Float32;

    /// <summary>What the blocks compute at: single precision unless the
    /// half-precision figure was asked for.</summary>
    private readonly HostDataType _compute;

    private readonly int _latentHeight;
    private readonly int _latentWidth;
    private readonly int _contextRows;
    private readonly int _blocks;
    private readonly int _width;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _mlpWidth;
    private readonly int _timeWidth;

    public QwenImageDit(DmlDevice device, IReadOnlyDictionary<string, HostTensor> parameters,
        int height, int width, int contextRows, ulong? residentBudget = null,
        bool int8Weights = false, bool halfCompute = false)
    {
        _compute = halfCompute ? HostDataType.Float16 : Wide;
        int scale = Wan22Vae.Latent.ScaleFactor * Patch;
        if (height % scale != 0 || width % scale != 0)
        {
            throw new ArgumentException($"{width}x{height} does not divide into {scale}-pixel cells");
        }
        _latentHeight = height / Wan22Vae.Latent.ScaleFactor;
        _latentWidth = width / Wan22Vae.Latent.ScaleFactor;
        _contextRows = contextRows;

        // The widths from the weights themselves, so a small model of the
        // same shape runs through the same graphs.
        int[] embedding = parameters["img_in.weight"].Shape;
        if (embedding.Length != 2 || embedding[1] != LatentChannels * Patch * Patch)
        {
            throw new InvalidDataException(
                $"img_in.weight is [{string.Join(", ", embedding)}], not [width, {LatentChannels}]");
        }
        _width = embedding[0];
        _headDim = parameters["transformer_blocks.0.attn.norm_q.weight"].Shape[0];
        if (_headDim != RopeAxes.Sum() || _width % _headDim != 0)
        {
            throw new InvalidDataException($"a head of {_headDim} features does not take the rotary axes");
        }
        _heads = _width / _headDim;
        _mlpWidth = parameters.ContainsKey($"transformer_blocks.0.{QwenImageCheckpoint.FusedGateUp}")
            ? parameters[$"transformer_blocks.0.{QwenImageCheckpoint.FusedGateUp}"].Shape[0] / 2
            : parameters[$"transformer_blocks.0.{QwenImageCheckpoint.GateProjection}"].Shape[0];
        _timeWidth = parameters["time_text_embed.timestep_embedder.linear_1.weight"].Shape[1];
        _blocks = 0;
        while (parameters.ContainsKey($"transformer_blocks.{_blocks}.attn.to_q.weight"))
        {
            _blocks++;
        }

        Build(device, Narrow, residentBudget, int8Weights, chain => Build(chain, parameters));
    }

    public int Tokens => _latentHeight * _latentWidth;

    /// <summary>The width the text encoder's rows must have to be projected
    /// by this checkpoint's text input.</summary>
    public static int ContextWidth(IReadOnlyDictionary<string, HostTensor> parameters) =>
        parameters["txt_in.text_norm.weight"].Shape[0];

    /// <summary>The flow at <paramref name="latent"/> for noise level
    /// <paramref name="sigma"/>, under one branch of <see cref="QwenImageConditioning"/>.
    /// The level is this model's time, scaled by a thousand in the embedding.</summary>
    public override HostTensor Predict(HostTensor latent, double timestep, double sigma,
        IReadOnlyDictionary<string, HostTensor> conditioning)
    {
        float[] time = TimeEmbedding(sigma, _timeWidth).Concat(TimeEmbedding(0, _timeWidth)).ToArray();
        var inputs = new Dictionary<string, HostTensor>
        {
            [Latent] = ToTokens(latent.ToFloats()),
            [Time] = HostTensor.FromFloats(time, 1, 1, 2, _timeWidth),
        };
        foreach (string name in QwenImageConditioning.Names)
        {
            inputs[name] = conditioning[name];
        }
        return FromTokens(Chain.Run(inputs).ToFloats());
    }

    // --- CPU-side pieces -----------------------------------------------------

    /// <summary>Flux's sinusoid of the time: t × 1000 through
    /// <see cref="Transformer.Sinusoid"/>.</summary>
    public static float[] TimeEmbedding(double time, int width) =>
        Transformer.Sinusoid((float)(time * 1000.0), width);

    /// <summary>The rotary tables for one branch, one row per token in the
    /// order the graph holds them — the image's cells in row-major order,
    /// then <paramref name="contextRows"/> text rows — and one column per
    /// pair of head features: the first 8 pairs turn with the first axis,
    /// the next 28 with the second, the last 28 with the third, at
    /// frequencies 10000^(−2k/d) over each axis's d features. A text token
    /// p sits at (p, p, p); the image's cells sit at
    /// (<paramref name="textLength"/>, row − ⌈h/2⌉, column − ⌈w/2⌉), the
    /// frame counted after the branch's real text and the grid centred.
    /// Returned as [1, tokens, 1, 64] to broadcast over heads.</summary>
    public static (HostTensor Cos, HostTensor Sin) RopeTables(int rows, int columns, int textLength,
        int contextRows)
    {
        int pairs = RopeAxes.Sum() / 2;
        int imageTokens = rows * columns;
        int tokens = imageTokens + contextRows;
        var cos = new float[tokens * pairs];
        var sin = new float[tokens * pairs];

        var frequencies = new double[pairs];
        int pair = 0;
        foreach (int axis in RopeAxes)
        {
            for (int k = 0; k < axis / 2; k++)
            {
                frequencies[pair++] = 1.0 / Math.Pow(RopeTheta, 2.0 * k / axis);
            }
        }

        void Fill(int token, double first, double second, double third)
        {
            int offset = token * pairs;
            pair = 0;
            foreach ((int axis, double position) in new[]
                     {
                         (RopeAxes[0], first), (RopeAxes[1], second), (RopeAxes[2], third),
                     })
            {
                for (int k = 0; k < axis / 2; k++, pair++)
                {
                    double angle = position * frequencies[pair];
                    cos[offset + pair] = (float)Math.Cos(angle);
                    sin[offset + pair] = (float)Math.Sin(angle);
                }
            }
        }

        int rowStart = rows - rows / 2, columnStart = columns - columns / 2;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                Fill(i * columns + j, textLength, i - rowStart, j - columnStart);
            }
        }
        for (int p = 0; p < contextRows; p++)
        {
            Fill(imageTokens + p, p, p, p);
        }
        return (HostTensor.FromFloats(cos, 1, tokens, 1, pairs),
            HostTensor.FromFloats(sin, 1, tokens, 1, pairs));
    }

    /// <summary>[64, H, W] to [tokens, 64]: cell (i, j)'s features are its channels.</summary>
    private HostTensor ToTokens(float[] latent)
    {
        int tokens = Tokens;
        var features = new float[tokens * LatentChannels];
        for (int c = 0; c < LatentChannels; c++)
        {
            for (int token = 0; token < tokens; token++)
            {
                features[token * LatentChannels + c] = latent[c * tokens + token];
            }
        }
        return HostTensor.FromFloats(features, 1, 1, tokens, LatentChannels);
    }

    /// <summary>[tokens, 64] back to [1, 64, H, W].</summary>
    private HostTensor FromTokens(float[] output)
    {
        int tokens = Tokens;
        var latent = new float[LatentChannels * tokens];
        for (int token = 0; token < tokens; token++)
        {
            for (int c = 0; c < LatentChannels; c++)
            {
                latent[c * tokens + token] = output[token * LatentChannels + c];
            }
        }
        return HostTensor.FromFloats(latent, 1, LatentChannels, _latentHeight, _latentWidth);
    }

    /// <summary>A query or key projection, or its per-head norm, with each
    /// head's features reordered from adjacent pairs (2i, 2i+1) to halves
    /// (i, 64+i): the reference rotates pairs, the graph rotates halves,
    /// and a norm and a dot product do not care which order the features
    /// come in as long as the query and the key agree.</summary>
    private HostTensor PairsToHalves(HostTensor tensor)
    {
        int[] shape = tensor.Shape;
        int half = _headDim / 2;
        int rows = shape[0];
        int columns = shape.Length == 2 ? shape[1] : 1;
        float[] wide = tensor.ToFloats();
        var permuted = new float[wide.Length];
        for (int row = 0; row < rows; row++)
        {
            int head = row / _headDim, feature = row % _headDim;
            int source = head * _headDim + (feature < half ? 2 * feature : 2 * (feature - half) + 1);
            Array.Copy(wide, source * columns, permuted, row * columns, columns);
        }
        return HostTensor.FromFloats(permuted, shape).ConvertTo(tensor.DataType);
    }

    // --- Graph construction ------------------------------------------------

    private static DmlExpression Slice(DmlExpression x, int axis, int offset, int size)
    {
        var offsets = new int[x.Shape.Length];
        int[] sizes = x.Shape.Select(extent => (int)extent).ToArray();
        var strides = Enumerable.Repeat(1, x.Shape.Length).ToArray();
        offsets[axis] = offset;
        sizes[axis] = size;
        return DmlOps.Slice(x, offsets, sizes, strides);
    }

    /// <summary>One of the modulation's four vectors, spread over the
    /// tokens: the noise level's row over the image tokens, the zero-time
    /// row over the text tokens.</summary>
    private DmlExpression Rows(DmlExpression modulation, int chunk)
    {
        DmlExpression Row(int row) => Layers.Broadcast(
            DmlOps.Slice(modulation, new[] { 0, 0, row, chunk * _width }, new[] { 1, 1, 1, _width },
                new[] { 1, 1, 1, 1 }),
            new[] { 1u, 1u, (uint)(row == 0 ? Tokens : _contextRows), (uint)_width });
        return DmlOps.Join(new[] { Row(0), Row(1) }, axis: 2);
    }

    /// <summary>The per-head norm at the precision the blocks compute at: the
    /// mean-variance normalization accumulates a head's sum of squares wider
    /// than half, so no cast is needed around it.</summary>
    private DmlExpression HeadNorm(ModelBuilder model, DmlExpression x, HostTensor weight) =>
        Transformer.HeadNorm(model, x, PairsToHalves(weight), _heads, _headDim, NormEpsilon);

    private DmlExpression Rotate(DmlExpression heads, DmlExpression cos, DmlExpression sin) =>
        Transformer.Rotate(heads, cos, sin, _heads, _headDim, _compute);

    /// <summary>The block's attention over the whole sequence, cut into
    /// links: the projections with their heads normed and rotated; then the
    /// image rows' queries over every key under the count of real ones, a
    /// chunk of rows per link; then the text rows' queries over the text
    /// keys under the causal table, the pieces joined back in sequence
    /// order and projected out. Cut because a compiled graph's scratch is
    /// the sum of its intermediates, not their peak: one link for a whole
    /// block came to 9 GiB at 1024², and the attention's scores over 4096
    /// rows are most of it.</summary>
    private DmlExpression Attention(GraphChain chain, DmlExpression x,
        IReadOnlyDictionary<string, HostTensor> parameters, string prefix)
    {
        ModelBuilder model = chain.Current;
        DmlExpression cos = chain.Get(QwenImageConditioning.CosName);
        DmlExpression sin = chain.Get(QwenImageConditioning.SinName);
        DmlExpression query = Rotate(HeadNorm(model,
            Layers.Linear(model, x, PairsToHalves(parameters[$"{prefix}.to_q.weight"])),
            parameters[$"{prefix}.norm_q.weight"]), cos, sin);
        DmlExpression key = Rotate(HeadNorm(model,
            Layers.Linear(model, x, PairsToHalves(parameters[$"{prefix}.to_k.weight"])),
            parameters[$"{prefix}.norm_k.weight"]), cos, sin);
        chain.Set(Query, Transformer.Flat(query, _width));
        chain.Set(Key, Transformer.Flat(key, _width));
        chain.Set(Value, Layers.Linear(model, x, parameters[$"{prefix}.to_v.weight"]));
        chain.Cut();

        int tokens = Tokens;
        var chunks = new List<string>();
        for (int start = 0; start < tokens; start += QueryChunk)
        {
            int rows = Math.Min(QueryChunk, tokens - start);
            string name = $"attended{chunks.Count}";
            chain.Set(name, Layers.Attend(Slice(chain.Get(Query), 2, start, rows), chain.Get(Key),
                chain.Get(Value), _heads, chain.Get(QwenImageConditioning.KeyLengthName),
                MultiheadAttentionMaskType.KeySequenceLength));
            chunks.Add(name);
            chain.Cut();
        }

        model = chain.Current;
        DmlExpression text = Layers.Attend(
            Slice(chain.Get(Query), 2, tokens, _contextRows),
            Slice(chain.Get(Key), 2, tokens, _contextRows),
            Slice(chain.Get(Value), 2, tokens, _contextRows), _heads,
            Layers.Broadcast(chain.Get(QwenImageConditioning.TextMaskName),
                new[] { 1u, (uint)_heads, (uint)_contextRows, (uint)_contextRows }),
            MultiheadAttentionMaskType.Boolean);
        var pieces = chunks.Select(chain.Get).ToList();
        pieces.Add(text);
        DmlExpression attended = DmlOps.Join(pieces, axis: 2);
        chain.Drop(Query);
        chain.Drop(Key);
        chain.Drop(Value);
        foreach (string name in chunks)
        {
            chain.Drop(name);
        }
        return Layers.Linear(model, attended, parameters[$"{prefix}.to_out.0.weight"]);
    }

    /// <summary>SwiGLU: SiLU of the gate projection times the up
    /// projection, down to the width — the two projections one matrix when
    /// the file fused them.</summary>
    private DmlExpression FeedForward(ModelBuilder model, DmlExpression x,
        IReadOnlyDictionary<string, HostTensor> parameters, string prefix)
    {
        DmlExpression gate, up;
        if (parameters.TryGetValue($"{prefix}.{QwenImageCheckpoint.FusedGateUp}", out HostTensor? fused))
        {
            DmlExpression both = Layers.Linear(model, x, fused);
            gate = Slice(both, 3, 0, _mlpWidth);
            up = Slice(both, 3, _mlpWidth, _mlpWidth);
        }
        else
        {
            gate = Layers.Linear(model, x, parameters[$"{prefix}.{QwenImageCheckpoint.GateProjection}"]);
            up = Layers.Linear(model, x, parameters[$"{prefix}.{QwenImageCheckpoint.UpProjection}"]);
        }
        return Layers.Linear(model, Layers.Silu(gate) * up, parameters[$"{prefix}.img_mlp.out.weight"]);
    }

    /// <summary>One block, over several links (see <see cref="Attention"/>):
    /// the residual stream and the modulation are read again in the link
    /// that adds the attention back and runs the feed-forward.</summary>
    private void Block(GraphChain chain, IReadOnlyDictionary<string, HostTensor> parameters,
        string prefix)
    {
        DmlExpression normed = Transformer.Modulate(chain.Get(X), Rows(chain.Get(Modulation), 0),
            NormEpsilon, _compute);
        DmlExpression result = Attention(chain, normed, parameters, $"{prefix}.attn");

        ModelBuilder model = chain.Current;
        DmlExpression x = chain.Get(X);
        DmlExpression modulation = chain.Get(Modulation);
        x += Rows(modulation, 1) * Layers.Cast(result, Wide);

        normed = Transformer.Modulate(x, Rows(modulation, 2), NormEpsilon, _compute);
        result = FeedForward(model, normed, parameters, prefix);
        x += Rows(modulation, 3) * Layers.Cast(result, Wide);

        chain.Set(X, x);
    }

    private void Build(GraphChain chain, IReadOnlyDictionary<string, HostTensor> parameters)
    {
        int tokens = Tokens;
        int contextWidth = ContextWidth(parameters);
        chain.Input(Latent, new[] { 1, 1, tokens, LatentChannels }, Wide);
        chain.Input(Time, new[] { 1, 1, 2, _timeWidth }, Wide);
        chain.Input(QwenImageConditioning.ContextName, new[] { 1, 1, _contextRows, contextWidth }, Wide);
        chain.Input(QwenImageConditioning.TextMaskName, new[] { 1, 1, _contextRows, _contextRows },
            HostDataType.Int32);
        chain.Input(QwenImageConditioning.KeyLengthName, new[] { 1, 1, 1, 1 }, HostDataType.Int32);
        chain.Input(QwenImageConditioning.CosName, new[] { 1, tokens + _contextRows, 1, _headDim / 2 },
            HostDataType.Float32);
        chain.Input(QwenImageConditioning.SinName, new[] { 1, tokens + _contextRows, 1, _headDim / 2 },
            HostDataType.Float32);

        // The embeddings: the latent's cells and the text through their
        // projections into one sequence, kept wide from here on; the two
        // times — the noise level's and zero — through the embedder to the
        // shared modulation, its gates through tanh.
        ModelBuilder model = chain.Current;
        DmlExpression image = Layers.Linear(model, chain.Get(Latent), parameters["img_in.weight"]);
        DmlExpression context = chain.Get(QwenImageConditioning.ContextName);
        // A zero-centred RMS norm: the stored scale is the weight less one.
        float[] textScale = parameters["txt_in.text_norm.weight"].ToFloats();
        for (int i = 0; i < textScale.Length; i++)
        {
            textScale[i] += 1f;
        }
        DmlExpression text = Layers.RmsNorm(model, context, HostTensor.FromFloats(textScale, contextWidth),
            NormEpsilon);
        text = Layers.Linear(model, GeluTanh(Layers.Linear(model, text, parameters["txt_in.in_layer.weight"])),
            parameters["txt_in.out_layer.weight"]);
        chain.Set(X, Layers.Cast(DmlOps.Join(new[] { image, text }, axis: 2), Wide));
        chain.Drop(Latent);
        chain.Drop(QwenImageConditioning.ContextName);

        DmlExpression time = chain.Get(Time);
        DmlExpression temb = Layers.Linear(model,
            Layers.Silu(Layers.Linear(model, time, parameters["time_text_embed.timestep_embedder.linear_1.weight"])),
            parameters["time_text_embed.timestep_embedder.linear_2.weight"]);
        chain.Set(Temb, temb);
        DmlExpression modulation = Layers.Linear(model, Layers.Silu(temb), parameters["modulation.1.weight"]);
        chain.Set(Modulation, DmlOps.Join(new[]
        {
            Slice(modulation, 3, 0, _width),
            DmlOps.ActivationTanh(Slice(modulation, 3, _width, _width)),
            Slice(modulation, 3, 2 * _width, _width),
            DmlOps.ActivationTanh(Slice(modulation, 3, 3 * _width, _width)),
        }, axis: 3));
        chain.Drop(Time);
        chain.Cut();

        for (int block = 0; block < _blocks; block++)
        {
            Block(chain, parameters, $"transformer_blocks.{block}");
            chain.Cut();
        }

        // The final layer reads the image rows only, scaled from the noise
        // level's row of the time embedding.
        model = chain.Current;
        DmlExpression x = Slice(chain.Get(X), 2, 0, tokens);
        DmlExpression scale = Layers.Linear(model, Layers.Silu(Slice(chain.Get(Temb), 2, 0, 1)),
            parameters["norm_out.linear.weight"]);
        scale = Layers.Broadcast(scale, new[] { 1u, 1u, (uint)tokens, (uint)_width });
        chain.Set(Out, Layers.Linear(model, Transformer.Modulate(x, scale, NormEpsilon, Wide),
            parameters["proj_out.weight"]));
        chain.Drop(X);
        chain.Drop(Temb);
        chain.Drop(Modulation);
        chain.Drop(QwenImageConditioning.TextMaskName);
        chain.Drop(QwenImageConditioning.KeyLengthName);
        chain.Drop(QwenImageConditioning.CosName);
        chain.Drop(QwenImageConditioning.SinName);
        chain.Finish(Out);
    }

    /// <summary>The tanh approximation of GELU the text projection uses,
    /// x · σ(2·√(2/π)·(x + 0.044715·x³)) — the same function as
    /// 0.5·x·(1 + tanh(…)), in the operators there are.</summary>
    private static DmlExpression GeluTanh(DmlExpression x)
    {
        const float scale = 2f * 0.7978845608f;
        DmlExpression cubic = x * x * x * 0.044715f + x;
        return x * DmlOps.ActivationSigmoid(cubic * scale);
    }
}
