using NaiveDiffusion.Dml;
using NaiveDiffusion.Graph;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Tensors;

namespace NaiveDiffusion.Models.Sdxl;

/// <summary>SDXL's UNet: a noisy latent, a timestep and the text conditioning
/// in, the predicted noise out. Runs at half precision as a chain of graphs —
/// one per resnet-and-attention pair at the levels that have attention, the
/// attention-free level folded into its neighbours — with the activations and
/// skip connections staying on the GPU between them.
///
/// The chain is what makes the memory tractable. DirectML folds a graph's
/// weights into one persistent buffer, and a single D3D12 buffer stops at
/// 4 GiB where the whole UNet is 4.78 GiB, so it had to be cut at least once;
/// cut a dozen times, no piece is over a gigabyte, the staging peak while
/// building is a piece rather than a half, and a piece that does not fit can
/// stay in system memory and be read from there (the resident budget).</summary>
public sealed class UNetModel : IDenoiser
{
    public static readonly int LatentChannels = SdxlVae.Latent.Channels;
    private static readonly int[] BlockOutChannels = { 320, 640, 1280 };
    private const int LayersPerBlock = 2;
    // Per resolution, going down. Level 0 has no attention at all.
    private static readonly int[] TransformerLayers = { 0, 2, 10 };
    private static readonly int[] AttentionHeads = { 5, 10, 20 };
    /// <summary>The width of the context the cross-attention reads: the two
    /// CLIP towers' hidden states side by side, 768 + 1280.</summary>
    public const int ContextWidth = 2048;
    private const int TimeEmbedDim = 1280;
    private const int AdditionTimeEmbedDim = 256;
    private const int NormGroups = 32;
    private const float NormEpsilon = 1e-5f;
    // Transformer2DModel hardcodes its GroupNorm epsilon, and it is not the one
    // the resnets use.
    private const float TransformerNormEpsilon = 1e-6f;
    /// <summary>torch's nn.LayerNorm default, which the transformer blocks use.</summary>
    private const float LayerNormEpsilon = 1e-5f;

    // DirectML lays a gemm's weight out for the shape it is about to be
    // multiplied by; when the row count is not a multiple of this it keeps a
    // second, repacked copy in the persistent resource. The rows are tokens,
    // so this is a property of the image size.
    private const int GemmRowAlignment = 64;

    // The names values travel under between links.
    private const string Sample = "sample";
    private const string TimeInput = "time";
    private const string AddInput = "add";
    private const string Context = "context";
    private const string Temb = "temb";
    private const string X = "x";
    private const string Noise = "noise";

    private readonly GraphChain _chain;

    /// <param name="residentBudget">Video memory to keep the weights in
    /// between steps; null keeps all of them.</param>
    /// <param name="int8Weights">Store the large matrices as block-quantized
    /// int8, see <see cref="ModelBuilder.Int8Weights"/>: about half the
    /// weight memory, at a small cost in accuracy.</param>
    public UNetModel(DmlDevice device, IReadOnlyDictionary<string, HostTensor> parameters,
        int height, int width, int tokens = 77, int batch = 1, ulong? residentBudget = null,
        bool int8Weights = false)
    {
        int[] latentShape = { batch, LatentChannels, height / SdxlVae.Latent.ScaleFactor, width / SdxlVae.Latent.ScaleFactor };
        _chain = new GraphChain(device, HostDataType.Float16, residentBudget, int8Weights);
        try
        {
            Build(_chain, parameters, latentShape, tokens);
        }
        catch
        {
            // The links built so far hold their weights on the card, and no
            // caller ever gets this object to give them back through.
            _chain.Dispose();
            throw;
        }
    }

    /// <summary>Predict the noise in <paramref name="latent"/> at
    /// <paramref name="timestep"/>, under one branch of
    /// <see cref="SdxlConditioning"/>. The UNet embeds the timestep; the
    /// level itself is not its concern.</summary>
    public HostTensor Predict(HostTensor latent, double timestep, double sigma,
        IReadOnlyDictionary<string, HostTensor> conditioning) =>
        Predict(latent, TimeConditioning(timestep), conditioning[SdxlConditioning.AddName],
            conditioning[SdxlConditioning.ContextName]);

    /// <summary>Predict the noise in <paramref name="latent"/>.</summary>
    public HostTensor Predict(HostTensor latent, HostTensor timeInput,
        HostTensor addInput, HostTensor context)
    {
        return _chain.Run(new Dictionary<string, HostTensor>
        {
            [Sample] = latent,
            [TimeInput] = timeInput,
            [AddInput] = addInput,
            [Context] = context,
        });
    }

    /// <summary>Memory the folded weights take, wherever they are.</summary>
    public ulong PersistentBytes => _chain.PersistentBytes;

    /// <summary>Of those, the bytes in system memory, read across the bus every step.</summary>
    public ulong StreamedBytes => _chain.StreamedBytes;

    /// <summary>And the bytes in video memory.</summary>
    public ulong ResidentBytes => _chain.ResidentBytes;

    /// <summary>Scratch one step needs, on top of the resident weights.</summary>
    public ulong TemporaryBytes => _chain.TemporaryBytes;

    /// <summary>How many graphs the model was cut into.</summary>
    public int GraphCount => _chain.LinkCount;

    public void Dispose() => _chain.Dispose();

    // --- Size diagnostics --------------------------------------------------

    /// <summary>Tokens the 1280-wide level sees — the row count of its gemms.
    /// A stride-2 convolution with one pixel of padding rounds up.</summary>
    private static int DeepestTokens(int height, int width)
    {
        int Extent(int size)
        {
            size /= 8;
            for (int i = 0; i < BlockOutChannels.Length - 1; i++)
            {
                size = (size + 1) / 2;
            }
            return size;
        }
        return Extent(height) * Extent(width);
    }

    /// <summary>Whether this size makes DirectML keep a second copy of the
    /// widest weights — at least 3.3 GiB more than the size needs.</summary>
    public static bool WeightsAreDuplicated(int height, int width) =>
        DeepestTokens(height, width) % GemmRowAlignment != 0;

    /// <summary>The closest height to <paramref name="height"/> that does not,
    /// searched in <paramref name="step"/>-pixel moves out to
    /// <paramref name="reach"/> and kept inside
    /// [<paramref name="minimum"/>, <paramref name="maximum"/>]. Null when
    /// nothing in range qualifies. The caller sets the grid: the CLI takes any
    /// multiple of 8, a front end may offer only multiples of 64.</summary>
    public static int? NearbyAligned(int height, int width, int step = 8, int reach = 256,
        int minimum = 1, int maximum = int.MaxValue)
    {
        for (int offset = step; offset <= reach; offset += step)
        {
            foreach (int candidate in new[] { height - offset, height + offset })
            {
                if (candidate >= minimum && candidate <= maximum &&
                    !WeightsAreDuplicated(candidate, width))
                {
                    return candidate;
                }
            }
        }
        return null;
    }

    // --- Conditioning, computed on the CPU ---------------------------------

    /// <summary>Sinusoidal timestep features. There are no weights here, and the
    /// timestep changes every step, so it stays out of the graph.</summary>
    public static float[] TimestepEmbedding(IReadOnlyList<double> timesteps, int dim,
        double maxPeriod = 10000)
    {
        int half = dim / 2;
        var embedding = new float[timesteps.Count * dim];
        for (int row = 0; row < timesteps.Count; row++)
        {
            for (int i = 0; i < half; i++)
            {
                double frequency = Math.Exp(-Math.Log(maxPeriod) * i / half);
                double angle = timesteps[row] * frequency;
                // flip_sin_to_cos: the cosine half leads.
                embedding[row * dim + i] = (float)Math.Cos(angle);
                embedding[row * dim + half + i] = (float)Math.Sin(angle);
            }
        }
        return embedding;
    }

    /// <summary>The timestep as the graph wants it alongside the latent: one
    /// row of sinusoids. Changes every step, unlike <see cref="AddConditioning"/>.</summary>
    public static HostTensor TimeConditioning(double timestep) =>
        HostTensor.FromFloats(TimestepEmbedding(new[] { timestep }, BlockOutChannels[0]),
            1, 1, 1, BlockOutChannels[0]);

    /// <summary>The other vector the graph wants. SDXL conditions on the
    /// resolution it is pretending to have been cropped from as well as on the
    /// prompt — original size, crop offset and target size go in as sinusoids,
    /// concatenated with the pooled text embedding. Fixed for a run.</summary>
    public static HostTensor AddConditioning(float[] pooledEmbeds,
        (int Height, int Width) originalSize, (int Top, int Left) crop,
        (int Height, int Width) targetSize)
    {
        var timeIds = new double[]
        {
            originalSize.Height, originalSize.Width, crop.Top, crop.Left,
            targetSize.Height, targetSize.Width,
        };
        float[] timeIdsEmbedding = TimestepEmbedding(timeIds, AdditionTimeEmbedDim);

        var addInput = new float[pooledEmbeds.Length + timeIdsEmbedding.Length];
        pooledEmbeds.CopyTo(addInput, 0);
        timeIdsEmbedding.CopyTo(addInput, pooledEmbeds.Length);
        return HostTensor.FromFloats(addInput, 1, 1, 1, addInput.Length);
    }

    // --- Graph construction ------------------------------------------------

    /// <summary>Linear, SiLU, Linear — how both embeddings reach the resnets' width.</summary>
    private static DmlExpression EmbeddingMlp(ModelBuilder model, DmlExpression x,
        IReadOnlyDictionary<string, HostTensor> parameters, string prefix)
    {
        x = Layers.Linear(model, x, parameters[$"{prefix}.linear_1.weight"],
            parameters[$"{prefix}.linear_1.bias"]);
        return Layers.Linear(model, Layers.Silu(x), parameters[$"{prefix}.linear_2.weight"],
            parameters[$"{prefix}.linear_2.bias"]);
    }

    /// <summary>The VAE's resnet plus a timestep term added between the convolutions.</summary>
    private static DmlExpression ResnetBlock(ModelBuilder model, DmlExpression x,
        DmlExpression temb, IReadOnlyDictionary<string, HostTensor> parameters, string prefix)
    {
        DmlExpression h = Layers.Conv2d(model,
            Layers.Silu(Layers.GroupNorm(model, x, parameters[$"{prefix}.norm1.weight"],
                parameters[$"{prefix}.norm1.bias"], NormGroups, NormEpsilon)),
            parameters[$"{prefix}.conv1.weight"], parameters[$"{prefix}.conv1.bias"]);

        DmlExpression projected = Layers.Linear(model, Layers.Silu(temb),
            parameters[$"{prefix}.time_emb_proj.weight"],
            parameters[$"{prefix}.time_emb_proj.bias"]);
        h += Layers.Broadcast(Layers.ToChannels(projected), h.Shape);

        h = Layers.Conv2d(model,
            Layers.Silu(Layers.GroupNorm(model, h, parameters[$"{prefix}.norm2.weight"],
                parameters[$"{prefix}.norm2.bias"], NormGroups, NormEpsilon)),
            parameters[$"{prefix}.conv2.weight"], parameters[$"{prefix}.conv2.bias"]);

        if (parameters.ContainsKey($"{prefix}.conv_shortcut.weight"))
        {
            x = Layers.Conv2d(model, x, parameters[$"{prefix}.conv_shortcut.weight"],
                parameters[$"{prefix}.conv_shortcut.bias"], padding: 0);
        }
        return x + h;
    }

    /// <summary>Attend over the image, cross-attend to the text, then a gated
    /// feed-forward — all residual.</summary>
    private static DmlExpression TransformerBlock(ModelBuilder model, DmlExpression x,
        DmlExpression context, IReadOnlyDictionary<string, HostTensor> parameters,
        string prefix, int heads)
    {
        x += UNetLayers.DiffusersAttention(model,
            Layers.LayerNorm(model, x, parameters[$"{prefix}.norm1.weight"],
                parameters[$"{prefix}.norm1.bias"], LayerNormEpsilon),
            parameters, $"{prefix}.attn1", heads);

        x += UNetLayers.DiffusersAttention(model,
            Layers.LayerNorm(model, x, parameters[$"{prefix}.norm2.weight"],
                parameters[$"{prefix}.norm2.bias"], LayerNormEpsilon),
            parameters, $"{prefix}.attn2", heads, context);

        return x + UNetLayers.Geglu(model,
            Layers.LayerNorm(model, x, parameters[$"{prefix}.norm3.weight"],
                parameters[$"{prefix}.norm3.bias"], LayerNormEpsilon),
            parameters, $"{prefix}.ff");
    }

    /// <summary>A stack of transformer blocks wrapped in a norm, a reshape and a residual.</summary>
    private static DmlExpression Transformer2d(ModelBuilder model, DmlExpression x,
        DmlExpression context, IReadOnlyDictionary<string, HostTensor> parameters,
        string prefix, int heads, int layers)
    {
        uint height = x.Shape[2], width = x.Shape[3];
        DmlExpression residual = x;

        DmlExpression tokens = Layers.ToTokens(Layers.GroupNorm(model, x,
            parameters[$"{prefix}.norm.weight"], parameters[$"{prefix}.norm.bias"],
            NormGroups, TransformerNormEpsilon));
        tokens = Layers.Linear(model, tokens, parameters[$"{prefix}.proj_in.weight"],
            parameters[$"{prefix}.proj_in.bias"]);

        for (int i = 0; i < layers; i++)
        {
            tokens = TransformerBlock(model, tokens, context, parameters,
                $"{prefix}.transformer_blocks.{i}", heads);
        }

        tokens = Layers.Linear(model, tokens, parameters[$"{prefix}.proj_out.weight"],
            parameters[$"{prefix}.proj_out.bias"]);
        return Layers.ToImage(tokens, height, width) + residual;
    }

    /// <summary>A resnet, then the transformers if this level has any: the
    /// unit the chain is cut at. Reads the running value under
    /// <paramref name="input"/> and leaves it under <paramref name="output"/>.</summary>
    private static void Unit(GraphChain chain, string input, string output,
        IReadOnlyDictionary<string, HostTensor> parameters, string prefix, int index,
        int heads, int layers)
    {
        ModelBuilder model = chain.Current;
        DmlExpression x = ResnetBlock(model, chain.Get(input), chain.Get(Temb), parameters,
            $"{prefix}.resnets.{index}");
        if (layers > 0)
        {
            x = Transformer2d(model, x, chain.Get(Context), parameters,
                $"{prefix}.attentions.{index}", heads, layers);
        }
        chain.Set(output, x);
    }

    /// <summary>Whether the chain is cut after every unit at this level. The
    /// attention-free level's units are a few megabytes of convolutions; a
    /// graph of their own would buy nothing but another fence to wait on.</summary>
    private static bool CutPerUnit(int level) => TransformerLayers[level] > 0;

    private static string Skip(int index) => $"skip{index}";

    /// <summary>The whole model: the embeddings and the input convolution,
    /// three levels down, the mid block, three levels up, the output
    /// convolution. Each level's skip connections are named as they are made
    /// and dropped as they are consumed, so their tensors live exactly as long
    /// as the U shape needs them.</summary>
    private static void Build(GraphChain chain,
        IReadOnlyDictionary<string, HostTensor> parameters, int[] latentShape, int tokens)
    {
        int batch = latentShape[0];
        chain.Input(Sample, latentShape);
        chain.Input(TimeInput, new[] { batch, 1, 1, BlockOutChannels[0] });
        chain.Input(AddInput, new[] { batch, 1, 1, TimeEmbedDim + 6 * AdditionTimeEmbedDim });
        chain.Input(Context, new[] { batch, 1, tokens, ContextWidth });

        ModelBuilder model = chain.Current;
        chain.Set(Temb, EmbeddingMlp(model, chain.Get(TimeInput), parameters, "time_embedding")
                      + EmbeddingMlp(model, chain.Get(AddInput), parameters, "add_embedding"));
        chain.Drop(TimeInput);
        chain.Drop(AddInput);

        int skips = 0;
        chain.Set(Skip(skips++), Layers.Conv2d(model, chain.Get(Sample),
            parameters["conv_in.weight"], parameters["conv_in.bias"]));
        chain.Drop(Sample);

        // Down: each unit's output is a skip connection, and the running value
        // is simply the newest skip.
        int levels = BlockOutChannels.Length;
        for (int level = 0; level < levels; level++)
        {
            string prefix = $"down_blocks.{level}";
            for (int i = 0; i < LayersPerBlock; i++)
            {
                Unit(chain, Skip(skips - 1), Skip(skips), parameters, prefix, i,
                    AttentionHeads[level], TransformerLayers[level]);
                skips++;
                if (CutPerUnit(level))
                {
                    chain.Cut();
                }
            }
            if (level != levels - 1)
            {
                chain.Set(Skip(skips), Layers.Conv2d(chain.Current, chain.Get(Skip(skips - 1)),
                    parameters[$"{prefix}.downsamplers.0.conv.weight"],
                    parameters[$"{prefix}.downsamplers.0.conv.bias"], stride: 2, padding: 1));
                skips++;
            }
        }
        if (!CutPerUnit(levels - 1))
        {
            chain.Cut();
        }

        // Mid.
        model = chain.Current;
        DmlExpression x = ResnetBlock(model, chain.Get(Skip(skips - 1)), chain.Get(Temb),
            parameters, "mid_block.resnets.0");
        x = Transformer2d(model, x, chain.Get(Context), parameters, "mid_block.attentions.0",
            AttentionHeads[^1], TransformerLayers[^1]);
        chain.Set(X, ResnetBlock(model, x, chain.Get(Temb), parameters, "mid_block.resnets.1"));
        chain.Cut();

        // Up: every unit first joins the newest remaining skip onto the running
        // value, which is the last time that skip is read.
        for (int i = 0; i < levels; i++)
        {
            int level = levels - 1 - i;
            string prefix = $"up_blocks.{i}";
            for (int j = 0; j < LayersPerBlock + 1; j++)
            {
                string skip = Skip(--skips);
                chain.Set(X, DmlOps.Join(new[] { chain.Get(X), chain.Get(skip) }, axis: 1));
                chain.Drop(skip);
                Unit(chain, X, X, parameters, prefix, j, AttentionHeads[level],
                    TransformerLayers[level]);
                if (CutPerUnit(level) && (j != LayersPerBlock || i == levels - 1))
                {
                    chain.Cut();
                }
            }

            if (i != levels - 1)
            {
                // A level that halved an odd extent cannot get back by doubling,
                // so the upsample is trimmed to the next skip connection's size
                // before the convolution — what diffusers does by handing
                // Upsample2D an output size.
                int[] next = chain.Shape(Skip(skips - 1));
                chain.Set(X, Layers.Conv2d(chain.Current,
                    Layers.CropTo(Layers.UpsampleNearest(chain.Get(X)), (uint)next[2], (uint)next[3]),
                    parameters[$"{prefix}.upsamplers.0.conv.weight"],
                    parameters[$"{prefix}.upsamplers.0.conv.bias"]));
                if (CutPerUnit(level))
                {
                    chain.Cut();
                }
            }
        }

        model = chain.Current;
        x = Layers.GroupNorm(model, chain.Get(X), parameters["conv_norm_out.weight"],
            parameters["conv_norm_out.bias"], NormGroups, NormEpsilon);
        chain.Set(Noise, Layers.Conv2d(model, Layers.Silu(x), parameters["conv_out.weight"],
            parameters["conv_out.bias"]));
        chain.Drop(X);
        chain.Drop(Temb);
        chain.Drop(Context);
        chain.Finish(Noise);
    }
}
