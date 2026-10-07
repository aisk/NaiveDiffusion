using NaiveDiffusion.Dml;
using NaiveDiffusion.Graph;
using NaiveDiffusion.Tensors;
using Vortice.DirectML;

namespace NaiveDiffusion.Cli;

/// <summary>Small graphs dispatched against CPU references — the proof that the
/// graph builder, the executor and each operator the SDXL graphs lean on all
/// hold up before any model is built over them.</summary>
public static class SmokeTest
{
    private static int _failures;

    public static int Run(DmlDevice device)
    {
        // First, while nothing has run on the device yet.
        RmsNormHalfRepeats(device);
        Gemm(device);
        GemmBroadcastWeight(device);
        ScaleBiasAndArithmetic(device);
        BroadcastAdd(device);
        TransposeView(device);
        Softmax(device);
        GroupNormalization(device);
        Convolution(device);
        Gather(device);
        JoinSliceUpsample(device);
        MultiheadAttention(device);
        MaskedAttention(device);
        PoolingAndTanh(device);
        OwnedWeights(device);
        ReduceAndPower(device);
        CastBothWays(device);
        RmsNormHalf(device);
        NormalizationVersion2(device);
        RotateHalf(device);
        MixedPrecisionBlock(device);
        Console.WriteLine(_failures == 0 ? "smoke: all passed" : $"smoke: {_failures} FAILED");
        return _failures == 0 ? 0 : 1;
    }

    private static void Check(string name, float[] actual, float[] expected, float tolerance = 1e-4f)
    {
        // The count is checked before the values: a graph that came back short
        // is one failure, not an exception ending the whole run.
        if (actual.Length != expected.Length)
        {
            Console.WriteLine($"  FAIL {name} ({actual.Length} values, expected {expected.Length})");
            _failures++;
            return;
        }
        float worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(actual[i] - expected[i]));
        }
        bool ok = worst <= tolerance;
        Console.WriteLine($"  {(ok ? "ok  " : "FAIL")} {name} (max error {worst:e2})");
        if (!ok)
        {
            _failures++;
        }
    }

    private static void Gemm(DmlDevice device)
    {
        var graph = new DmlGraph(device);
        DmlExpression a = graph.Input(TensorDataType.Float32, new uint[] { 1, 1, 2, 3 });
        DmlExpression b = graph.Input(TensorDataType.Float32, new uint[] { 1, 1, 3, 4 });
        using DmlCompiledModel model = graph.Compile(new[] { DmlOps.Gemm(a, b) });

        float[] aData = { 1, 2, 3, 4, 5, 6 };
        float[] bData = Enumerable.Range(0, 12).Select(v => (float)v).ToArray();
        HostTensor[] outputs = model.Dispatch(new Dictionary<DmlExpression, HostTensor>
        {
            [a] = HostTensor.FromFloats(aData, 1, 1, 2, 3),
            [b] = HostTensor.FromFloats(bData, 1, 1, 3, 4),
        });

        var expected = new float[8];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < 3; k++)
                    expected[i * 4 + j] += aData[i * 3 + k] * bData[k * 4 + j];
        Check("gemm", outputs[0].ToFloats(), expected);
    }

    private static void GemmBroadcastWeight(DmlDevice device)
    {
        // A linear layer under guidance batching: [2, 1, T, K] times a weight
        // broadcast across the batch by a zero stride.
        var graph = new DmlGraph(device);
        DmlExpression x = graph.Input(TensorDataType.Float32, new uint[] { 2, 1, 2, 3 });
        DmlExpression w = graph.Input(TensorDataType.Float32, new uint[] { 1, 1, 4, 3 });
        DmlExpression broadcast = DmlOps.Reinterpret(w, new uint[] { 2, 1, 4, 3 },
            new uint[] { 0, 12, 3, 1 });
        using DmlCompiledModel model = graph.Compile(new[]
        {
            DmlOps.Gemm(x, broadcast, transB: MatrixTransform.Transpose),
        });

        float[] xData = Enumerable.Range(0, 12).Select(v => (float)v).ToArray();
        float[] wData = Enumerable.Range(0, 12).Select(v => (float)(v % 5) - 2).ToArray();
        HostTensor[] outputs = model.Dispatch(new Dictionary<DmlExpression, HostTensor>
        {
            [x] = HostTensor.FromFloats(xData, 2, 1, 2, 3),
            [w] = HostTensor.FromFloats(wData, 1, 1, 4, 3),
        });

        var expected = new float[16];
        for (int batch = 0; batch < 2; batch++)
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 4; j++)
                    for (int k = 0; k < 3; k++)
                        expected[batch * 8 + i * 4 + j] +=
                            xData[batch * 6 + i * 3 + k] * wData[j * 3 + k];
        Check("gemm broadcast weight", outputs[0].ToFloats(), expected);
    }

    private static void ScaleBiasAndArithmetic(DmlDevice device)
    {
        var graph = new DmlGraph(device);
        DmlExpression x = graph.Input(TensorDataType.Float32, new uint[] { 1, 1, 1, 4 });
        DmlExpression y = 2.0f * x + 1.0f;      // scale-bias identities
        DmlExpression z = (x * y - x) / y;      // elementwise chain
        using DmlCompiledModel model = graph.Compile(new[] { z });

        float[] data = { 1, 2, 3, 4 };
        HostTensor[] outputs = model.Dispatch(new Dictionary<DmlExpression, HostTensor>
        {
            [x] = HostTensor.FromFloats(data, 1, 1, 1, 4),
        });
        float[] expected = data.Select(v =>
        {
            float w = 2 * v + 1;
            return (v * w - v) / w;
        }).ToArray();
        Check("scale-bias arithmetic", outputs[0].ToFloats(), expected);
    }

    private static void BroadcastAdd(DmlDevice device)
    {
        // A [1, C, 1, 1] bias broadcast over [1, C, H, W] by zero strides.
        var graph = new DmlGraph(device);
        DmlExpression image = graph.Input(TensorDataType.Float32, new uint[] { 1, 2, 2, 3 });
        DmlExpression bias = graph.Input(TensorDataType.Float32, new uint[] { 1, 2, 1, 1 });
        DmlExpression spread = DmlOps.Reinterpret(bias, new uint[] { 1, 2, 2, 3 },
            new uint[] { 2, 1, 0, 0 });
        using DmlCompiledModel model = graph.Compile(new[] { image + spread });

        float[] imageData = Enumerable.Range(0, 12).Select(v => (float)v).ToArray();
        float[] biasData = { 10, 20 };
        HostTensor[] outputs = model.Dispatch(new Dictionary<DmlExpression, HostTensor>
        {
            [image] = HostTensor.FromFloats(imageData, 1, 2, 2, 3),
            [bias] = HostTensor.FromFloats(biasData, 1, 2, 1, 1),
        });
        float[] expected = imageData.Select((v, i) => v + biasData[i / 6]).ToArray();
        Check("broadcast add", outputs[0].ToFloats(), expected);
    }

    private static void TransposeView(DmlDevice device)
    {
        // to_tokens: [1, C, H, W] viewed as [1, 1, H*W, C], packed by an identity.
        var graph = new DmlGraph(device);
        DmlExpression x = graph.Input(TensorDataType.Float32, new uint[] { 1, 3, 2, 2 });
        DmlExpression tokens = DmlOps.Reinterpret(x, new uint[] { 1, 1, 4, 3 },
            new uint[] { 12, 12, 1, 4 });
        using DmlCompiledModel model = graph.Compile(new[] { DmlOps.ActivationIdentity(tokens) });

        float[] data = Enumerable.Range(0, 12).Select(v => (float)v).ToArray();
        HostTensor[] outputs = model.Dispatch(new Dictionary<DmlExpression, HostTensor>
        {
            [x] = HostTensor.FromFloats(data, 1, 3, 2, 2),
        });
        var expected = new float[12];
        for (int token = 0; token < 4; token++)
            for (int channel = 0; channel < 3; channel++)
                expected[token * 3 + channel] = data[channel * 4 + token];
        Check("transpose view", outputs[0].ToFloats(), expected);
    }

    private static void Softmax(DmlDevice device)
    {
        var graph = new DmlGraph(device);
        DmlExpression x = graph.Input(TensorDataType.Float32, new uint[] { 1, 1, 2, 3 });
        using DmlCompiledModel model = graph.Compile(new[]
        {
            DmlOps.ActivationSoftmax(x, new[] { 3 }),
        });

        float[] data = { 1, 2, 3, 0, 0, 10 };
        HostTensor[] outputs = model.Dispatch(new Dictionary<DmlExpression, HostTensor>
        {
            [x] = HostTensor.FromFloats(data, 1, 1, 2, 3),
        });
        var expected = new float[6];
        for (int row = 0; row < 2; row++)
        {
            double sum = 0;
            for (int i = 0; i < 3; i++) sum += Math.Exp(data[row * 3 + i]);
            for (int i = 0; i < 3; i++) expected[row * 3 + i] = (float)(Math.Exp(data[row * 3 + i]) / sum);
        }
        Check("softmax", outputs[0].ToFloats(), expected);
    }

    private static void GroupNormalization(DmlDevice device)
    {
        // The group_norm recipe: view [1, C, H, W] as [1, G, C/G, H*W] and
        // normalize axes 2 and 3.
        var graph = new DmlGraph(device);
        DmlExpression x = graph.Input(TensorDataType.Float32, new uint[] { 1, 4, 1, 3 });
        DmlExpression grouped = DmlOps.Reinterpret(x, new uint[] { 1, 2, 2, 3 });
        DmlExpression normalized = DmlOps.MeanVarianceNormalization(grouped, new[] { 2, 3 }, 1e-5f);
        using DmlCompiledModel model = graph.Compile(new[]
        {
            DmlOps.Reinterpret(normalized, new uint[] { 1, 4, 1, 3 }),
        });

        float[] data = { 1, 2, 3, 4, 5, 6, 0, 0, 0, 1, 1, 4 };
        HostTensor[] outputs = model.Dispatch(new Dictionary<DmlExpression, HostTensor>
        {
            [x] = HostTensor.FromFloats(data, 1, 4, 1, 3),
        });

        var expected = new float[12];
        for (int group = 0; group < 2; group++)
        {
            double mean = 0, variance = 0;
            for (int i = 0; i < 6; i++) mean += data[group * 6 + i];
            mean /= 6;
            for (int i = 0; i < 6; i++) variance += Math.Pow(data[group * 6 + i] - mean, 2);
            variance /= 6;
            for (int i = 0; i < 6; i++)
                expected[group * 6 + i] = (float)((data[group * 6 + i] - mean) / Math.Sqrt(variance + 1e-5));
        }
        Check("group normalization", outputs[0].ToFloats(), expected);
    }

    private static void Convolution(DmlDevice device)
    {
        var graph = new DmlGraph(device);
        DmlExpression x = graph.Input(TensorDataType.Float32, new uint[] { 1, 1, 3, 3 });
        DmlExpression filter = graph.Input(TensorDataType.Float32, new uint[] { 2, 1, 3, 3 });
        DmlExpression bias = graph.Input(TensorDataType.Float32, new uint[] { 1, 2, 1, 1 });
        using DmlCompiledModel model = graph.Compile(new[]
        {
            DmlOps.Convolution(x, filter, bias,
                strides: new[] { 1, 1 }, startPadding: new[] { 1, 1 }, endPadding: new[] { 1, 1 }),
        });

        float[] xData = Enumerable.Range(0, 9).Select(v => (float)v).ToArray();
        var filterData = new float[18];
        filterData[4] = 1;                       // first filter: identity tap
        for (int i = 9; i < 18; i++) filterData[i] = 1;  // second: 3x3 box sum
        float[] biasData = { 0.5f, -1 };

        HostTensor[] outputs = model.Dispatch(new Dictionary<DmlExpression, HostTensor>
        {
            [x] = HostTensor.FromFloats(xData, 1, 1, 3, 3),
            [filter] = HostTensor.FromFloats(filterData, 2, 1, 3, 3),
            [bias] = HostTensor.FromFloats(biasData, 1, 2, 1, 1),
        });

        var expected = new float[18];
        for (int i = 0; i < 9; i++) expected[i] = xData[i] + 0.5f;
        for (int row = 0; row < 3; row++)
            for (int column = 0; column < 3; column++)
            {
                float sum = 0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int y = row + dy, s = column + dx;
                        if (y >= 0 && y < 3 && s >= 0 && s < 3) sum += xData[y * 3 + s];
                    }
                expected[9 + row * 3 + column] = sum - 1;
            }
        Check("convolution", outputs[0].ToFloats(), expected);
    }

    private static void Gather(DmlDevice device)
    {
        // The token-embedding lookup: [1, 1, vocab, width] indexed by [1, 1, 1, T].
        var graph = new DmlGraph(device);
        DmlExpression table = graph.Input(TensorDataType.Float32, new uint[] { 1, 1, 5, 2 });
        DmlExpression indices = graph.Input(TensorDataType.Uint32, new uint[] { 1, 1, 1, 3 });
        using DmlCompiledModel model = graph.Compile(new[]
        {
            DmlOps.Gather(table, indices, axis: 2, indexDimensions: 1),
        });

        float[] tableData = Enumerable.Range(0, 10).Select(v => (float)v).ToArray();
        uint[] indexData = { 4, 0, 2 };
        HostTensor[] outputs = model.Dispatch(new Dictionary<DmlExpression, HostTensor>
        {
            [table] = HostTensor.FromFloats(tableData, 1, 1, 5, 2),
            [indices] = HostTensor.FromUInt32(indexData, 1, 1, 1, 3),
        });
        float[] expected = { 8, 9, 0, 1, 4, 5 };
        Check("gather", outputs[0].ToFloats(), expected);
    }

    private static void JoinSliceUpsample(DmlDevice device)
    {
        var graph = new DmlGraph(device);
        DmlExpression a = graph.Input(TensorDataType.Float32, new uint[] { 1, 1, 2, 2 });
        DmlExpression b = graph.Input(TensorDataType.Float32, new uint[] { 1, 1, 2, 2 });
        DmlExpression joined = DmlOps.Join(new[] { a, b }, axis: 1);           // [1, 2, 2, 2]
        DmlExpression upsampled = DmlOps.Upsample2D(joined, 2, 2,
            InterpolationMode.NearestNeighbor);                                // [1, 2, 4, 4]
        DmlExpression cropped = DmlOps.Slice(upsampled,
            offsets: new[] { 0, 0, 0, 0 }, sizes: new[] { 1, 2, 3, 3 },
            strides: new[] { 1, 1, 1, 1 });                                    // [1, 2, 3, 3]
        using DmlCompiledModel model = graph.Compile(new[] { cropped });

        float[] aData = { 1, 2, 3, 4 };
        float[] bData = { 5, 6, 7, 8 };
        HostTensor[] outputs = model.Dispatch(new Dictionary<DmlExpression, HostTensor>
        {
            [a] = HostTensor.FromFloats(aData, 1, 1, 2, 2),
            [b] = HostTensor.FromFloats(bData, 1, 1, 2, 2),
        });

        var expected = new float[18];
        for (int channel = 0; channel < 2; channel++)
        {
            float[] source = channel == 0 ? aData : bData;
            for (int y = 0; y < 3; y++)
                for (int x = 0; x < 3; x++)
                    expected[channel * 9 + y * 3 + x] = source[(y / 2) * 2 + x / 2];
        }
        Check("join + upsample + slice", outputs[0].ToFloats(), expected);
    }

    private static void MultiheadAttention(DmlDevice device)
    {
        var graph = new DmlGraph(device);
        DmlExpression q = graph.Input(TensorDataType.Float32, new uint[] { 1, 1, 3, 4 });
        DmlExpression k = graph.Input(TensorDataType.Float32, new uint[] { 1, 1, 3, 4 });
        DmlExpression v = graph.Input(TensorDataType.Float32, new uint[] { 1, 1, 3, 4 });
        float scale = 1.0f / (float)Math.Sqrt(2);
        using DmlCompiledModel model = graph.Compile(new[]
        {
            DmlOps.MultiheadAttention(q, k, v, headCount: 2, scale),
        });

        var random = new Random(7);
        float[] Draw() => Enumerable.Range(0, 12).Select(_ => (float)random.NextDouble() - 0.5f).ToArray();
        float[] qData = Draw(), kData = Draw(), vData = Draw();

        HostTensor[] outputs = model.Dispatch(new Dictionary<DmlExpression, HostTensor>
        {
            [q] = HostTensor.FromFloats(qData, 1, 1, 3, 4),
            [k] = HostTensor.FromFloats(kData, 1, 1, 3, 4),
            [v] = HostTensor.FromFloats(vData, 1, 1, 3, 4),
        });

        // CPU reference: two heads of width 2 over 3 tokens.
        var expected = new float[12];
        for (int head = 0; head < 2; head++)
        {
            for (int i = 0; i < 3; i++)
            {
                var scores = new double[3];
                double max = double.MinValue;
                for (int j = 0; j < 3; j++)
                {
                    double dot = 0;
                    for (int d = 0; d < 2; d++)
                        dot += qData[i * 4 + head * 2 + d] * kData[j * 4 + head * 2 + d];
                    scores[j] = dot * scale;
                    max = Math.Max(max, scores[j]);
                }
                double sum = 0;
                for (int j = 0; j < 3; j++) { scores[j] = Math.Exp(scores[j] - max); sum += scores[j]; }
                for (int d = 0; d < 2; d++)
                {
                    double mixed = 0;
                    for (int j = 0; j < 3; j++) mixed += scores[j] / sum * vData[j * 4 + head * 2 + d];
                    expected[i * 4 + head * 2 + d] = (float)mixed;
                }
            }
        }
        Check("multihead attention", outputs[0].ToFloats(), expected);
    }

    /// <summary>The two masks the single-stream transformer attends under:
    /// a boolean table (here causal) over the keys, and a count of real
    /// leading keys for a padded key sequence — each against the same CPU
    /// softmax with the ruled-out keys left out.</summary>
    private static void MaskedAttention(DmlDevice device)
    {
        const int tokens = 4, width = 4, heads = 2, dim = width / heads;
        float scale = 1.0f / MathF.Sqrt(dim);
        var random = new Random(11);
        float[] Draw() => Enumerable.Range(0, tokens * width)
            .Select(_ => (float)random.NextDouble() - 0.5f).ToArray();
        float[] qData = Draw(), kData = Draw(), vData = Draw();

        float[] Reference(Func<int, int, bool> allowed)
        {
            var expected = new float[tokens * width];
            for (int head = 0; head < heads; head++)
            {
                for (int i = 0; i < tokens; i++)
                {
                    var scores = new double[tokens];
                    double max = double.MinValue;
                    for (int j = 0; j < tokens; j++)
                    {
                        if (!allowed(i, j))
                        {
                            scores[j] = double.NegativeInfinity;
                            continue;
                        }
                        double dot = 0;
                        for (int d = 0; d < dim; d++)
                            dot += qData[i * width + head * dim + d] * kData[j * width + head * dim + d];
                        scores[j] = dot * scale;
                        max = Math.Max(max, scores[j]);
                    }
                    double sum = 0;
                    for (int j = 0; j < tokens; j++) { scores[j] = Math.Exp(scores[j] - max); sum += scores[j]; }
                    for (int d = 0; d < dim; d++)
                    {
                        double mixed = 0;
                        for (int j = 0; j < tokens; j++) mixed += scores[j] / sum * vData[j * width + head * dim + d];
                        expected[i * width + head * dim + d] = (float)mixed;
                    }
                }
            }
            return expected;
        }

        float[] Run(Func<DmlGraph, (DmlExpression Input, DmlExpression Mask)> maskInput,
            HostTensor maskData, MultiheadAttentionMaskType maskType)
        {
            var graph = new DmlGraph(device);
            DmlExpression q = graph.Input(TensorDataType.Float32, new uint[] { 1, 1, tokens, width });
            DmlExpression k = graph.Input(TensorDataType.Float32, new uint[] { 1, 1, tokens, width });
            DmlExpression v = graph.Input(TensorDataType.Float32, new uint[] { 1, 1, tokens, width });
            (DmlExpression input, DmlExpression mask) = maskInput(graph);
            using DmlCompiledModel model = graph.Compile(new[]
            {
                Layers.Attend(q, k, v, heads, mask, maskType),
            });
            HostTensor[] outputs = model.Dispatch(new Dictionary<DmlExpression, HostTensor>
            {
                [q] = HostTensor.FromFloats(qData, 1, 1, tokens, width),
                [k] = HostTensor.FromFloats(kData, 1, 1, tokens, width),
                [v] = HostTensor.FromFloats(vData, 1, 1, tokens, width),
                [input] = maskData,
            });
            return outputs[0].ToFloats();
        }

        // DirectML wants the mask as int32, the boolean one as a full
        // [batch, heads, tokens, keys] — which a [1, 1, tokens, keys] table
        // reaches by a zero stride over the heads — and the key count as
        // [1, 1, 1, 1].
        var causal = new int[tokens * tokens];
        for (int i = 0; i < tokens; i++)
            for (int j = 0; j <= i; j++)
                causal[i * tokens + j] = 1;
        Check("boolean mask (causal), broadcast over heads",
            Run(graph =>
                {
                    DmlExpression table = graph.Input(TensorDataType.Int32, new uint[] { 1, 1, tokens, tokens });
                    return (table, Layers.Broadcast(table, new uint[] { 1, heads, tokens, tokens }));
                },
                HostTensor.FromInt32(causal, 1, 1, tokens, tokens), MultiheadAttentionMaskType.Boolean),
            Reference((i, j) => j <= i));

        const int realKeys = 3;
        Check("key-length mask (3 of 4 keys)",
            Run(graph =>
                {
                    DmlExpression count = graph.Input(TensorDataType.Int32, new uint[] { 1, 1, 1, 1 });
                    return (count, count);
                },
                HostTensor.FromInt32(new[] { realKeys }, 1, 1, 1, 1), MultiheadAttentionMaskType.KeySequenceLength),
            Reference((i, j) => j < realKeys));
    }

    /// <summary>The two operators the Wan 2.2 VAE's shortcuts and
    /// Qwen-Image's gates use: a 2×2 average pool with no padding, and
    /// tanh — each against the arithmetic on the CPU.</summary>
    private static void PoolingAndTanh(DmlDevice device)
    {
        const int channels = 2, height = 4, width = 6;
        var random = new Random(13);
        float[] data = Enumerable.Range(0, channels * height * width)
            .Select(_ => (float)random.NextDouble() * 4f - 2f).ToArray();

        var graph = new DmlGraph(device);
        DmlExpression input = graph.Input(TensorDataType.Float32, new uint[] { 1, channels, height, width });
        using DmlCompiledModel model = graph.Compile(new[]
        {
            DmlOps.AveragePooling(input, 2),
            DmlOps.ActivationTanh(input),
        });
        HostTensor[] outputs = model.Dispatch(new Dictionary<DmlExpression, HostTensor>
        {
            [input] = HostTensor.FromFloats(data, 1, channels, height, width),
        });

        var pooled = new float[channels * (height / 2) * (width / 2)];
        for (int c = 0; c < channels; c++)
            for (int i = 0; i < height / 2; i++)
                for (int j = 0; j < width / 2; j++)
                {
                    float sum = 0;
                    for (int m = 0; m < 2; m++)
                        for (int n = 0; n < 2; n++)
                            sum += data[(c * height + 2 * i + m) * width + 2 * j + n];
                    pooled[(c * (height / 2) + i) * (width / 2) + j] = sum / 4f;
                }
        Check("average pooling 2x2", outputs[0].ToFloats(), pooled);
        Check("tanh", outputs[1].ToFloats(), data.Select(MathF.Tanh).ToArray());
    }

    private static void OwnedWeights(DmlDevice device)
    {
        // The initialize-then-dispatch split: a weight handed over once.
        var graph = new DmlGraph(device);
        DmlExpression x = graph.Input(TensorDataType.Float32, new uint[] { 1, 1, 2, 2 });
        DmlExpression w = graph.Input(TensorDataType.Float32, new uint[] { 1, 1, 2, 2 }, owned: true);
        using DmlCompiledModel model = graph.Compile(new[] { DmlOps.Gemm(x, w) });

        float[] wData = { 1, 0, 0, 2 };
        model.Initialize(new Dictionary<DmlExpression, HostTensor>
        {
            [w] = HostTensor.FromFloats(wData, 1, 1, 2, 2),
        });

        for (int round = 0; round < 2; round++)
        {
            float[] xData = { 1 + round, 2, 3, 4 };
            HostTensor[] outputs = model.Dispatch(new Dictionary<DmlExpression, HostTensor>
            {
                [x] = HostTensor.FromFloats(xData, 1, 1, 2, 2),
            });
            float[] expected = { xData[0], 2 * xData[1], xData[2], 2 * xData[3] };
            Check($"owned weights round {round}", outputs[0].ToFloats(), expected);
        }
    }

    private static float[] Draw(int seed, int count, float spread = 1f)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, count).Select(_ => ((float)random.NextDouble() - 0.5f) * spread).ToArray();
    }

    /// <summary>The two operators RMSNorm is composed of, on their own.</summary>
    private static void ReduceAndPower(DmlDevice device)
    {
        var graph = new DmlGraph(device);
        DmlExpression x = graph.Input(TensorDataType.Float32, new uint[] { 1, 1, 3, 4 });
        DmlExpression squares = DmlOps.Reduce(x, new[] { 3 }, ReduceFunction.SumSquare);
        using DmlCompiledModel model = graph.Compile(new[]
        {
            squares, DmlOps.ConstantPow(squares, -0.5f, 0.25f, 1e-6f),
        });
        float[] data = Draw(3, 12, 2f);
        HostTensor[] outputs = model.Dispatch(new Dictionary<DmlExpression, HostTensor>
        {
            [x] = HostTensor.FromFloats(data, 1, 1, 3, 4),
        });
        var sums = new float[3];
        var inverse = new float[3];
        for (int m = 0; m < 3; m++)
        {
            for (int i = 0; i < 4; i++) sums[m] += data[m * 4 + i] * data[m * 4 + i];
            inverse[m] = 1.0f / MathF.Sqrt(sums[m] / 4 + 1e-6f);
        }
        Check("reduce sum of squares", outputs[0].ToFloats(), sums);
        Check("constant power", outputs[1].ToFloats(), inverse);
    }

    private static void CastBothWays(DmlDevice device)
    {
        var graph = new DmlGraph(device);
        DmlExpression x = graph.Input(TensorDataType.Float32, new uint[] { 1, 1, 2, 4 });
        DmlExpression half = DmlOps.Cast(x, TensorDataType.Float16);
        using DmlCompiledModel model = graph.Compile(new[]
        {
            half, DmlOps.Cast(half * 2.0f, TensorDataType.Float32),
        });
        float[] data = Draw(4, 8, 4f);
        HostTensor[] outputs = model.Dispatch(new Dictionary<DmlExpression, HostTensor>
        {
            [x] = HostTensor.FromFloats(data, 1, 1, 2, 4),
        });
        float[] rounded = data.Select(v => (float)(Half)v).ToArray();
        Check("cast to half", outputs[0].ToFloats(), rounded, 1e-3f);
        Check("cast back", outputs[1].ToFloats(), rounded.Select(v => (float)(Half)(v * 2)).ToArray(), 4e-3f);
    }

    /// <summary>Layers.RmsNorm at half precision with a weight constant, the
    /// way the transformer's head norms run.</summary>
    private static void RmsNormHalf(DmlDevice device)
    {
        using var model = new ModelBuilder(device, HostDataType.Float16);
        DmlExpression x = model.Placeholder(new[] { 1, 3, 2, 4 });
        float[] weight = { 0.5f, 1f, 1.5f, 2f };
        model.Compile(new[]
        {
            Layers.RmsNorm(model, x, HostTensor.FromFloats(weight, 4), 1e-6f),
        });
        float[] data = Draw(5, 24, 3f);
        HostTensor[] outputs = model.Run(HostTensor.FromFloats(data, 1, 3, 2, 4));
        var expected = new float[24];
        for (int row = 0; row < 6; row++)
        {
            double square = 0;
            for (int i = 0; i < 4; i++) square += data[row * 4 + i] * (double)data[row * 4 + i];
            float scale = (float)(1 / Math.Sqrt(square / 4 + 1e-6));
            for (int i = 0; i < 4; i++) expected[row * 4 + i] = data[row * 4 + i] * scale * weight[i];
        }
        Check("rms norm (half)", outputs[0].ToFloats(), expected, 1e-2f);
    }

    /// <summary>The half-precision RMSNorm dispatched twice on the same
    /// input, the first time being the first thing the device runs. A
    /// driver has rounded the first dispatch of a shader in a process
    /// differently from every later one, which made the first step of the
    /// first image differ from the same step of the second; the two have
    /// to agree to the bit. Enough values for a last-bit difference to show.</summary>
    private static void RmsNormHalfRepeats(DmlDevice device)
    {
        const int tokens = 256, heads = 16, dim = 128;
        using var model = new ModelBuilder(device, HostDataType.Float16);
        DmlExpression x = model.Placeholder(new[] { 1, tokens, heads, dim });
        model.Compile(new[]
        {
            Layers.Cast(Layers.RmsNorm(model, x, HostTensor.FromFloats(Draw(12, dim, 0.8f), dim), 1e-6f),
                HostDataType.Float32),
        });
        HostTensor input = HostTensor.FromFloats(Draw(11, tokens * heads * dim, 6f), 1, tokens, heads, dim);
        float[] first = model.Run(input)[0].ToFloats();
        float[] second = model.Run(input)[0].ToFloats();
        Check("rms norm (half), first dispatch against the second", first, second, 0f);
    }

    /// <summary>The second mean-variance normalization: RMS normalization with
    /// a weight that varies along the normalized axis, the same at half
    /// precision on values whose squares overflow half, and group
    /// normalization with its per-channel affine inside the operator.</summary>
    private static void NormalizationVersion2(DmlDevice device)
    {
        static float[] Rms(float[] data, int width, float[] weight, float epsilon)
        {
            var result = new float[data.Length];
            for (int row = 0; row < data.Length / width; row++)
            {
                double square = 0;
                for (int i = 0; i < width; i++) square += data[row * width + i] * (double)data[row * width + i];
                double scale = 1 / Math.Sqrt(square / width + epsilon);
                for (int i = 0; i < width; i++) result[row * width + i] = (float)(data[row * width + i] * scale * weight[i]);
            }
            return result;
        }

        float[] weight = { 0.5f, 1f, 1.5f, 2f };
        {
            var graph = new DmlGraph(device);
            DmlExpression x = graph.Input(TensorDataType.Float32, new uint[] { 1, 1, 3, 4 });
            DmlExpression w = graph.Input(TensorDataType.Float32, new uint[] { 1, 1, 1, 4 });
            DmlExpression h = DmlOps.Cast(x, TensorDataType.Float16);
            DmlExpression hw = DmlOps.Cast(w, TensorDataType.Float16);
            using DmlCompiledModel model = graph.Compile(new[]
            {
                DmlOps.MeanVarianceNormalization2(x, new[] { 3 }, false, 1e-6f, w),
                DmlOps.Cast(DmlOps.MeanVarianceNormalization2(h, new[] { 3 }, false, 1e-6f, hw),
                    TensorDataType.Float32),
            });
            float[] data = Draw(6, 12, 2f);
            // Half precision on values up to ±300: the sum of squares (up to
            // 4·90000) overflows half, which the composed RMSNorm cannot survive.
            float[] large = Draw(7, 12, 600f).Select(v => (float)(Half)v).ToArray();
            HostTensor[] small = model.Dispatch(new Dictionary<DmlExpression, HostTensor>
            {
                [x] = HostTensor.FromFloats(data, 1, 1, 3, 4),
                [w] = HostTensor.FromFloats(weight, 1, 1, 1, 4),
            });
            Check("mvn2 rms norm", small[0].ToFloats(), Rms(data, 4, weight, 1e-6f));
            HostTensor[] big = model.Dispatch(new Dictionary<DmlExpression, HostTensor>
            {
                [x] = HostTensor.FromFloats(large, 1, 1, 3, 4),
                [w] = HostTensor.FromFloats(weight, 1, 1, 1, 4),
            });
            Check("mvn2 rms norm (half, squares overflow half)", big[1].ToFloats(),
                Rms(large, 4, weight, 1e-6f), 1e-2f);
        }
        {
            // Group norm on the [1, G, C/G, H*W] view with the scale and bias
            // per channel, i.e. varying along a normalized axis.
            var graph = new DmlGraph(device);
            DmlExpression x = graph.Input(TensorDataType.Float32, new uint[] { 1, 2, 2, 3 });
            DmlExpression scale = graph.Input(TensorDataType.Float32, new uint[] { 1, 2, 2, 1 });
            DmlExpression bias = graph.Input(TensorDataType.Float32, new uint[] { 1, 2, 2, 1 });
            using DmlCompiledModel model = graph.Compile(new[]
            {
                DmlOps.MeanVarianceNormalization2(x, new[] { 2, 3 }, true, 1e-5f, scale, bias),
            });
            float[] data = { 1, 2, 3, 4, 5, 6, 0, 0, 0, 1, 1, 4 };
            float[] gamma = { 0.5f, 2f, -1f, 1.5f };
            float[] beta = { 0f, 1f, -2f, 0.25f };
            HostTensor[] outputs = model.Dispatch(new Dictionary<DmlExpression, HostTensor>
            {
                [x] = HostTensor.FromFloats(data, 1, 2, 2, 3),
                [scale] = HostTensor.FromFloats(gamma, 1, 2, 2, 1),
                [bias] = HostTensor.FromFloats(beta, 1, 2, 2, 1),
            });
            var expected = new float[12];
            for (int group = 0; group < 2; group++)
            {
                double mean = 0, variance = 0;
                for (int i = 0; i < 6; i++) mean += data[group * 6 + i];
                mean /= 6;
                for (int i = 0; i < 6; i++) variance += Math.Pow(data[group * 6 + i] - mean, 2);
                variance /= 6;
                for (int i = 0; i < 6; i++)
                {
                    int channel = group * 2 + i / 3;
                    expected[group * 6 + i] = (float)((data[group * 6 + i] - mean)
                        / Math.Sqrt(variance + 1e-5) * gamma[channel] + beta[channel]);
                }
            }
            Check("mvn2 group norm with per-channel affine", outputs[0].ToFloats(), expected);
        }
    }

    /// <summary>The rotate-half rotary composition: slices of the head axis
    /// combined with broadcast tables and joined back.</summary>
    private static void RotateHalf(DmlDevice device)
    {
        var graph = new DmlGraph(device);
        DmlExpression x = graph.Input(TensorDataType.Float32, new uint[] { 1, 3, 2, 4 });
        DmlExpression cos = graph.Input(TensorDataType.Float32, new uint[] { 1, 3, 1, 2 });
        DmlExpression sin = graph.Input(TensorDataType.Float32, new uint[] { 1, 3, 1, 2 });
        DmlExpression Half(int offset) => DmlOps.Slice(x, new[] { 0, 0, 0, offset },
            new[] { 1, 3, 2, 2 }, new[] { 1, 1, 1, 1 });
        DmlExpression c = Layers.Broadcast(cos, new uint[] { 1, 3, 2, 2 });
        DmlExpression s = Layers.Broadcast(sin, new uint[] { 1, 3, 2, 2 });
        DmlExpression low = Half(0), high = Half(2);
        using DmlCompiledModel model = graph.Compile(new[]
        {
            DmlOps.Join(new[] { low * c - high * s, high * c + low * s }, axis: 3),
        });
        float[] data = Draw(6, 24, 2f);
        float[] angles = Draw(7, 6, 6f);
        HostTensor[] outputs = model.Dispatch(new Dictionary<DmlExpression, HostTensor>
        {
            [x] = HostTensor.FromFloats(data, 1, 3, 2, 4),
            [cos] = HostTensor.FromFloats(angles.Select(MathF.Cos).ToArray(), 1, 3, 1, 2),
            [sin] = HostTensor.FromFloats(angles.Select(MathF.Sin).ToArray(), 1, 3, 1, 2),
        });
        var expected = new float[24];
        for (int t = 0; t < 3; t++)
        {
            for (int head = 0; head < 2; head++)
            {
                int offset = (t * 2 + head) * 4;
                for (int i = 0; i < 2; i++)
                {
                    float cs = MathF.Cos(angles[t * 2 + i]), sn = MathF.Sin(angles[t * 2 + i]);
                    expected[offset + i] = data[offset + i] * cs - data[offset + 2 + i] * sn;
                    expected[offset + 2 + i] = data[offset + 2 + i] * cs + data[offset + i] * sn;
                }
            }
        }
        Check("rotate half", outputs[0].ToFloats(), expected);
    }

    /// <summary>One DiT sub-block in miniature: a single-precision residual,
    /// its layer norm modulated, narrowed for a half-precision gemm, widened
    /// and gated back in.</summary>
    private static void MixedPrecisionBlock(DmlDevice device)
    {
        using var model = new ModelBuilder(device, HostDataType.Float16);
        DmlExpression x = model.Placeholder(new[] { 1, 1, 3, 4 }, HostDataType.Float32);
        DmlExpression shift = model.Placeholder(new[] { 1, 1, 1, 4 }, HostDataType.Float32);
        float[] weight = Draw(8, 16, 1f);
        DmlExpression normed = Layers.Cast(Layers.LayerNorm(x, 1e-6f) * 1.5f + Layers.Broadcast(shift, x.Shape),
            HostDataType.Float16);
        DmlExpression projected = Layers.Linear(model, normed, HostTensor.FromFloats(weight, 4, 4));
        model.Compile(new[] { x + Layers.Cast(projected, HostDataType.Float32) * 0.5f });

        float[] data = Draw(9, 12, 4f);
        float[] shiftData = Draw(10, 4, 1f);
        HostTensor[] outputs = model.Run(HostTensor.FromFloats(data, 1, 1, 3, 4),
            HostTensor.FromFloats(shiftData, 1, 1, 1, 4));
        var expected = new float[12];
        for (int m = 0; m < 3; m++)
        {
            double mean = 0, variance = 0;
            for (int i = 0; i < 4; i++) mean += data[m * 4 + i] / 4;
            for (int i = 0; i < 4; i++) variance += Math.Pow(data[m * 4 + i] - mean, 2) / 4;
            var n = new float[4];
            for (int i = 0; i < 4; i++)
                n[i] = (float)((data[m * 4 + i] - mean) / Math.Sqrt(variance + 1e-6) * 1.5 + shiftData[i]);
            for (int o = 0; o < 4; o++)
            {
                double dot = 0;
                for (int i = 0; i < 4; i++) dot += n[i] * weight[o * 4 + i];
                expected[m * 4 + o] = data[m * 4 + o] + (float)dot * 0.5f;
            }
        }
        Check("mixed precision block", outputs[0].ToFloats(), expected, 2e-2f);
    }
}
