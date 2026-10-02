using System.Diagnostics;
using NaiveDiffusion.Dml;
using NaiveDiffusion.Graph;
using NaiveDiffusion.Tensors;
using Vortice.DirectML;

namespace NaiveDiffusion.Cli;

/// <summary>Anima's self-attention at 1024² — 4096 tokens, 16 heads of 128 —
/// written several ways, each at single and half precision, to see which one
/// the device runs fastest: DirectML's attention operator over separate
/// query, key and value (what the transformers use now), the same operator
/// over one stacked tensor (the layout fused driver kernels tend to want),
/// that stacked tensor joined inside the graph from the separate three (what
/// a model would have to do after its norms and rotations), and the explicit
/// GEMM, softmax, GEMM over every head at once and over groups of heads.
/// Inputs already sit in video memory and outputs stay there, so the times
/// are the dispatches alone; every output is compared against the separate
/// single-precision operator's.</summary>
public static class AttentionProbe
{
    private const int Tokens = 4096;
    private const int Heads = 16;
    private const int HeadSize = 128;
    private const int Width = Heads * HeadSize;
    private const int Rounds = 10;

    private delegate DmlExpression Build(DmlGraph graph, HostDataType type,
        out DmlExpression[] inputs);

    public static int Run(DmlDevice device)
    {
        Console.WriteLine($"self-attention, {Tokens} tokens, {Heads} heads of {HeadSize}:");
        var random = new Random(7);
        // Unit root-mean-square entries, as a head comes out of its RMSNorm,
        // so the softmax is neither flat nor one-hot.
        float[] Draw() => Enumerable.Range(0, Tokens * Width)
            .Select(_ => (float)(random.NextDouble() * 2 - 1) * MathF.Sqrt(3)).ToArray();
        float[] q = Draw(), k = Draw(), v = Draw();
        float[] stacked = Stack(q, k, v);

        var variants = new (string Name, Build Build, float[][] Data)[]
        {
            ("operator, separate q/k/v", Separate, new[] { q, k, v }),
            ("operator, stacked qkv input", Stacked, new[] { stacked }),
            ("operator, qkv joined in graph", Joined, new[] { q, k, v }),
            ("gemm-softmax-gemm, 16 heads", (DmlGraph g, HostDataType t, out DmlExpression[] i) =>
                Explicit(g, t, Heads, out i), new[] { q, k, v }),
            ("gemm-softmax-gemm, 4 x 4 heads", (DmlGraph g, HostDataType t, out DmlExpression[] i) =>
                Explicit(g, t, 4, out i), new[] { q, k, v }),
        };

        float[]? reference = null;
        foreach (HostDataType type in new[] { HostDataType.Float32, HostDataType.Float16 })
        {
            Console.WriteLine(type == HostDataType.Float32 ? "fp32:" : "fp16:");
            foreach ((string name, Build build, float[][] data) in variants)
            {
                try
                {
                    float[] output = Measure(device, name, build, type, data, out double milliseconds);
                    reference ??= output;
                    Console.WriteLine($"      {AttentionFlops / milliseconds / 1e9:0.0} TFLOPS, " +
                                      $"rel rms vs fp32 operator {RelativeRms(output, reference):0.00000}");
                }
                catch (Exception error) when (error is not OutOfMemoryException)
                {
                    Console.WriteLine($"  {name}: failed, {error.GetType().Name}: {error.Message}");
                }
                device.ReleasePooledBuffers();
            }
            // What the device makes of a plain GEMM the size of one of the
            // block's projections, to read the attention's rate against.
            Measure(device, "reference gemm [4096, 2048] x [2048, 2048]", Projection, type,
                new[] { q, k[..(Width * Width)] }, out double gemm);
            Console.WriteLine($"      {2.0 * Tokens * Width * Width / gemm / 1e9:0.0} TFLOPS");
            device.ReleasePooledBuffers();
        }
        return 0;
    }

    private static float[] Measure(DmlDevice device, string name, Build build,
        HostDataType type, float[][] data, out double milliseconds)
    {
        var graph = new DmlGraph(device);
        DmlExpression output = build(graph, type, out DmlExpression[] inputs);
        using DmlCompiledModel model = graph.Compile(new[] { Layers.Cast(output, HostDataType.Float32) });

        var bound = new Dictionary<DmlExpression, DeviceTensor>();
        try
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                int[] shape = inputs[i].Shape.Select(extent => (int)extent).ToArray();
                bound[inputs[i]] = InVideoMemory(device,
                    HostTensor.FromFloats(data[i], shape).ConvertTo(type));
            }

            HostTensor result;
            using (DeviceTensor first = model.DispatchOnDevice(bound)[0])
            {
                result = first.Download();
            }
            model.DispatchOnDevice(bound)[0].Dispose();
            var watch = Stopwatch.StartNew();
            for (int round = 0; round < Rounds; round++)
            {
                model.DispatchOnDevice(bound)[0].Dispose();
            }
            milliseconds = watch.Elapsed.TotalMilliseconds / Rounds;
            Console.WriteLine($"  {name}: {milliseconds:0.0} ms, " +
                              $"temporary {model.TemporarySize / (double)(1 << 20):0} MiB");
            return result.ToFloats();
        }
        finally
        {
            foreach (DeviceTensor tensor in bound.Values)
            {
                tensor.Dispose();
            }
        }
    }

    /// <summary>What the transformers do now.</summary>
    private static DmlExpression Separate(DmlGraph graph, HostDataType type,
        out DmlExpression[] inputs)
    {
        inputs = Enumerable.Range(0, 3)
            .Select(_ => graph.Input(type, new[] { 1, 1, Tokens, Width })).ToArray();
        return DmlOps.MultiheadAttention(inputs[0], inputs[1], inputs[2], Heads, Scale);
    }

    private static DmlExpression Stacked(DmlGraph graph, HostDataType type,
        out DmlExpression[] inputs)
    {
        inputs = new[] { graph.Input(type, new[] { 1, Tokens, Heads, 3, HeadSize }) };
        return DmlOps.MultiheadAttention(inputs[0], Heads, Scale);
    }

    private static DmlExpression Joined(DmlGraph graph, HostDataType type,
        out DmlExpression[] inputs)
    {
        inputs = Enumerable.Range(0, 3)
            .Select(_ => graph.Input(type, new[] { 1, 1, Tokens, Width })).ToArray();
        DmlExpression stacked = DmlOps.Join(inputs
            .Select(input => DmlOps.Reinterpret(input, new[] { 1, Tokens, Heads, 1, HeadSize }))
            .ToArray(), axis: 3);
        return DmlOps.MultiheadAttention(stacked, Heads, Scale);
    }

    /// <summary>Scores, softmax and mix as three operators over a view that
    /// puts the heads in front, <paramref name="group"/> heads to a set of
    /// operators, the heads joined and packed back to tokens by width.</summary>
    private static DmlExpression Explicit(DmlGraph graph, HostDataType type, int group,
        out DmlExpression[] inputs)
    {
        inputs = Enumerable.Range(0, 3)
            .Select(_ => graph.Input(type, new[] { 1, 1, Tokens, Width })).ToArray();
        DmlExpression[] byHead = inputs
            .Select(input => DmlOps.Reinterpret(input, new[] { 1, Heads, Tokens, HeadSize },
                new[] { Tokens * Width, HeadSize, Width, 1 }))
            .ToArray();

        var parts = new List<DmlExpression>();
        for (int first = 0; first < Heads; first += group)
        {
            DmlExpression[] slice = byHead.Select(view => group == Heads
                ? view
                : DmlOps.Slice(view, new[] { 0, first, 0, 0 }, new[] { 1, group, Tokens, HeadSize },
                    new[] { 1, 1, 1, 1 })).ToArray();
            DmlExpression scores = DmlOps.Gemm(slice[0], slice[1],
                transB: MatrixTransform.Transpose, alpha: Scale);
            DmlExpression weights = DmlOps.ActivationSoftmax(scores, new[] { 3 });
            parts.Add(DmlOps.Gemm(weights, slice[2]));
        }
        DmlExpression heads = parts.Count == 1 ? parts[0] : DmlOps.Join(parts, axis: 1);
        DmlExpression byToken = DmlOps.Identity(DmlOps.Reinterpret(heads,
            new[] { 1, Tokens, Heads, HeadSize }, new[] { Tokens * Width, HeadSize, Tokens * HeadSize, 1 }));
        return DmlOps.Reinterpret(byToken, new[] { 1, 1, Tokens, Width });
    }

    private static DmlExpression Projection(DmlGraph graph, HostDataType type,
        out DmlExpression[] inputs)
    {
        inputs = new[]
        {
            graph.Input(type, new[] { 1, 1, Tokens, Width }),
            graph.Input(type, new[] { 1, 1, Width, Width }),
        };
        return DmlOps.Gemm(inputs[0], inputs[1], transB: MatrixTransform.Transpose);
    }

    /// <summary>Two GEMMs of tokens by tokens by head size, per head.</summary>
    private const double AttentionFlops = 2.0 * 2 * Heads * (double)Tokens * Tokens * HeadSize;

    private static float Scale => 1.0f / MathF.Sqrt(HeadSize);

    /// <summary>[tokens, heads, 3, headSize] from three [tokens, heads * headSize].</summary>
    private static float[] Stack(float[] q, float[] k, float[] v)
    {
        var stacked = new float[3 * Tokens * Width];
        float[][] parts = { q, k, v };
        for (int token = 0; token < Tokens; token++)
        {
            for (int head = 0; head < Heads; head++)
            {
                for (int part = 0; part < 3; part++)
                {
                    Array.Copy(parts[part], token * Width + head * HeadSize, stacked,
                        ((token * Heads + head) * 3 + part) * HeadSize, HeadSize);
                }
            }
        }
        return stacked;
    }

    /// <summary>Upload lands in system memory, which the GPU would read across
    /// the bus on every dispatch; one copy through a graph moves it over.</summary>
    private static DeviceTensor InVideoMemory(DmlDevice device, HostTensor tensor)
    {
        var graph = new DmlGraph(device);
        DmlExpression input = graph.Input(tensor.DataType, tensor.Shape);
        using DmlCompiledModel copy = graph.Compile(new[] { DmlOps.Identity(input) });
        using DeviceTensor staged = device.Upload(tensor);
        return copy.DispatchOnDevice(new Dictionary<DmlExpression, DeviceTensor> { [input] = staged })[0];
    }

    private static double RelativeRms(float[] output, float[] reference)
    {
        double error = 0, scale = 0;
        for (int i = 0; i < output.Length; i++)
        {
            double difference = output[i] - reference[i];
            error += difference * difference;
            scale += (double)reference[i] * reference[i];
        }
        return Math.Sqrt(error / scale);
    }
}
