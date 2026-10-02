using System.Diagnostics;
using NaiveDiffusion.Dml;
using NaiveDiffusion.Tensors;
using Vortice.DirectML;

namespace NaiveDiffusion.Cli;

/// <summary>One linear layer three ways, to answer whether DirectML keeps
/// block-quantized weights quantized: fp16 weights folded into the persistent
/// resource, int8 weights plus scales dequantized in the graph and folded, and
/// the same int8 graph with nothing folded. If the initializer constant-folds
/// the dequantize, the second case's persistent size comes out at the fp16
/// size and quantizing saves no video memory.</summary>
public static class QuantProbe
{
    private const int BlockSize = 32;

    public static int Run(DmlDevice device)
    {
        bool ok = true;
        ok &= Probe(device, rows: 4096, outFeatures: 1280, inFeatures: 1280);
        ok &= Probe(device, rows: 4096, outFeatures: 5120, inFeatures: 1280);
        Console.WriteLine(ok ? "quantprobe: outputs agree" : "quantprobe: outputs DISAGREE");
        return ok ? 0 : 1;
    }

    private static bool Probe(DmlDevice device, int rows, int outFeatures, int inFeatures)
    {
        Console.WriteLine($"linear {outFeatures}x{inFeatures}, {rows} rows:");
        var random = new Random(7);
        float[] weight = Enumerable.Range(0, outFeatures * inFeatures)
            .Select(_ => (float)(random.NextDouble() * 2 - 1) * 0.05f).ToArray();
        float[] x = Enumerable.Range(0, rows * inFeatures)
            .Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();

        (sbyte[] quantized, Half[] scales) = Quantize(weight, outFeatures, inFeatures);
        HostTensor xTensor = HostTensor.FromFloats(x, 1, 1, rows, inFeatures).ConvertTo(HostDataType.Float16);
        HostTensor wTensor = HostTensor.FromFloats(weight, 1, 1, outFeatures, inFeatures).ConvertTo(HostDataType.Float16);
        HostTensor qTensor = HostTensor.FromInt8(quantized, 1, 1, outFeatures, inFeatures);
        HostTensor sTensor = HostTensor.FromHalves(scales, 1, 1, outFeatures, inFeatures / BlockSize);

        ulong fp16Bytes = (ulong)wTensor.Data.Length;
        ulong int8Bytes = (ulong)(qTensor.Data.Length + sTensor.Data.Length);
        Console.WriteLine($"  weight bytes: fp16 {Mib(fp16Bytes)}, int8+scales {Mib(int8Bytes)}");

        float[] reference = Fp16Gemm(device, xTensor, wTensor, out float[] fp16Out);
        float[] quantizedOut = Int8Gemm(device, xTensor, qTensor, sTensor, owned: true);
        Int8Gemm(device, xTensor, qTensor, sTensor, owned: false);

        // Quantization error against the exact product, and against the fp16
        // graph, which carries fp16 rounding of its own.
        float scale = reference.Max(MathF.Abs);
        float fp16Error = fp16Out.Zip(reference).Max(pair => MathF.Abs(pair.First - pair.Second)) / scale;
        float int8Error = quantizedOut.Zip(reference).Max(pair => MathF.Abs(pair.First - pair.Second)) / scale;
        Console.WriteLine($"  max error vs fp32 reference, relative to max |y|: " +
                          $"fp16 graph {fp16Error:e2}, int8 graph {int8Error:e2}");
        return int8Error < 2e-2f;
    }

    private static float[] Fp16Gemm(DmlDevice device, HostTensor x, HostTensor w, out float[] output)
    {
        var graph = new DmlGraph(device);
        DmlExpression xIn = graph.Input(TensorDataType.Float16, Sizes(x));
        DmlExpression wIn = graph.Input(TensorDataType.Float16, Sizes(w), owned: true);
        using DmlCompiledModel model = graph.Compile(new[]
        {
            DmlOps.Gemm(xIn, wIn, transB: MatrixTransform.Transpose),
        });
        model.Initialize(new Dictionary<DmlExpression, HostTensor> { [wIn] = w });
        Report("fp16 weights, folded", model);
        output = Time(model, new Dictionary<DmlExpression, HostTensor> { [xIn] = x });

        // The exact product, from the fp16-rounded operands both graphs see.
        float[] xf = x.ToFloats();
        float[] wf = w.ToFloats();
        int rows = x.Shape[2], inFeatures = x.Shape[3], outFeatures = w.Shape[2];
        var reference = new float[rows * outFeatures];
        Parallel.For(0, rows, row =>
        {
            for (int column = 0; column < outFeatures; column++)
            {
                double sum = 0;
                for (int k = 0; k < inFeatures; k++)
                {
                    sum += xf[row * inFeatures + k] * wf[column * inFeatures + k];
                }
                reference[row * outFeatures + column] = (float)sum;
            }
        });
        return reference;
    }

    private static float[] Int8Gemm(DmlDevice device, HostTensor x, HostTensor q, HostTensor s, bool owned)
    {
        var graph = new DmlGraph(device);
        DmlExpression xIn = graph.Input(TensorDataType.Float16, Sizes(x));
        DmlExpression qIn = graph.Input(TensorDataType.Int8, Sizes(q), owned);
        DmlExpression sIn = graph.Input(TensorDataType.Float16, Sizes(s), owned);
        using DmlCompiledModel model = graph.Compile(new[]
        {
            DmlOps.Gemm(xIn, DmlOps.Dequantize(qIn, sIn), transB: MatrixTransform.Transpose),
        });
        var inputs = new Dictionary<DmlExpression, HostTensor> { [xIn] = x };
        if (owned)
        {
            model.Initialize(new Dictionary<DmlExpression, HostTensor> { [qIn] = q, [sIn] = s });
        }
        else
        {
            inputs[qIn] = q;
            inputs[sIn] = s;
        }
        Report(owned ? "int8 weights + scales, folded" : "int8 weights + scales, bound per dispatch", model);
        return Time(model, inputs);
    }

    private static void Report(string what, DmlCompiledModel model)
    {
        Console.WriteLine($"  {what}: persistent {Mib(model.PersistentSize)}, " +
                          $"temporary {Mib(model.TemporarySize)}");
    }

    /// <summary>Dispatch a few times and print the mean, uploads and readback
    /// included: the same overhead for every graph here, so the differences
    /// are what to read.</summary>
    private static float[] Time(DmlCompiledModel model, Dictionary<DmlExpression, HostTensor> inputs)
    {
        HostTensor[] outputs = model.Dispatch(inputs);
        const int rounds = 10;
        var watch = Stopwatch.StartNew();
        for (int round = 0; round < rounds; round++)
        {
            model.Dispatch(inputs);
        }
        Console.WriteLine($"    {watch.Elapsed.TotalMilliseconds / rounds:0.0} ms per dispatch (with upload and readback)");
        return outputs[0].ToFloats();
    }

    /// <summary>Symmetric per-block int8 along the input axis: each run of
    /// <see cref="BlockSize"/> weights shares one fp16 scale.</summary>
    private static (sbyte[] Quantized, Half[] Scales) Quantize(float[] weight, int outFeatures, int inFeatures)
    {
        int blocks = inFeatures / BlockSize;
        var quantized = new sbyte[weight.Length];
        var scales = new Half[outFeatures * blocks];
        for (int row = 0; row < outFeatures; row++)
        {
            for (int block = 0; block < blocks; block++)
            {
                int start = row * inFeatures + block * BlockSize;
                float peak = 0;
                for (int i = 0; i < BlockSize; i++)
                {
                    peak = MathF.Max(peak, MathF.Abs(weight[start + i]));
                }
                Half scale = (Half)(peak / 127f);
                float inverse = peak == 0 ? 0 : 1f / (float)scale;
                scales[row * blocks + block] = scale;
                for (int i = 0; i < BlockSize; i++)
                {
                    quantized[start + i] = (sbyte)Math.Clamp(MathF.Round(weight[start + i] * inverse), -127, 127);
                }
            }
        }
        return (quantized, scales);
    }

    private static uint[] Sizes(HostTensor tensor) => tensor.Shape.Select(extent => (uint)extent).ToArray();

    private static string Mib(ulong bytes) => $"{bytes / (double)(1 << 20):0.00} MiB";
}
