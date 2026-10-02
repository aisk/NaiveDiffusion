using System.Runtime.InteropServices;
using NaiveDiffusion.Tensors;

namespace NaiveDiffusion.Tests;

/// <summary>The CPU matrix paths the text encoders run on, against the one
/// thing that matters most: a matrix has to mean the same numbers at every
/// width a checkpoint stores it at. A bfloat16 checkpoint read as float16
/// encodes the prompt into values in the millions.</summary>
public class CpuMathTests
{
    private const int Outer = 5, Inner = 8, Rows = 3;

    /// <summary>Values exactly representable at every width, so the three
    /// readings must agree bit for bit rather than within a tolerance.</summary>
    private static readonly float[] Weights = Enumerable.Range(0, Outer * Inner)
        .Select(i => (i % 7 - 3) * 0.25f).ToArray();

    private static HostTensor Matrix(HostDataType dataType)
    {
        byte[] data = dataType switch
        {
            HostDataType.Float32 => MemoryMarshal.AsBytes<float>(Weights).ToArray(),
            HostDataType.Float16 => MemoryMarshal.AsBytes<Half>(
                Weights.Select(w => (Half)w).ToArray()).ToArray(),
            HostDataType.BFloat16 => MemoryMarshal.AsBytes<ushort>(
                Weights.Select(w => (ushort)(BitConverter.SingleToUInt32Bits(w) >> 16)).ToArray())
                .ToArray(),
            _ => throw new ArgumentOutOfRangeException(nameof(dataType)),
        };
        return new HostTensor(dataType, new[] { Outer, Inner }, data);
    }

    private static readonly HostDataType[] Widths =
    {
        HostDataType.Float32, HostDataType.Float16, HostDataType.BFloat16,
    };

    [Test]
    public void LinearReadsEveryFloatWidthAlike()
    {
        float[] x = Enumerable.Range(0, Rows * Inner).Select(i => MathF.Sin(i)).ToArray();
        float[] bias = { 1, 2, 3, 4, 5 };
        var expected = new float[Rows * Outer];
        for (int m = 0; m < Rows; m++)
        {
            for (int o = 0; o < Outer; o++)
            {
                float sum = bias[o];
                for (int i = 0; i < Inner; i++)
                {
                    sum += x[m * Inner + i] * Weights[o * Inner + i];
                }
                expected[m * Outer + o] = sum;
            }
        }

        foreach (HostDataType width in Widths)
        {
            var y = new float[Rows * Outer];
            CpuMath.Linear(x, Rows, Inner, Matrix(width), bias, y, Outer);
            Assert.That(y, Is.EqualTo(expected).Within(1e-5f), width.ToString());
        }
    }

    [Test]
    public void RowsReadEveryFloatWidthAlike()
    {
        foreach (HostDataType width in Widths)
        {
            HostTensor matrix = Matrix(width);
            var copied = new float[Inner];
            CpuMath.CopyRow(matrix, row: 3, Inner, copied, offset: 0);
            Assert.That(copied, Is.EqualTo(Weights.Skip(3 * Inner).Take(Inner)), width.ToString());

            float[] added = Enumerable.Repeat(10f, Inner).ToArray();
            CpuMath.AddRow(matrix, row: 1, Inner, added, offset: 0);
            Assert.That(added, Is.EqualTo(Weights.Skip(Inner).Take(Inner).Select(w => w + 10f)),
                width.ToString());
        }
    }
}
