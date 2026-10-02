using NaiveDiffusion.Tensors;

namespace NaiveDiffusion.Tests;

/// <summary>Block-wise int8: the layout the dequantize op expects and the
/// error bound the scheme promises.</summary>
public class BlockQuantizerTests
{
    [Test]
    public void FitsNeedsWholeBlocksPerRow()
    {
        Assert.That(BlockQuantizer.Fits(4 * 64, 4));
        Assert.That(BlockQuantizer.Fits(4 * 48, 4), Is.False);
        Assert.That(BlockQuantizer.Fits(100, 3), Is.False);
        Assert.That(BlockQuantizer.Fits(64, 0), Is.False);
    }

    [Test]
    public void DequantizingComesBackWithinHalfAStep()
    {
        const int rows = 4, columns = 96;
        var random = new Random(1);
        float[] values = Enumerable.Range(0, rows * columns)
            .Select(_ => (float)(random.NextDouble() * 2 - 1) * (random.Next(4) == 0 ? 8f : 1f))
            .ToArray();
        (HostTensor quantized, HostTensor scales) =
            BlockQuantizer.Quantize(HostTensor.FromFloats(values, rows, columns), rows);

        Assert.That(quantized.DataType, Is.EqualTo(HostDataType.Int8));
        Assert.That(quantized.Shape, Is.EqualTo(new[] { rows, columns }));
        Assert.That(scales.DataType, Is.EqualTo(HostDataType.Float16));
        Assert.That(scales.Shape, Is.EqualTo(new[] { rows, columns / BlockQuantizer.BlockSize }));

        float[] scale = scales.ToFloats();
        ReadOnlySpan<byte> bytes = quantized.Data.Span;
        for (int row = 0; row < rows; row++)
        {
            for (int block = 0; block < columns / BlockQuantizer.BlockSize; block++)
            {
                int start = row * columns + block * BlockQuantizer.BlockSize;
                float peak = values.Skip(start).Take(BlockQuantizer.BlockSize).Max(MathF.Abs);
                float s = scale[row * (columns / BlockQuantizer.BlockSize) + block];
                // The scale is the peak over 127, rounded to half precision.
                Assert.That(s, Is.EqualTo(peak / 127f).Within(peak / 127f * 1e-3f));
                for (int i = 0; i < BlockQuantizer.BlockSize; i++)
                {
                    sbyte q = (sbyte)bytes[start + i];
                    Assert.That(q, Is.InRange(-127, 127));
                    Assert.That(q * s, Is.EqualTo(values[start + i]).Within(s * 0.5f + 1e-6f));
                }
            }
        }
    }

    [Test]
    public void AZeroBlockHasAZeroScale()
    {
        float[] values = new float[2 * 32];
        values[40] = 1e-9f;
        (HostTensor quantized, HostTensor scales) =
            BlockQuantizer.Quantize(HostTensor.FromFloats(values, 2, 32), 2);
        Assert.That(scales.ToFloats(), Is.All.Zero);
        Assert.That(quantized.Data.ToArray(), Is.All.Zero);
    }

    [Test]
    public void RefusesAShapeThatDoesNotTile()
    {
        Assert.That(() => BlockQuantizer.Quantize(HostTensor.FromFloats(new float[48], 1, 48), 1),
            Throws.ArgumentException);
    }
}
