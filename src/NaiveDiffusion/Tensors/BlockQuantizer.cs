using System.Buffers.Binary;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace NaiveDiffusion.Tensors;

/// <summary>Weight-only int8 quantization in blocks: a [rows, columns] matrix
/// becomes int8 values and one fp16 scale per run of <see cref="BlockSize"/>
/// columns, the block's largest magnitude mapped to 127. Symmetric, so there
/// is no zero point to carry. Half the bytes of fp16, and on SDXL's layers an
/// error a few times fp16's own rounding — DirectML dequantizes the block on
/// the way into the gemm, so the arithmetic is still half precision.</summary>
public static class BlockQuantizer
{
    public const int BlockSize = 32;

    /// <summary>The smallest weight the graphs quantize; the smaller ones
    /// stay at float16 (see <c>ModelBuilder.Weight</c>).</summary>
    public const long MinimumElements = 1 << 16;

    /// <summary>Whether a [rows, columns] weight can be quantized: the blocks
    /// have to tile the columns exactly.</summary>
    public static bool Fits(long elements, int rows) =>
        rows > 0 && elements % rows == 0 && (elements / rows) % BlockSize == 0;

    /// <summary>Quantize <paramref name="weight"/> viewed as [rows, columns].
    /// Returns the int8 values in that shape and the scales as [rows, columns / BlockSize].
    ///
    /// Each row is widened to float and quantized by the thread that took it,
    /// straight out of the stored bytes: widening the whole matrix first is
    /// a single-threaded pass writing twice the weight's size, and on the
    /// UNet's 2.6 billion parameters it costs more than the quantizing.</summary>
    public static (HostTensor Quantized, HostTensor Scales) Quantize(HostTensor weight, int rows)
    {
        if (!Fits(weight.ElementCount, rows))
        {
            throw new ArgumentException(
                $"{weight.ElementCount} elements in {rows} rows do not split into blocks of {BlockSize}");
        }
        int columns = checked((int)(weight.ElementCount / rows));
        int blocks = columns / BlockSize;

        // Written straight into the byte arrays the tensors will own, so the
        // output is not copied once more on the way out.
        var quantized = new byte[rows * columns];
        var scales = new byte[rows * blocks * sizeof(ushort)];
        ReadOnlyMemory<byte> data = weight.Data;
        HostDataType type = weight.DataType;
        int rowBytes = columns * HostTensor.BytesPerElement(type);

        Parallel.For(0, rows, () => new float[columns], (row, _, values) =>
        {
            HostTensor.Widen(type, data.Span.Slice(row * rowBytes, rowBytes), values);
            for (int block = 0; block < blocks; block++)
            {
                int start = block * BlockSize;
                ReadOnlySpan<float> source = values.AsSpan(start, BlockSize);
                float peak = MathF.Abs(TensorPrimitives.MaxMagnitude(source));

                // The scale is stored at half precision, so the values are
                // quantized against the scale as stored, not as computed;
                // a block too small to give a nonzero half scale is zero.
                Half scale = (Half)(peak / 127f);
                float stored = (float)scale;
                BinaryPrimitives.WriteHalfLittleEndian(
                    scales.AsSpan((row * blocks + block) * sizeof(ushort)), scale);
                if (stored == 0)
                {
                    continue;
                }
                QuantizeBlock(source, 1f / stored, quantized.AsSpan(row * columns + start, BlockSize));
            }
            return values;
        }, _ => { });

        return (
            new HostTensor(HostDataType.Int8, new[] { rows, columns }, quantized),
            new HostTensor(HostDataType.Float16, new[] { rows, blocks }, scales));
    }

    /// <summary>One block: scale, round to nearest even, clamp to ±127. The
    /// vector path is the same arithmetic eight lanes at a time, so the two
    /// agree bit for bit.</summary>
    private static void QuantizeBlock(ReadOnlySpan<float> values, float inverse, Span<byte> output)
    {
        if (Vector256.IsHardwareAccelerated && BlockSize == 4 * Vector256<float>.Count)
        {
            Vector256<float> scale = Vector256.Create(inverse);
            Vector256<short> first = Vector256.Narrow(
                Lane(values.Slice(0, 8), scale), Lane(values.Slice(8, 8), scale));
            Vector256<short> second = Vector256.Narrow(
                Lane(values.Slice(16, 8), scale), Lane(values.Slice(24, 8), scale));
            Vector256.Narrow(first, second).AsByte().CopyTo(output);
            return;
        }
        for (int i = 0; i < values.Length; i++)
        {
            output[i] = (byte)(sbyte)Math.Clamp(MathF.Round(values[i] * inverse), -127, 127);
        }
    }

    private static Vector256<int> Lane(ReadOnlySpan<float> values, Vector256<float> scale) =>
        Vector256.ConvertToInt32(Vector256.Max(Vector256.Create(-127f),
            Vector256.Min(Vector256.Create(127f), Vector256.Round(Vector256.Create(values) * scale))));
}
