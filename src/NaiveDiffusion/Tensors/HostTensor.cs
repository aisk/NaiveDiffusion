using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace NaiveDiffusion.Tensors;

/// <summary>Element types a host tensor can carry. BFloat16 appears only as a
/// checkpoint source type; it is widened before reaching DirectML.</summary>
public enum HostDataType
{
    Float32,
    Float16,
    BFloat16,
    UInt32,
    Int32,
    UInt16,
    Int64,
    Int8,
}

/// <summary>A CPU-side tensor: a shape, an element type, and the packed bytes.
/// This is the currency between checkpoint loading, graph binding and readback.
///
/// The bytes are a <see cref="ReadOnlyMemory{T}"/> rather than an array because
/// a checkpoint's tensors are windows onto a memory-mapped file, not copies of
/// it: nothing is read into the heap that is only going to be handed straight to
/// the GPU or walked once. A tensor is only valid while whatever produced it is
/// alive — for mapped ones, the <see cref="Weights.SafetensorsFile"/>.</summary>
public sealed class HostTensor
{
    public HostDataType DataType { get; }
    public int[] Shape { get; }
    public ReadOnlyMemory<byte> Data { get; }

    public HostTensor(HostDataType dataType, int[] shape, ReadOnlyMemory<byte> data)
    {
        if (Count(shape) * BytesPerElement(dataType) != data.Length)
        {
            throw new ArgumentException(
                $"{data.Length} bytes do not fill a {dataType} tensor of [{string.Join(", ", shape)}]");
        }
        DataType = dataType;
        Shape = shape;
        Data = data;
    }

    public long ElementCount => Count(Shape);

    private static long Count(int[] shape)
    {
        long elements = 1;
        foreach (int extent in shape)
        {
            elements *= extent;
        }
        return elements;
    }

    public static int BytesPerElement(HostDataType dataType) => dataType switch
    {
        HostDataType.Float32 => 4,
        HostDataType.Float16 => 2,
        HostDataType.BFloat16 => 2,
        HostDataType.UInt32 => 4,
        HostDataType.Int32 => 4,
        HostDataType.UInt16 => 2,
        HostDataType.Int64 => 8,
        HostDataType.Int8 => 1,
        _ => throw new ArgumentOutOfRangeException(nameof(dataType)),
    };

    public static HostTensor FromFloats(float[] values, params int[] shape) =>
        new(HostDataType.Float32, shape, MemoryMarshal.AsBytes<float>(values).ToArray());

    public static HostTensor FromUInt32(uint[] values, params int[] shape) =>
        new(HostDataType.UInt32, shape, MemoryMarshal.AsBytes<uint>(values).ToArray());

    public static HostTensor FromInt32(int[] values, params int[] shape) =>
        new(HostDataType.Int32, shape, MemoryMarshal.AsBytes<int>(values).ToArray());

    public static HostTensor FromHalves(Half[] values, params int[] shape) =>
        new(HostDataType.Float16, shape, MemoryMarshal.AsBytes<Half>(values).ToArray());

    public static HostTensor FromInt8(sbyte[] values, params int[] shape) =>
        new(HostDataType.Int8, shape, MemoryMarshal.AsBytes<sbyte>(values).ToArray());

    /// <summary>A run of whole rows along axis 0, as a view — for a packed
    /// tensor that is just a byte range, so nothing is copied.</summary>
    public HostTensor SliceRows(int start, int count)
    {
        long rowBytes = Data.Length / Shape[0];
        int[] shape = Shape.ToArray();
        shape[0] = count;
        return new HostTensor(DataType, shape,
            Data.Slice(checked((int)(rowBytes * start)), checked((int)(rowBytes * count))));
    }

    public HostTensor Reshape(params int[] shape)
    {
        if (Count(shape) != ElementCount)
        {
            throw new ArgumentException(
                $"cannot view {ElementCount} elements as [{string.Join(", ", shape)}]");
        }
        return new HostTensor(DataType, shape, Data);
    }

    /// <summary>This tensor's data as <paramref name="target"/>, sharing the buffer
    /// when no conversion is needed. Only the float widths convert; anything else
    /// must already match.</summary>
    public HostTensor ConvertTo(HostDataType target)
    {
        if (target == DataType)
        {
            return this;
        }

        float[] wide = ToFloats();
        if (target == HostDataType.Float32)
        {
            return new HostTensor(HostDataType.Float32, Shape,
                MemoryMarshal.AsBytes<float>(wide).ToArray());
        }
        if (target == HostDataType.Float16)
        {
            var narrow = new byte[wide.Length * 2];
            TensorPrimitives.ConvertToHalf(wide, MemoryMarshal.Cast<byte, Half>(narrow.AsSpan()));
            return new HostTensor(HostDataType.Float16, Shape, narrow);
        }
        throw new NotSupportedException($"cannot convert {DataType} to {target}");
    }

    /// <summary>The data widened to float32, whatever float width it is stored at.</summary>
    public float[] ToFloats()
    {
        var wide = new float[ElementCount];
        Widen(DataType, Data.Span, wide);
        return wide;
    }

    /// <summary>The values of an int32 tensor — a mask, a key count.</summary>
    public int[] ToInt32s()
    {
        if (DataType != HostDataType.Int32)
        {
            throw new InvalidOperationException($"{DataType} is not int32");
        }
        return MemoryMarshal.Cast<byte, int>(Data.Span).ToArray();
    }

    /// <summary>One row of a packed matrix — <paramref name="width"/> values
    /// along the last axis — as float32, at whichever float width it is
    /// stored: the row is read straight off the mapping, so a tower's
    /// matrices can be widened one row at a time inside the loop that uses
    /// them, without a float32 copy of the whole tower ever existing.</summary>
    public void WidenRow(long row, int width, Span<float> target)
    {
        int itemSize = BytesPerElement(DataType);
        Widen(DataType, Data.Span.Slice(checked((int)(row * width * itemSize)), width * itemSize),
            target);
    }

    /// <summary>Packed <paramref name="type"/> values as float32. Each width
    /// is its own case: a bfloat16 value read as float16 is not a little off,
    /// it is garbage in the millions — the exponent field is in a different
    /// place — and a checkpoint read that way comes out as an all-NaN UNet,
    /// not as a wrong image. Float32 represents every float16 and bfloat16
    /// value exactly, so nothing is lost either way.</summary>
    public static void Widen(HostDataType type, ReadOnlySpan<byte> bytes, Span<float> target)
    {
        switch (type)
        {
            case HostDataType.Float32:
                MemoryMarshal.Cast<byte, float>(bytes).CopyTo(target);
                return;
            case HostDataType.Float16:
                TensorPrimitives.ConvertToSingle(MemoryMarshal.Cast<byte, Half>(bytes), target);
                return;
            case HostDataType.BFloat16:
            {
                // bfloat16 is the top half of a float32.
                ReadOnlySpan<ushort> bits = MemoryMarshal.Cast<byte, ushort>(bytes);
                Span<uint> raw = MemoryMarshal.Cast<float, uint>(target);
                for (int i = 0; i < bits.Length; i++)
                {
                    raw[i] = (uint)bits[i] << 16;
                }
                return;
            }
            default:
                throw new NotSupportedException($"cannot widen {type} to float32");
        }
    }
}
