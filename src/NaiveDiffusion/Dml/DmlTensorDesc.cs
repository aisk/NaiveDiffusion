using NaiveDiffusion.Tensors;
using Vortice.DirectML;

namespace NaiveDiffusion.Dml;

/// <summary>A DML buffer tensor description in builder-friendly form: element
/// type, sizes, optional strides, flags, and the byte size of the buffer that
/// backs it. Immutable; reinterpret produces a new view over the same buffer.</summary>
public sealed class DmlTensorDesc
{
    public TensorDataType DataType { get; }
    public TensorFlags Flags { get; }
    public uint[] Sizes { get; }
    public uint[]? Strides { get; }
    public ulong TotalBytes { get; }

    private DmlTensorDesc(TensorDataType dataType, TensorFlags flags, uint[] sizes,
        uint[]? strides, ulong totalBytes)
    {
        DataType = dataType;
        Flags = flags;
        Sizes = sizes;
        Strides = strides;
        TotalBytes = totalBytes;
    }

    /// <summary>A packed tensor: no strides, buffer exactly the implied size.</summary>
    public static DmlTensorDesc Packed(TensorDataType dataType, uint[] sizes,
        TensorFlags flags = TensorFlags.None)
    {
        return new DmlTensorDesc(dataType, flags, sizes, null,
            BufferTensorDescription.CalculateMinimumImpliedSize(dataType, sizes));
    }

    /// <summary>The DirectMLX reinterpret rule: new sizes, strides and possibly
    /// type over the same buffer, keeping the flags and the buffer's byte size.</summary>
    public DmlTensorDesc Reinterpret(uint[] sizes, uint[]? strides, TensorDataType? dataType = null)
    {
        return new DmlTensorDesc(dataType ?? DataType, Flags, sizes, strides, TotalBytes);
    }

    public int ElementCount
    {
        get
        {
            long count = 1;
            foreach (uint extent in Sizes)
            {
                count *= extent;
            }
            return checked((int)count);
        }
    }

    public BufferTensorDescription ToBuffer() => new()
    {
        DataType = DataType,
        Flags = Flags,
        Sizes = Sizes,
        Strides = Strides,
        TotalTensorSizeInBytes = TotalBytes,
        GuaranteedBaseOffsetAlignment = 0,
    };

    public TensorDescription ToTensor() => ToBuffer();

    public override string ToString() =>
        $"{DataType} [{string.Join(", ", Sizes)}]" +
        (Strides is null ? "" : $" strides [{string.Join(", ", Strides)}]");

    public static TensorDataType ToDataType(HostDataType dataType) => dataType switch
    {
        HostDataType.Float32 => TensorDataType.Float32,
        HostDataType.Float16 => TensorDataType.Float16,
        HostDataType.UInt32 => TensorDataType.Uint32,
        HostDataType.Int32 => TensorDataType.Int32,
        HostDataType.UInt16 => TensorDataType.Uint16,
        HostDataType.Int64 => TensorDataType.Int64,
        HostDataType.Int8 => TensorDataType.Int8,
        _ => throw new NotSupportedException($"{dataType} has no DirectML equivalent"),
    };

    public static HostDataType ToHostDataType(TensorDataType dataType) => dataType switch
    {
        TensorDataType.Float32 => HostDataType.Float32,
        TensorDataType.Float16 => HostDataType.Float16,
        TensorDataType.Uint32 => HostDataType.UInt32,
        TensorDataType.Int32 => HostDataType.Int32,
        TensorDataType.Uint16 => HostDataType.UInt16,
        TensorDataType.Int64 => HostDataType.Int64,
        TensorDataType.Int8 => HostDataType.Int8,
        _ => throw new NotSupportedException($"{dataType} has no host equivalent"),
    };
}
