using NaiveDiffusion.Tensors;
using Vortice.Direct3D12;

namespace NaiveDiffusion.Dml;

/// <summary>A tensor that lives in a GPU buffer: the output of one graph on
/// its way into the next without a trip through host memory, or an input
/// uploaded once and read by several graphs. The buffer is rented from the
/// device's pool and goes back there on dispose, so a tensor that is done with
/// should be disposed promptly — the pool only reuses what has been returned.</summary>
public sealed class DeviceTensor : IDisposable
{
    private readonly DmlDevice _device;
    private ID3D12Resource? _buffer;

    /// <summary>Whether the buffer is in system memory (CPU-written, read by
    /// the GPU across the bus) rather than video memory.</summary>
    internal bool Shared { get; }

    internal DmlTensorDesc Desc { get; }

    internal DeviceTensor(DmlDevice device, ID3D12Resource buffer, DmlTensorDesc desc, bool shared)
    {
        _device = device;
        _buffer = buffer;
        Desc = desc;
        Shared = shared;
    }

    internal ID3D12Resource Buffer =>
        _buffer ?? throw new ObjectDisposedException(nameof(DeviceTensor));

    public uint[] Shape => Desc.Sizes;

    public HostDataType DataType => DmlTensorDesc.ToHostDataType(Desc.DataType);

    /// <summary>Bytes the tensor occupies, which is what a graph input binding
    /// has to match exactly.</summary>
    public ulong TotalBytes => Desc.TotalBytes;

    /// <summary>Copy the tensor back to host memory.</summary>
    public HostTensor Download() => _device.Download(this);

    public void Dispose()
    {
        if (_buffer is not null)
        {
            _device.ReturnBuffer(_buffer, Shared);
            _buffer = null;
        }
    }

    public override string ToString() => $"<device {Desc}>";
}
