namespace NaiveDiffusion.Dml;

/// <summary>A dispatch needed more video memory than the adapter could give.
/// The device itself is still usable.</summary>
public sealed class OutOfVideoMemoryException : InvalidOperationException
{
    public OutOfVideoMemoryException(string message, Exception inner)
        : base(message, inner)
    {
    }
}

/// <summary>The D3D12 device was removed — a driver reset or timeout. Nothing
/// on it works any more; the device has to be recreated.</summary>
public sealed class DeviceRemovedException : InvalidOperationException
{
    public DeviceRemovedException(string message)
        : base(message)
    {
    }
}
