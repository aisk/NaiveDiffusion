using System.Numerics;
using Vortice.Direct3D12;

namespace NaiveDiffusion.Dml;

/// <summary>The buffers every initialize and dispatch runs through, kept between
/// calls and resized to what each one asks for.
///
/// The inputs buffer lives in system memory on purpose. DirectML reads it once
/// per initialize — for a model-sized graph that is gigabytes of weights on
/// their way into the persistent resource — and putting it in video memory
/// would keep the weights there twice at the peak, once staged and once
/// folded. A custom heap in L0 with unordered access lets the CPU write it
/// directly and the GPU read it across PCIe: one copy instead of two, no upload
/// heap and no copy between them, and the read costs a fraction of a second
/// against gigabytes of video memory saved. The readback heap is CPU-visible
/// because the CPU reads it; outputs and scratch stay on the GPU, where
/// every dispatch touches them.
///
/// Tensors that stay on the GPU between graphs rent their buffers here. A
/// chain of graphs rents and returns the same few sizes over and over, so a
/// returned buffer is kept on a free list keyed by its rounded size and handed
/// out again rather than recreated; <see cref="TrimFree"/> lets the free lists
/// go once the chain is done with them.</summary>
internal sealed class DmlBufferPool : IDisposable
{
    private readonly ID3D12Device _device;

    private readonly Dictionary<ulong, Stack<ID3D12Resource>> _freeLocal = new();
    private readonly Dictionary<ulong, Stack<ID3D12Resource>> _freeShared = new();
    private ulong _rentedLocal;
    private ulong _rentedShared;
    private ulong _freeLocalBytes;
    private ulong _freeSharedBytes;
    private ulong _temporaryFloor;

    private ID3D12Resource? _readbackHeap;
    private ID3D12Resource? _inputs;
    private ID3D12Resource? _outputs;
    private ID3D12Resource? _temporary;
    private ID3D12Resource? _window;
    private ID3D12DescriptorHeap? _descriptorHeap;
    private uint _descriptorCount;

    public DmlBufferPool(ID3D12Device device) => _device = device;

    public ID3D12Resource ReadbackHeap => _readbackHeap!;
    public ID3D12Resource Inputs => _inputs!;
    public ID3D12Resource Outputs => _outputs!;
    public ID3D12Resource Temporary => _temporary!;
    public ID3D12Resource Window => _window!;
    public ID3D12DescriptorHeap DescriptorHeap => _descriptorHeap!;

    public void EnsureReadbackHeap(ulong requestedSize)
    {
        ulong newSize = GrowBufferSize(requestedSize);
        if (_readbackHeap is null || _readbackHeap.Description.Width != newSize)
        {
            Drop(ref _readbackHeap);
            _readbackHeap = _device.CreateCommittedResource(
                HeapProperties.ReadbackHeapProperties, HeapFlags.None,
                ResourceDescription.Buffer(newSize), ResourceStates.CopyDest);
        }
    }

    /// <summary>The staging buffer, in system memory and mapped by the CPU.
    /// A committed resource on a custom heap can be both CPU-writable and an
    /// unordered access view; the upload heap type cannot, which is why this
    /// does not use it.</summary>
    public void EnsureInputs(ulong requestedSize)
    {
        ulong newSize = GrowBufferSize(requestedSize);
        if (_inputs is null || _inputs.Description.Width != newSize)
        {
            Drop(ref _inputs);
            _inputs = _device.CreateCommittedResource(
                new HeapProperties(CpuPageProperty.WriteCombine, MemoryPool.L0), HeapFlags.None,
                ResourceDescription.Buffer(newSize, ResourceFlags.AllowUnorderedAccess),
                ResourceStates.UnorderedAccess);
        }
    }

    public void EnsureOutputs(ulong requestedSize) => EnsureBuffer(ref _outputs, requestedSize);

    /// <summary>The scratch buffer is resized to what each dispatch asks for,
    /// which suits a run of one graph. A chain of graphs asks for a different
    /// size at every link, and reallocating a buffer of hundreds of megabytes
    /// per link is not free; the chain sets a floor at the largest it will ask
    /// for, so the buffer is created once, and clears it when it is done.</summary>
    public void EnsureTemporary(ulong requestedSize) =>
        EnsureBuffer(ref _temporary, Math.Max(requestedSize, _temporaryFloor));

    public ulong TemporaryFloor
    {
        get => _temporaryFloor;
        set => _temporaryFloor = value;
    }

    /// <summary>The buffer a streamed model's weights are copied into for the
    /// length of one dispatch. One is enough: dispatches are serial, and it is
    /// sized for the largest streamed graph and kept until
    /// <see cref="TrimFree"/>, since a chain runs its links through it in turn.</summary>
    public void EnsureWindow(ulong requestedSize)
    {
        ulong newSize = GrowBufferSize(requestedSize);
        if (_window is null || _window.Description.Width < newSize)
        {
            Drop(ref _window);
            _window = _device.CreateCommittedResource(
                HeapProperties.DefaultHeapProperties, HeapFlags.None,
                ResourceDescription.Buffer(newSize, ResourceFlags.AllowUnorderedAccess),
                ResourceStates.UnorderedAccess);
        }
    }

    private void EnsureBuffer(ref ID3D12Resource? buffer, ulong requestedSize)
    {
        ulong newSize = GrowBufferSize(requestedSize);
        if (buffer is null || buffer.Description.Width != newSize)
        {
            Drop(ref buffer);
            buffer = _device.CreateCommittedResource(
                HeapProperties.DefaultHeapProperties, HeapFlags.None,
                ResourceDescription.Buffer(newSize, ResourceFlags.AllowUnorderedAccess),
                ResourceStates.UnorderedAccess);
        }
    }

    /// <summary>Release a buffer that is about to be replaced, and forget it
    /// before the replacement is asked for: creating one can fail for want
    /// of memory, which a caller may recover from, and a field still naming
    /// a released resource would be read on the next attempt. Old first,
    /// then new, so the two never coexist — for a scratch buffer of
    /// gigabytes that would be its own way to run out.</summary>
    private static void Drop(ref ID3D12Resource? buffer)
    {
        buffer?.Dispose();
        buffer = null;
    }

    public void EnsureDescriptorHeap(uint requestedCount)
    {
        // Grows and never shrinks: a descriptor heap is kilobytes, and
        // recreating one every time a smaller graph follows a larger is waste.
        uint newCount = BitOperations.RoundUpToPowerOf2(Math.Max(requestedCount, 1));
        if (_descriptorHeap is null || _descriptorCount < newCount)
        {
            _descriptorHeap?.Dispose();
            _descriptorHeap = null;
            _descriptorHeap = _device.CreateDescriptorHeap(new DescriptorHeapDescription
            {
                Type = DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView,
                DescriptorCount = newCount,
                Flags = DescriptorHeapFlags.ShaderVisible,
            });
            _descriptorCount = newCount;
        }
    }

    /// <summary>A buffer for a tensor that stays on the GPU between graphs.
    /// <paramref name="shared"/> asks for system memory the CPU can write —
    /// for an input uploaded once — instead of video memory.</summary>
    public ID3D12Resource RentBuffer(ulong requestedSize, bool shared)
    {
        ulong size = GrowBufferSize(requestedSize);
        Dictionary<ulong, Stack<ID3D12Resource>> free = shared ? _freeShared : _freeLocal;
        ID3D12Resource buffer;
        if (free.TryGetValue(size, out Stack<ID3D12Resource>? stack) && stack.Count > 0)
        {
            buffer = stack.Pop();
            if (shared)
            {
                _freeSharedBytes -= size;
            }
            else
            {
                _freeLocalBytes -= size;
            }
        }
        else
        {
            buffer = _device.CreateCommittedResource(
                shared
                    ? new HeapProperties(CpuPageProperty.WriteCombine, MemoryPool.L0)
                    : HeapProperties.DefaultHeapProperties,
                HeapFlags.None,
                ResourceDescription.Buffer(size, ResourceFlags.AllowUnorderedAccess),
                ResourceStates.UnorderedAccess);
        }
        if (shared)
        {
            _rentedShared += size;
        }
        else
        {
            _rentedLocal += size;
        }
        return buffer;
    }

    /// <summary>A rented buffer coming back, kept for the next rent of its size.</summary>
    public void ReturnBuffer(ID3D12Resource buffer, bool shared)
    {
        ulong size = buffer.Description.Width;
        Dictionary<ulong, Stack<ID3D12Resource>> free = shared ? _freeShared : _freeLocal;
        if (!free.TryGetValue(size, out Stack<ID3D12Resource>? stack))
        {
            stack = new Stack<ID3D12Resource>();
            free[size] = stack;
        }
        stack.Push(buffer);
        if (shared)
        {
            _rentedShared -= size;
            _freeSharedBytes += size;
        }
        else
        {
            _rentedLocal -= size;
            _freeLocalBytes += size;
        }
    }

    /// <summary>Release every buffer on the free lists. Rented ones are their
    /// holders' to return first.</summary>
    public void TrimFree()
    {
        foreach (Stack<ID3D12Resource> stack in _freeLocal.Values.Concat(_freeShared.Values))
        {
            while (stack.Count > 0)
            {
                stack.Pop().Dispose();
            }
        }
        _freeLocal.Clear();
        _freeShared.Clear();
        _freeLocalBytes = 0;
        _freeSharedBytes = 0;
        _window?.Dispose();
        _window = null;
    }

    /// <summary>Drop every pooled buffer. They are all recreated on demand,
    /// so this costs nothing but the next allocation; what it buys is a true
    /// reading of what the process holds, at a moment when the decoder's
    /// scratch from the last run is still sitting here waiting to be resized.</summary>
    public void ReleaseAll()
    {
        TrimFree();
        ReleaseStaging();
        _readbackHeap?.Dispose();
        _readbackHeap = null;
        _outputs?.Dispose();
        _outputs = null;
        _temporary?.Dispose();
        _temporary = null;
    }

    /// <summary>Drop the staging buffer. Initializing a model-sized graph grows
    /// it to the size of that graph's weights — gigabytes that nothing needs
    /// once the persistent resource holds them — and dropping it keeps two
    /// initializations from stacking their staging on top of each other. The
    /// next dispatch recreates it at the size it actually asks for.</summary>
    public void ReleaseStaging()
    {
        _inputs?.Dispose();
        _inputs = null;
    }

    // Each buffer's size, reported separately: a total says memory grew, and
    // only the breakdown says which allocation grew, which is the whole point of
    // counting them ourselves when DXGI already reports the total.
    public ulong InputsBytes => Width(_inputs);
    public ulong TemporaryBytes => Width(_temporary);

    /// <summary>Video memory holding a streamed graph's weights for its dispatch.</summary>
    public ulong WindowBytes => Width(_window);

    /// <summary>The outputs buffer plus every video-memory buffer lent to a
    /// tensor that stays on the GPU, and the ones waiting on the free list to
    /// be lent again: all of it is activations, whichever way it got there.</summary>
    public ulong OutputsBytes => Width(_outputs) + _rentedLocal + _freeLocalBytes;

    /// <summary>Bytes of the pool's buffers that live in system memory. The
    /// staging pair is CPU-visible and therefore costs no video memory — worth
    /// keeping apart, or the total would not be comparable against what DXGI
    /// reports for the local segment. The descriptor heap is left out of both:
    /// it is a few kilobytes against gigabytes, and it is not a buffer whose
    /// width means anything.</summary>
    public ulong TrackedShared =>
        InputsBytes + Width(_readbackHeap) + _rentedShared + _freeSharedBytes;

    private static ulong Width(ID3D12Resource? buffer) => buffer?.Description.Width ?? 0;

    /// <summary>Add every live buffer to a residency set.</summary>
    public void CollectInto(List<ID3D12Pageable> residencySet)
    {
        foreach (ID3D12Pageable? pageable in new ID3D12Pageable?[]
        {
            _readbackHeap, _inputs, _outputs, _temporary, _descriptorHeap,
        })
        {
            if (pageable is not null)
            {
                residencySet.Add(pageable);
            }
        }
    }

    /// <summary>How large a buffer to actually allocate for a request. Doubling
    /// keeps repeated small growth from reallocating every time; past 256 MiB it
    /// grows by fixed steps instead, because rounding a 2.1 GiB request up to a
    /// 4 GiB single resource can remove the device outright.</summary>
    private static ulong GrowBufferSize(ulong requestedSize)
    {
        const ulong minimumSize = 65536;
        const ulong stepSize = 256UL << 20;
        if (requestedSize <= stepSize)
        {
            return Math.Max(BitOperations.RoundUpToPowerOf2(requestedSize), minimumSize);
        }
        return DmlDevice.RoundUp(requestedSize, stepSize);
    }

    public void Dispose()
    {
        TrimFree();
        _readbackHeap?.Dispose();
        _inputs?.Dispose();
        _outputs?.Dispose();
        _temporary?.Dispose();
        _window?.Dispose();
        _descriptorHeap?.Dispose();
    }
}
