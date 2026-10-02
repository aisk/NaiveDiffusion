using NaiveDiffusion.Tensors;
using SharpGen.Runtime;
using Vortice.Direct3D12;
using Vortice.DirectML;
using Vortice.DXGI;

namespace NaiveDiffusion.Dml;

/// <summary>The execution machinery: one D3D12 device with a compute queue, one
/// DirectML device, and the pooled buffers every initialize and dispatch runs
/// through. Everything is synchronous — each call records, executes and waits.
/// Tensors can stay on the GPU between calls (<see cref="DeviceTensor"/>), so a
/// chain of graphs pays the wait between links but not a trip through host
/// memory.</summary>
public sealed unsafe class DmlDevice : IDisposable
{
    // DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT.
    private const ulong TensorAlignment = 16;

    private readonly ID3D12Device _d3dDevice;
    private readonly ID3D12CommandQueue _queue;
    private readonly ID3D12CommandAllocator _allocator;
    private readonly ID3D12GraphicsCommandList _list;
    private readonly ID3D12Fence _fence;
    private readonly AutoResetEvent _fenceEvent = new(false);
    private ulong _fenceValue;

    private readonly IDMLCommandRecorder _recorder;
    private readonly IDMLOperatorInitializer _initializer;
    private IDMLBindingTable? _bindingTable;

    private readonly DmlBufferPool _pool;

    // Residency management on Direct3D 12's own API, which PyDirectML leaves
    // to gpgmm: everything a submission touches is made resident, in one
    // call, before ExecuteCommandLists. Under memory pressure the OS demotes
    // idle allocations — ours included — and a DirectML dispatch that
    // demand-pages mid-flight is how DXGI_ERROR_DEVICE_HUNG happens; making
    // the working set resident up front moves the paging before the dispatch
    // instead of inside it, and turns "over budget" into an error instead of
    // a hang.
    private readonly List<ID3D12Pageable> _residencySet = new();
    private readonly IDXGIAdapter3? _adapter3;
    private readonly ID3D12Device1? _device1;

    // Video memory this device has handed to models and never got back, tracked
    // alongside the pool's own buffers so a reading can say which allocation
    // grew rather than only that something did.
    private ulong _persistentBytes;
    // The same for streamed models, whose weights are in system memory and
    // belong in the shared column.
    private ulong _streamedBytes;

    public IDMLDevice1 Device { get; }

    public string AdapterName { get; }

    /// <summary>Every video memory reading taken so far. Sampling costs one
    /// cheap DXGI call per submission, so it is always on; a caller that does
    /// not want the history to grow across a long session clears it.</summary>
    public MemoryLog Memory { get; } = new();

    /// <summary>What the caller is in the middle of, attached to every reading
    /// taken from now on. The device cannot tell a UNet dispatch from a VAE one,
    /// so whoever is driving it labels the phases.</summary>
    public string CurrentStage { get; set; } = "";

    /// <summary>Take a reading now. Submissions sample themselves; this is for
    /// the moments in between — after a model is disposed, say, where the point
    /// is to confirm the memory actually came back.</summary>
    public void Sample(string what)
    {
        if (_adapter3 is null)
        {
            return;
        }
        (QueryVideoMemoryInfo local, QueryVideoMemoryInfo nonLocal) = QueryMemory();
        // The inputs buffer lives in system memory (see DmlBufferPool), so it
        // is reported on its own and counted under shared, not local.
        Memory.Add(CurrentStage, what, local.CurrentUsage, local.Budget, nonLocal.CurrentUsage,
            _persistentBytes + _pool.WindowBytes, _pool.TemporaryBytes, _pool.InputsBytes,
            _pool.OutputsBytes, _pool.TrackedShared + _streamedBytes);
    }

    /// <summary>Video memory as it stands right now, without recording it:
    /// what this process holds, and what the OS is currently willing to let it
    /// hold. The budget is the number worth deciding against — it already falls
    /// when other processes take memory, and crossing it is where the working
    /// set starts spilling to system memory. Both are zero on an adapter that
    /// does not report them, which callers should read as "do not gamble".</summary>
    public (ulong Usage, ulong Budget) VideoMemory()
    {
        if (_adapter3 is null)
        {
            return (0, 0);
        }
        QueryVideoMemoryInfo local = QueryMemory().Local;
        return (local.CurrentUsage, local.Budget);
    }

    /// <summary>DXGI's two segment groups: local is the card's own memory,
    /// non-local the system memory the working set spills into. DXGI refuses
    /// to report on a removed device, and the removal can land here first —
    /// it is noticed asynchronously, after the submission that caused it has
    /// already been waited for — so the refusal is translated into the reason.</summary>
    private (QueryVideoMemoryInfo Local, QueryVideoMemoryInfo NonLocal) QueryMemory()
    {
        try
        {
            return (_adapter3!.QueryVideoMemoryInfo(0, MemorySegmentGroup.Local),
                _adapter3.QueryVideoMemoryInfo(0, MemorySegmentGroup.NonLocal));
        }
        catch (SharpGenException exception)
        {
            Result removed = _d3dDevice.DeviceRemovedReason;
            if (removed.Failure)
            {
                throw new DeviceRemovedException(
                    $"the device was removed: {removed} ({exception.Message})");
            }
            throw;
        }
    }

    /// <summary>A model is giving its weights back.</summary>
    internal void ForgetPersistent(ulong bytes, bool streamed)
    {
        if (streamed)
        {
            _streamedBytes -= bytes;
        }
        else
        {
            _persistentBytes -= bytes;
        }
    }

    /// <summary>Occupy <paramref name="bytes"/> of video memory until the
    /// result is disposed, at the highest residency priority, so that under
    /// pressure the OS pages everything else out first. It stands in for a
    /// smaller card — the only way to test what a run does when the weights
    /// do not fit without owning one.</summary>
    public IDisposable HoldVideoMemory(ulong bytes)
    {
        const ulong chunk = 1UL << 30;
        var buffers = new List<ID3D12Resource>();
        try
        {
            for (ulong held = 0; held < bytes; held += chunk)
            {
                ID3D12Resource buffer = _d3dDevice.CreateCommittedResource(
                    HeapProperties.DefaultHeapProperties, HeapFlags.None,
                    ResourceDescription.Buffer(Math.Min(chunk, bytes - held)),
                    ResourceStates.Common);
                buffers.Add(buffer);
            }
            if (_device1 is not null && buffers.Count > 0)
            {
                _device1.SetResidencyPriority((uint)buffers.Count,
                    buffers.Cast<ID3D12Pageable>().ToArray(),
                    Enumerable.Repeat(ResidencyPriority.Maximum, buffers.Count).ToArray());
            }
            _d3dDevice.MakeResident(buffers.Cast<ID3D12Pageable>().ToArray());
        }
        catch
        {
            // Asking for more than the card has is the likely failure, and
            // the chunks that did fit are not to stay held by a hog that
            // never came to be.
            foreach (ID3D12Resource buffer in buffers)
            {
                buffer.Dispose();
            }
            throw;
        }
        return new Held(buffers);
    }

    private sealed class Held : IDisposable
    {
        private readonly List<ID3D12Resource> _buffers;

        public Held(List<ID3D12Resource> buffers) => _buffers = buffers;

        public void Dispose()
        {
            foreach (ID3D12Resource buffer in _buffers)
            {
                buffer.Dispose();
            }
        }
    }

    /// <summary>One adapter DirectML could run on. <paramref name="Index"/> is
    /// the value to hand the constructor; indices follow DXGI's
    /// high-performance ordering, so 0 is what the default selection picks.
    /// <paramref name="Software"/> marks CPU rasterizers (WARP). They compute
    /// correctly at a small scale and cannot run a generation at all: a
    /// dispatch touching a buffer over 2 GiB removes the WARP device with
    /// DXGI_ERROR_DRIVER_INTERNAL_ERROR, and half a UNet is ~2.5 GiB. A front
    /// end should list hardware adapters only; the CLI still opens one by
    /// index, for the smoke test and the VAE.</summary>
    public readonly record struct AdapterInfo(
        int Index, string Name, ulong DedicatedMemory, bool Software);

    /// <summary>Every adapter, best first, software renderers last.</summary>
    public static IReadOnlyList<AdapterInfo> EnumerateAdapters()
    {
        using IDXGIFactory6 factory = DXGI.CreateDXGIFactory2<IDXGIFactory6>(false);
        var adapters = new List<AdapterInfo>();
        for (uint i = 0;
             factory.EnumAdapterByGpuPreference(i, GpuPreference.HighPerformance,
                 out IDXGIAdapter1? adapter).Success;
             i++)
        {
            using (adapter)
            {
                AdapterDescription1 description = adapter!.Description1;
                adapters.Add(new AdapterInfo((int)i, description.Description,
                    (ulong)description.DedicatedVideoMemory,
                    (description.Flags & AdapterFlags.Software) != AdapterFlags.None));
            }
        }
        return adapters;
    }

    /// <summary>Create the device on one adapter. The default picks the first
    /// hardware adapter in DXGI's high-performance order; a non-negative
    /// <paramref name="adapterIndex"/> — an <see cref="AdapterInfo.Index"/> —
    /// insists on that adapter, software (WARP) included, and throws if it
    /// cannot serve.</summary>
    public DmlDevice(bool useDebugLayer = false, int adapterIndex = -1)
    {
        if (useDebugLayer &&
            Vortice.Direct3D12.D3D12.D3D12GetDebugInterface(
                out Vortice.Direct3D12.Debug.ID3D12Debug? debug).Success)
        {
            debug!.EnableDebugLayer();
            debug.Dispose();
        }

        using IDXGIFactory6 factory = DXGI.CreateDXGIFactory2<IDXGIFactory6>(useDebugLayer);

        ID3D12Device? device = null;
        string adapterName = "unknown adapter";
        for (uint i = adapterIndex < 0 ? 0u : (uint)adapterIndex;
             factory.EnumAdapterByGpuPreference(i, GpuPreference.HighPerformance,
                 out IDXGIAdapter1? adapter).Success;
             i++)
        {
            using (adapter)
            {
                AdapterDescription1 description = adapter!.Description1;
                // Auto-selection never lands on a CPU rasterizer; picking one
                // on purpose, by index, is allowed.
                if (adapterIndex < 0 &&
                    (description.Flags & AdapterFlags.Software) != AdapterFlags.None)
                {
                    continue;
                }
                if (Vortice.Direct3D12.D3D12.D3D12CreateDevice(adapter,
                        Vortice.Direct3D.FeatureLevel.Level_11_0, out device).Success)
                {
                    adapterName = description.Description;
                    _adapter3 = adapter.QueryInterfaceOrNull<IDXGIAdapter3>();
                    break;
                }
                if (adapterIndex >= 0)
                {
                    throw new ArgumentException(
                        $"adapter {adapterIndex} ({description.Description}) " +
                        "cannot create a Direct3D 12 device", nameof(adapterIndex));
                }
            }
        }
        _d3dDevice = device ?? throw new InvalidOperationException(
            adapterIndex < 0
                ? "no hardware Direct3D 12 device is available"
                : $"no adapter at index {adapterIndex}");
        AdapterName = adapterName;
        // Anything created from here on is released again if a later step
        // throws: a device that fails to open is not one that gets disposed.
        try
        {
            _device1 = _d3dDevice.QueryInterfaceOrNull<ID3D12Device1>();
            _pool = new DmlBufferPool(_d3dDevice);

            _queue = _d3dDevice.CreateCommandQueue(new CommandQueueDescription
            {
                Type = CommandListType.Compute,
                Flags = CommandQueueFlags.None,
            });
            _allocator = _d3dDevice.CreateCommandAllocator(CommandListType.Compute);
            _list = _d3dDevice.CreateCommandList<ID3D12GraphicsCommandList>(
                CommandListType.Compute, _allocator);
            _fence = _d3dDevice.CreateFence();

            Device = DML.DMLCreateDevice<IDMLDevice1>(_d3dDevice,
                useDebugLayer ? CreateDeviceFlags.Debug : CreateDeviceFlags.None);
            _recorder = Device.CreateCommandRecorder();
            _initializer = Device.CreateOperatorInitializer(null);
        }
        catch
        {
            _initializer?.Dispose();
            _recorder?.Dispose();
            Device?.Dispose();
            _fence?.Dispose();
            _list?.Dispose();
            _allocator?.Dispose();
            _queue?.Dispose();
            _pool?.Dispose();
            _device1?.Dispose();
            _adapter3?.Dispose();
            _d3dDevice.Dispose();
            throw;
        }
    }

    /// <summary>Bind the owned inputs and run the operator initializer, leaving
    /// the result in the model's persistent resource. Once done, those inputs
    /// live on the GPU and dispatching does not touch them again.</summary>
    internal void Initialize(DmlCompiledModel model,
        IReadOnlyList<(int Index, HostTensor Tensor)> staged)
    {
        // Unchecked by the binding: a Reset that failed would leave the last
        // model's binding properties to be read below as this one's.
        _initializer.Reset(new[] { model.Op }).CheckError();

        (BufferBinding[] bindings, ulong inputsSize) = LayoutBindings(model.Inputs, owned: true);

        BindingProperties initializerProps = _initializer.GetBindingProperties();
        BindingProperties executeProps = model.Op.GetBindingProperties();

        // The staging goes whether or not the initialization came through: an
        // out-of-memory here leaves the device in use for the next run, which
        // should not find a model's worth of upload heap still allocated.
        try
        {
            _pool.EnsureInputs(inputsSize);
            _pool.EnsureTemporary(initializerProps.TemporaryResourceSize);
            ulong persistentBefore = model.PersistentBytes;
            bool createdPersistent =
                model.EnsurePersistentResource(_d3dDevice, executeProps.PersistentResourceSize);
            // The resource only ever grows, so this cannot go negative.
            _persistentBytes += model.PersistentBytes - persistentBefore;
            _pool.EnsureDescriptorHeap(initializerProps.RequiredDescriptorCount);

            if (createdPersistent && !model.Streamed && _device1 is not null &&
                model.PersistentResource is not null)
            {
                // The weights are read by every dispatch; ask the OS to demote
                // everything else first.
                _device1.SetResidencyPriority(1,
                    new ID3D12Pageable[] { model.PersistentResource },
                    new[] { ResidencyPriority.High });
            }

            CollectResidencySet(model.PersistentResource);
            AttachBuffer(bindings, _pool.Inputs);

            UploadStagedInputs(staged, model.Inputs, bindings, inputsSize, owned: true);

            ResetBindingTable(_initializer, initializerProps.RequiredDescriptorCount);
            _bindingTable!.BindInputs(new BindingDescription[]
            {
                new BufferArrayBinding { Bindings = bindings },
            });
            if (executeProps.PersistentResourceSize != 0)
            {
                _bindingTable.BindOutputs(new BindingDescription[]
                {
                    new BufferBinding
                    {
                        Buffer = model.PersistentResource!,
                        Offset = 0,
                        SizeInBytes = executeProps.PersistentResourceSize,
                    },
                });
            }
            if (initializerProps.TemporaryResourceSize != 0)
            {
                _bindingTable.BindTemporaryResource(new BufferBinding
                {
                    Buffer = _pool.Temporary,
                    Offset = 0,
                    SizeInBytes = initializerProps.TemporaryResourceSize,
                });
            }

            _list.SetDescriptorHeaps(1, new[] { _pool.DescriptorHeap });
            _recorder.RecordDispatch(_list, _initializer, _bindingTable);
            ExecuteAndWait("initialize");
        }
        finally
        {
            _pool.ReleaseStaging();
        }
        if (model.Streamed)
        {
            MoveWeightsToSystemMemory(model);
        }
        model.Initialized = true;
        Sample("staging released");
    }

    /// <summary>Copy a streamed model's folded weights out of video memory
    /// into a buffer in system memory, and drop the video-memory one. Each
    /// dispatch copies them back into the pool's window buffer first
    /// (<see cref="RecordDispatch"/>): one pass across the bus by the copy
    /// engine, which is the cheapest way the weights can get there.
    ///
    /// Two shorter routes were measured and rejected. Folding the weights
    /// straight into system memory removes the device: the initializer
    /// writing its output into a custom-heap buffer in L0 — CPU-visible or not
    /// — faults on the driver this was developed on, though its reads from
    /// the same kind of buffer (the staging inputs) always worked. And
    /// dispatching straight out of system memory, with no copy, ran a step
    /// four times slower than the copy does: a matrix multiply reads its
    /// weights once per tile of rows, and every one of those re-reads crossed
    /// the bus.</summary>
    private void MoveWeightsToSystemMemory(DmlCompiledModel model)
    {
        ID3D12Resource? resident = model.PersistentResource;
        if (resident is null)
        {
            return;
        }
        ulong size = resident.Description.Width;
        ID3D12Resource streamed = _d3dDevice.CreateCommittedResource(
            new HeapProperties(CpuPageProperty.WriteCombine, MemoryPool.L0), HeapFlags.None,
            ResourceDescription.Buffer(size, ResourceFlags.AllowUnorderedAccess),
            ResourceStates.CopyDest);

        try
        {
            _residencySet.Clear();
            _residencySet.Add(resident);
            _residencySet.Add(streamed);
            _list.ResourceBarrierTransition(resident,
                ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            _list.CopyBufferRegion(streamed, 0, resident, 0, size);
            // From here on the buffer is only ever the source of a copy.
            _list.ResourceBarrierTransition(streamed,
                ResourceStates.CopyDest, ResourceStates.CopySource);
            ExecuteAndWait("stream weights");
        }
        catch
        {
            // The model still owns the resident copy; only the new buffer
            // would leak.
            streamed.Dispose();
            throw;
        }

        model.ReplacePersistentResource(streamed);
        _persistentBytes -= size;
        _streamedBytes += size;
    }

    /// <summary>Upload the inputs that are not owned by DirectML, run the
    /// operator, and read the outputs back shaped by their descriptors.</summary>
    internal HostTensor[] Dispatch(DmlCompiledModel model,
        IReadOnlyList<(int Index, HostTensor Tensor)> staged)
    {
        if (!model.Initialized)
        {
            throw new InvalidOperationException("initialize the operator before dispatching it");
        }

        // An input owned by DirectML lives in the persistent resource and is not
        // bound at execution.
        (BufferBinding[] inputBindings, ulong inputsSize) =
            LayoutBindings(model.Inputs, owned: false);

        var outputBindings = new BufferBinding[model.OutputDescs.Length];
        ulong outputsSize = 0;
        for (int i = 0; i < model.OutputDescs.Length; i++)
        {
            ulong offset = RoundUp(outputsSize, TensorAlignment);
            outputBindings[i] = new BufferBinding
            {
                Offset = offset,
                SizeInBytes = model.OutputDescs[i].TotalBytes,
            };
            outputsSize = offset + model.OutputDescs[i].TotalBytes;
        }

        BindingProperties executeProps = model.Op.GetBindingProperties();

        _pool.EnsureInputs(inputsSize);
        _pool.EnsureReadbackHeap(outputsSize);
        _pool.EnsureOutputs(outputsSize);
        _pool.EnsureTemporary(executeProps.TemporaryResourceSize);
        _pool.EnsureDescriptorHeap(executeProps.RequiredDescriptorCount);

        CollectResidencySet(model.PersistentResource);

        AttachBuffer(inputBindings, _pool.Inputs);
        for (int i = 0; i < outputBindings.Length; i++)
        {
            outputBindings[i].Buffer = _pool.Outputs;
        }

        UploadStagedInputs(staged, model.Inputs, inputBindings, inputsSize, owned: false);

        var inputDescs = new BindingDescription[inputBindings.Length];
        for (int i = 0; i < inputBindings.Length; i++)
        {
            // A default description is DML_BINDING_TYPE_NONE, which is how an
            // owned input is skipped at execution.
            inputDescs[i] = inputBindings[i].SizeInBytes != 0
                ? inputBindings[i]
                : default(BindingDescription);
        }
        var outputDescs = new BindingDescription[outputBindings.Length];
        for (int i = 0; i < outputBindings.Length; i++)
        {
            outputDescs[i] = outputBindings[i];
        }

        RecordDispatch(model, executeProps, inputDescs, outputDescs);

        // Copy the outputs to the readback heap in the same submission.
        if (outputsSize != 0)
        {
            _list.ResourceBarrierTransition(_pool.Outputs,
                ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            _list.CopyBufferRegion(_pool.ReadbackHeap, 0, _pool.Outputs, 0, outputsSize);
            _list.ResourceBarrierTransition(_pool.Outputs,
                ResourceStates.CopySource, ResourceStates.UnorderedAccess);
        }
        ExecuteAndWait("dispatch");

        return DownloadOutputs(model.OutputDescs, outputBindings, outputsSize);
    }

    /// <summary>Run the operator over inputs already on the GPU, leaving the
    /// outputs there too, each in a buffer of its own that the caller owns
    /// from here on. Nothing crosses to the host; a chain of graphs runs link
    /// after link with only the fence between them.</summary>
    internal DeviceTensor[] DispatchOnDevice(DmlCompiledModel model,
        IReadOnlyList<(int Index, DeviceTensor Tensor)> inputs)
    {
        if (!model.Initialized)
        {
            throw new InvalidOperationException("initialize the operator before dispatching it");
        }

        BindingProperties executeProps = model.Op.GetBindingProperties();
        _pool.EnsureTemporary(executeProps.TemporaryResourceSize);
        _pool.EnsureDescriptorHeap(executeProps.RequiredDescriptorCount);

        CollectResidencySet(model.PersistentResource);

        var inputDescs = new BindingDescription[model.Inputs.Length];
        foreach ((int index, DeviceTensor tensor) in inputs)
        {
            inputDescs[index] = new BufferBinding
            {
                Buffer = tensor.Buffer,
                Offset = 0,
                SizeInBytes = tensor.TotalBytes,
            };
            _residencySet.Add(tensor.Buffer);
        }

        var outputs = new DeviceTensor[model.OutputDescs.Length];
        var outputDescs = new BindingDescription[model.OutputDescs.Length];
        // Renting is inside the try: a graph with several outputs — the
        // UNet's down blocks hand on their skips — can run out of memory on
        // the second, and the first then goes back to the pool.
        try
        {
            for (int i = 0; i < outputs.Length; i++)
            {
                DmlTensorDesc desc = model.OutputDescs[i];
                ID3D12Resource buffer = _pool.RentBuffer(desc.TotalBytes, shared: false);
                outputs[i] = new DeviceTensor(this, buffer, desc, shared: false);
                outputDescs[i] = new BufferBinding
                {
                    Buffer = buffer,
                    Offset = 0,
                    SizeInBytes = desc.TotalBytes,
                };
                _residencySet.Add(buffer);
            }

            RecordDispatch(model, executeProps, inputDescs, outputDescs);
            ExecuteAndWait("dispatch");
        }
        catch
        {
            foreach (DeviceTensor? output in outputs)
            {
                output?.Dispose();
            }
            throw;
        }

        return outputs;
    }

    /// <summary>Bind everything and record the operator into the open command
    /// list. A streamed model's weights are first copied from system memory
    /// into the window buffer, in the same submission, and the operator is
    /// bound to that.</summary>
    private void RecordDispatch(DmlCompiledModel model, BindingProperties executeProps,
        BindingDescription[] inputDescs, BindingDescription[] outputDescs)
    {
        ID3D12Resource? persistent = model.PersistentResource;
        if (model.Streamed && persistent is not null)
        {
            ulong size = persistent.Description.Width;
            _pool.EnsureWindow(size);
            ID3D12Resource window = _pool.Window;
            _residencySet.Add(window);
            _list.ResourceBarrierTransition(window,
                ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
            _list.CopyBufferRegion(window, 0, persistent, 0, size);
            _list.ResourceBarrierTransition(window,
                ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
            persistent = window;
        }

        ResetBindingTable(model.Op, executeProps.RequiredDescriptorCount);
        _bindingTable!.BindInputs(inputDescs);
        _bindingTable.BindOutputs(outputDescs);

        if (executeProps.PersistentResourceSize != 0)
        {
            _bindingTable.BindPersistentResource(new BufferBinding
            {
                Buffer = persistent!,
                Offset = 0,
                SizeInBytes = executeProps.PersistentResourceSize,
            });
        }
        if (executeProps.TemporaryResourceSize != 0)
        {
            _bindingTable.BindTemporaryResource(new BufferBinding
            {
                Buffer = _pool.Temporary,
                Offset = 0,
                SizeInBytes = executeProps.TemporaryResourceSize,
            });
        }

        _list.SetDescriptorHeaps(1, new[] { _pool.DescriptorHeap });
        _recorder.RecordDispatch(_list, model.Op, _bindingTable);
    }

    /// <summary>Put a host tensor where graphs can read it. The buffer is in
    /// system memory and written by the CPU directly, so this costs one copy
    /// and no submission; the GPU reads it across the bus, which for the
    /// kilobytes a conditioning vector weighs is nothing.</summary>
    public DeviceTensor Upload(HostTensor tensor)
    {
        var desc = DmlTensorDesc.Packed(DmlTensorDesc.ToDataType(tensor.DataType),
            tensor.Shape.Select(extent => checked((uint)extent)).ToArray());
        ID3D12Resource buffer = _pool.RentBuffer(desc.TotalBytes, shared: true);
        var result = new DeviceTensor(this, buffer, desc, shared: true);
        try
        {
            byte* data = buffer.Map<byte>(0);
            try
            {
                ReadOnlySpan<byte> bytes = tensor.Data.Span;
                fixed (byte* source = bytes)
                {
                    Buffer.MemoryCopy(source, data, (long)desc.TotalBytes, bytes.Length);
                }
                for (ulong pad = (ulong)bytes.Length; pad < desc.TotalBytes; pad++)
                {
                    data[pad] = 0;
                }
            }
            finally
            {
                buffer.Unmap(0);
            }
        }
        catch
        {
            result.Dispose();
            throw;
        }
        return result;
    }

    /// <summary>Copy a tensor on the GPU back to the host: one submission
    /// holding a single copy, then a wait.</summary>
    internal HostTensor Download(DeviceTensor tensor)
    {
        DmlTensorDesc desc = tensor.Desc;
        if (desc.Strides is not null)
        {
            throw new NotSupportedException("strided tensors cannot be downloaded");
        }
        _pool.EnsureReadbackHeap(desc.TotalBytes);
        _residencySet.Clear();
        _residencySet.Add(_pool.ReadbackHeap);
        _residencySet.Add(tensor.Buffer);

        _list.ResourceBarrierTransition(tensor.Buffer,
            ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        _list.CopyBufferRegion(_pool.ReadbackHeap, 0, tensor.Buffer, 0, desc.TotalBytes);
        _list.ResourceBarrierTransition(tensor.Buffer,
            ResourceStates.CopySource, ResourceStates.UnorderedAccess);
        ExecuteAndWait("download");

        var binding = new BufferBinding { Offset = 0, SizeInBytes = desc.TotalBytes };
        return DownloadOutputs(new[] { desc }, new[] { binding }, desc.TotalBytes)[0];
    }

    internal void ReturnBuffer(ID3D12Resource buffer, bool shared) =>
        _pool.ReturnBuffer(buffer, shared);

    /// <summary>Let go of the buffers the pool is holding for reuse. A chain
    /// calls this when it is disposed: its activations are no use to the
    /// decoder that follows, and they are video memory the decoder can want.</summary>
    public void TrimBuffers() => _pool.TrimFree();

    /// <summary>Let go of every pooled buffer, so that
    /// <see cref="VideoMemory"/> reports what the models hold and nothing
    /// else. Meant for the moment before a decision is made on that number.</summary>
    public void ReleasePooledBuffers() => _pool.ReleaseAll();

    /// <summary>The floor under the scratch buffer's size; see
    /// <see cref="DmlBufferPool.EnsureTemporary"/>. Zero clears it.</summary>
    public ulong TemporaryFloor
    {
        get => _pool.TemporaryFloor;
        set => _pool.TemporaryFloor = value;
    }

    /// <summary>Lay one phase's inputs out end to end in the inputs buffer, each
    /// aligned as DirectML requires. A slot belonging to the other phase keeps a
    /// zero-sized binding, which is how it is skipped.</summary>
    private static (BufferBinding[] Bindings, ulong TotalSize) LayoutBindings(
        DmlGraph.InputSlot[] slots, bool owned)
    {
        var bindings = new BufferBinding[slots.Length];
        ulong size = 0;
        for (int i = 0; i < slots.Length; i++)
        {
            if (slots[i].Owned != owned)
            {
                continue;
            }
            ulong offset = RoundUp(size, TensorAlignment);
            bindings[i] = new BufferBinding
            {
                Offset = offset,
                SizeInBytes = slots[i].Desc.TotalBytes,
            };
            size = offset + slots[i].Desc.TotalBytes;
        }
        return (bindings, size);
    }

    private static void AttachBuffer(BufferBinding[] bindings, ID3D12Resource buffer)
    {
        for (int i = 0; i < bindings.Length; i++)
        {
            if (bindings[i].SizeInBytes != 0)
            {
                bindings[i].Buffer = buffer;
            }
        }
    }

    /// <summary>Copy this phase's inputs into the staging buffer. Write-combined
    /// memory: the writes here are sequential and never read back, which is the
    /// access pattern it is fast for.</summary>
    private void UploadStagedInputs(IReadOnlyList<(int Index, HostTensor Tensor)> staged,
        DmlGraph.InputSlot[] slots, BufferBinding[] bindings, ulong inputsSize, bool owned)
    {
        if (inputsSize == 0)
        {
            return;
        }

        // Written straight into the buffer DirectML will read: it is in system
        // memory and CPU-writable, so there is no second copy to stage through.
        byte* uploadData = _pool.Inputs.Map<byte>(0);
        try
        {
            foreach ((int index, HostTensor tensor) in staged)
            {
                if (index >= slots.Length || slots[index].Owned != owned)
                {
                    throw new ArgumentException($"staged input {index} is not bound in this phase");
                }
                BufferBinding binding = bindings[index];
                if ((ulong)tensor.Data.Length > binding.SizeInBytes)
                {
                    throw new ArgumentException(
                        $"staged input {index} does not fit its tensor " +
                        $"({tensor.Data.Length} > {binding.SizeInBytes} bytes)");
                }

                ReadOnlySpan<byte> bytes = tensor.Data.Span;
                fixed (byte* source = bytes)
                {
                    Buffer.MemoryCopy(source, uploadData + binding.Offset,
                        (long)binding.SizeInBytes, bytes.Length);
                }
                // DirectML rounds a tensor's size up to a 4-byte boundary; the
                // tail still has to hold defined values.
                for (ulong pad = (ulong)tensor.Data.Length; pad < binding.SizeInBytes; pad++)
                {
                    uploadData[binding.Offset + pad] = 0;
                }
            }
        }
        finally
        {
            _pool.Inputs.Unmap(0);
        }
    }

    private HostTensor[] DownloadOutputs(DmlTensorDesc[] outputDescs,
        BufferBinding[] bindings, ulong outputsSize)
    {
        var outputs = new HostTensor[outputDescs.Length];
        if (outputsSize == 0)
        {
            return outputs;
        }

        byte* readbackData = _pool.ReadbackHeap.Map<byte>(0);
        try
        {
            for (int i = 0; i < outputDescs.Length; i++)
            {
                DmlTensorDesc desc = outputDescs[i];
                if (desc.Strides is not null)
                {
                    throw new NotSupportedException("strided graph outputs are not supported");
                }
                HostDataType dataType = DmlTensorDesc.ToHostDataType(desc.DataType);
                int[] shape = desc.Sizes.Select(extent => checked((int)extent)).ToArray();
                var data = new byte[desc.ElementCount * (long)HostTensor.BytesPerElement(dataType)];
                fixed (byte* destination = data)
                {
                    Buffer.MemoryCopy(readbackData + bindings[i].Offset, destination,
                        data.LongLength, data.LongLength);
                }
                outputs[i] = new HostTensor(dataType, shape, data);
            }
        }
        finally
        {
            _pool.ReadbackHeap.Unmap(0);
        }
        return outputs;
    }

    private void ResetBindingTable(IDMLDispatchable dispatchable, uint descriptorCount)
    {
        var description = new BindingTableDescription
        {
            Dispatchable = dispatchable,
            CPUDescriptorHandle = _pool.DescriptorHeap.GetCPUDescriptorHandleForHeapStart(),
            GPUDescriptorHandle = _pool.DescriptorHeap.GetGPUDescriptorHandleForHeapStart(),
            SizeInDescriptors = descriptorCount,
        };
        if (_bindingTable is null)
        {
            _bindingTable = Device.CreateBindingTable(description);
        }
        else
        {
            _bindingTable.Reset(description).CheckError();
        }
    }

    /// <summary>Everything the recorded submission can touch, for one
    /// MakeResident call in <see cref="ExecuteAndWait"/>.</summary>
    private void CollectResidencySet(ID3D12Resource? persistent)
    {
        _residencySet.Clear();
        _pool.CollectInto(_residencySet);
        if (persistent is not null)
        {
            _residencySet.Add(persistent);
        }
    }

    /// <summary>What memory looked like when something went wrong. The
    /// system-memory figure is the one that distinguishes the two failures that
    /// look alike: nothing there means the card genuinely ran out, while a large
    /// number means the working set spilled across PCIe and the dispatch was
    /// killed for taking too long rather than for wanting too much.</summary>
    private string DescribeBudget()
    {
        if (_adapter3 is null)
        {
            return "";
        }
        (QueryVideoMemoryInfo local, QueryVideoMemoryInfo nonLocal) = QueryMemory();
        return $" (video memory: {local.CurrentUsage >> 20} MB of a " +
               $"{local.Budget >> 20} MB budget in use, plus " +
               $"{nonLocal.CurrentUsage >> 20} MB in system memory)";
    }

    private void ExecuteAndWait(string what)
    {
        _list.Close();

        // Taken before the submission, not after it. DXGI's usage figures lag,
        // so a reading at the instant a dispatch completes can still describe
        // the state before it; and an initialize holds its peak — the staging
        // copy and the persistent resource both alive — only until it returns.
        Sample(what);

        if (_residencySet.Count > 0)
        {
            try
            {
                _d3dDevice.MakeResident(_residencySet.ToArray());
            }
            catch (SharpGenException exception)
            {
                _residencySet.Clear();
                // The list is closed with this dispatch still recorded in it.
                // The device is meant to stay usable after this — a caller
                // keeps it and asks again with a smaller size — so the list
                // has to be empty and open for the next recording.
                _allocator.Reset();
                _list.Reset(_allocator);
                throw new OutOfVideoMemoryException(
                    "not enough video memory to make this dispatch resident" +
                    DescribeBudget(), exception);
            }
            _residencySet.Clear();
        }

        _queue.ExecuteCommandList(_list);
        _queue.Signal(_fence, ++_fenceValue);
        if (_fence.CompletedValue < _fenceValue)
        {
            _fence.SetEventOnCompletion(_fenceValue, _fenceEvent).CheckError();
            _fenceEvent.WaitOne();
        }

        Result removed = _d3dDevice.DeviceRemovedReason;
        if (removed.Failure)
        {
            throw new DeviceRemovedException(
                $"the device was removed: {removed}{DescribeBudget()}");
        }

        _allocator.Reset();
        _list.Reset(_allocator);
    }

    internal static ulong RoundUp(ulong value, ulong multiple)
    {
        ulong remainder = value % multiple;
        return remainder == 0 ? value : value + multiple - remainder;
    }

    public void Dispose()
    {
        _bindingTable?.Dispose();
        _initializer.Dispose();
        _recorder.Dispose();
        _pool.Dispose();
        Device.Dispose();
        _fence.Dispose();
        _list.Dispose();
        _allocator.Dispose();
        _queue.Dispose();
        _device1?.Dispose();
        _adapter3?.Dispose();
        _d3dDevice.Dispose();
        _fenceEvent.Dispose();
    }
}
