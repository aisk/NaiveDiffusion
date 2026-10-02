using NaiveDiffusion.Tensors;
using Vortice.Direct3D12;
using Vortice.DirectML;

namespace NaiveDiffusion.Dml;

/// <summary>A compiled graph, self-contained: it snapshots the input slots and
/// output descriptions at compile time so the graph can be dropped, and it owns
/// the persistent resource its weights are folded into.</summary>
public sealed class DmlCompiledModel : IDisposable
{
    internal IDMLCompiledOperator Op { get; }
    internal DmlGraph.InputSlot[] Inputs { get; }
    internal DmlTensorDesc[] OutputDescs { get; }
    internal ID3D12Resource? PersistentResource { get; private set; }
    internal bool Initialized { get; set; }

    /// <summary>Whether the folded weights live in system memory instead of
    /// video memory, copied across the bus into a window buffer for each
    /// dispatch. A streamed model holds no video memory between dispatches
    /// and costs its persistent size in bandwidth per dispatch. Decided
    /// before initialization, because that is when the weights are moved
    /// (see <see cref="DmlDevice.Initialize"/>).
    ///
    /// Explicit paging was tried first and lost: evicting a resource and
    /// making it resident again copies it out and back through the OS's
    /// paging path, which measured at about a gigabyte a second and doubled
    /// the traffic, where the copy here costs one pass at bus speed.</summary>
    public bool Streamed
    {
        get => _streamed;
        set
        {
            if (Initialized)
            {
                throw new InvalidOperationException(
                    "where the weights live is decided before they are uploaded");
            }
            _streamed = value;
        }
    }

    private bool _streamed;

    private readonly DmlDevice _device;

    internal DmlCompiledModel(DmlDevice device, IDMLCompiledOperator op,
        DmlGraph.InputSlot[] inputs, DmlTensorDesc[] outputDescs)
    {
        _device = device;
        Op = op;
        Inputs = inputs;
        OutputDescs = outputDescs;
    }

    /// <summary>Bytes of scratch memory one dispatch of this graph needs; every
    /// intermediate tensor lives there.</summary>
    public ulong TemporarySize => Op.GetBindingProperties().TemporaryResourceSize;

    /// <summary>Bytes the folded weights occupy, wherever they live.</summary>
    public ulong PersistentSize => Op.GetBindingProperties().PersistentResourceSize;

    /// <summary>Upload the owned inputs and run the operator initializer. The
    /// data is read once, here, and lives on the GPU from then on; nothing here
    /// keeps a copy.</summary>
    public void Initialize(IReadOnlyDictionary<DmlExpression, HostTensor> weights)
    {
        if (Initialized)
        {
            // A second pass would bind a streamed model's system-memory copy
            // as the initializer's output and count its bytes twice.
            throw new InvalidOperationException("the weights were already uploaded");
        }
        _device.Initialize(this, Stage(weights, owned: true));
    }

    /// <summary>Run the graph. Returns one tensor per compiled output, in order.
    /// A graph without owned inputs initializes itself on the first call.</summary>
    public HostTensor[] Dispatch(IReadOnlyDictionary<DmlExpression, HostTensor>? inputs = null)
    {
        if (!Initialized)
        {
            if (Inputs.Any(slot => slot.Owned))
            {
                throw new InvalidOperationException(
                    "initialize this operator before dispatching it; it has owned inputs");
            }
            _device.Initialize(this, Array.Empty<(int, HostTensor)>());
        }
        return _device.Dispatch(this,
            Stage(inputs ?? new Dictionary<DmlExpression, HostTensor>(), owned: false));
    }

    /// <summary>Run the graph over tensors already on the GPU and leave the
    /// outputs there. Every non-owned input has to be supplied.</summary>
    public DeviceTensor[] DispatchOnDevice(IReadOnlyDictionary<DmlExpression, DeviceTensor> inputs)
    {
        if (!Initialized)
        {
            if (Inputs.Any(slot => slot.Owned))
            {
                throw new InvalidOperationException(
                    "initialize this operator before dispatching it; it has owned inputs");
            }
            _device.Initialize(this, Array.Empty<(int, HostTensor)>());
        }

        var staged = new List<(int, DeviceTensor)>(inputs.Count);
        var provided = new HashSet<int>();
        foreach ((DmlExpression expression, DeviceTensor tensor) in inputs)
        {
            int index = expression.InputIndex;
            if (index < 0 || index >= Inputs.Length)
            {
                throw new ArgumentException($"{expression} is not an input of this graph");
            }
            DmlGraph.InputSlot slot = Inputs[index];
            if (slot.Owned)
            {
                throw new ArgumentException($"input {index} is owned by DirectML");
            }
            if (slot.Desc.DataType != tensor.Desc.DataType ||
                slot.Desc.TotalBytes != tensor.TotalBytes)
            {
                throw new ArgumentException(
                    $"{tensor} does not fit input {index} ({slot.Desc})");
            }
            if (!provided.Add(index))
            {
                throw new ArgumentException($"input {index} was bound twice");
            }
            staged.Add((index, tensor));
        }
        for (int index = 0; index < Inputs.Length; index++)
        {
            if (!Inputs[index].Owned && !provided.Contains(index))
            {
                throw new ArgumentException($"missing input {index} ({Inputs[index].Desc})");
            }
        }
        return _device.DispatchOnDevice(this, staged);
    }

    /// <summary>Match the dict against the graph's inputs: everything that can
    /// go wrong is raised from here, before any upload starts.</summary>
    private List<(int, HostTensor)> Stage(
        IReadOnlyDictionary<DmlExpression, HostTensor> values, bool owned)
    {
        var staged = new List<(int, HostTensor)>(values.Count);
        var provided = new HashSet<int>();

        foreach ((DmlExpression expression, HostTensor tensor) in values)
        {
            int index = expression.InputIndex;
            if (index < 0 || index >= Inputs.Length)
            {
                throw new ArgumentException($"{expression} is not an input of this graph");
            }
            DmlGraph.InputSlot slot = Inputs[index];
            if (slot.Owned != owned)
            {
                throw new ArgumentException(owned
                    ? $"input {index} is not owned by DirectML; bind it at dispatch instead"
                    : $"input {index} is owned by DirectML; bind it at initialize instead");
            }

            HostDataType target = DmlTensorDesc.ToHostDataType(slot.Desc.DataType);
            HostTensor converted = tensor.DataType == target ? tensor : tensor.ConvertTo(target);

            if (converted.ElementCount != slot.Desc.ElementCount)
            {
                throw new ArgumentException(
                    $"an array of {converted.ElementCount} elements does not fill input {index} ({slot.Desc})");
            }

            if (!provided.Add(index))
            {
                throw new ArgumentException($"input {index} was bound twice");
            }
            staged.Add((index, converted));
        }

        for (int index = 0; index < Inputs.Length; index++)
        {
            if (Inputs[index].Owned == owned && !provided.Contains(index))
            {
                throw new ArgumentException(
                    $"missing input {index} ({Inputs[index].Desc})");
            }
        }
        return staged;
    }

    /// <summary>Memory this model's weights occupy — video memory, or system
    /// memory when streamed — which is all of the memory it holds; everything
    /// else it uses comes from the shared pool.</summary>
    internal ulong PersistentBytes => PersistentResource?.Description.Width ?? 0;

    /// <summary>Returns whether a new resource was created.</summary>
    internal bool EnsurePersistentResource(ID3D12Device device, ulong size)
    {
        if (size == 0 || (PersistentResource is not null &&
                          PersistentResource.Description.Width >= size))
        {
            return false;
        }
        // Released and forgotten before the replacement is asked for, as
        // DmlBufferPool does: a failed creation must not leave a field naming
        // a resource that is gone.
        PersistentResource?.Dispose();
        PersistentResource = null;
        PersistentResource = device.CreateCommittedResource(
            HeapProperties.DefaultHeapProperties, HeapFlags.None,
            ResourceDescription.Buffer(size, ResourceFlags.AllowUnorderedAccess),
            ResourceStates.UnorderedAccess);
        return true;
    }

    /// <summary>Swap in the system-memory copy of the folded weights, and
    /// drop the video-memory one.</summary>
    internal void ReplacePersistentResource(ID3D12Resource resource)
    {
        PersistentResource?.Dispose();
        PersistentResource = resource;
        _weightsInSystemMemory = true;
    }

    // Where the bytes were counted — not where they were meant to go: a
    // streamed model whose copy to system memory failed still holds them in
    // video memory, and gives them back from there.
    private bool _weightsInSystemMemory;

    public void Dispose()
    {
        _device.ForgetPersistent(PersistentBytes, _weightsInSystemMemory);
        PersistentResource?.Dispose();
        PersistentResource = null;
        Op.Dispose();
    }
}
