using NaiveDiffusion.Dml;
using NaiveDiffusion.Tensors;

namespace NaiveDiffusion.Graph;

/// <summary>One model built as a sequence of graphs that run back to back,
/// with the tensors between them staying on the GPU.
///
/// The builder writes the model as if it were one graph, naming the values
/// that outlive a link — the running activation, a skip connection, an
/// embedding every block reads — and calling <see cref="Cut"/> wherever a
/// graph should end. Whatever named values the link produced become its
/// outputs; whatever the next link asks for by name becomes a placeholder
/// there. At run time each name is a <see cref="DeviceTensor"/> handed from the
/// link that made it to every link that reads it, and dropped after the last.
///
/// Cutting is what makes the pieces individually small: each has its own
/// persistent resource, its own staging peak while it initializes, and its own
/// dispatch — and a piece can keep its weights in system memory
/// (a resident budget) where a whole model could not afford to.</summary>
public sealed class GraphChain : IDisposable
{
    private sealed class Link
    {
        public required ModelBuilder Model { get; init; }
        public required string[] Inputs { get; init; }
        public required string[] Outputs { get; init; }
        public required string[] DropAfter { get; init; }
    }

    private readonly DmlDevice _device;
    private readonly HostDataType _dataType;
    private readonly List<Link> _links = new();

    // The link under construction.
    private ModelBuilder _current;
    private readonly List<string> _currentInputs = new();
    private readonly Dictionary<string, DmlExpression> _values = new();
    private readonly List<string> _produced = new();
    private readonly List<string> _dropped = new();

    // Every value that crosses between links, by shape and precision, so the
    // next link can declare a placeholder for it. A value is carried at the
    // width it was produced at, which need not be the chain's: a residual
    // stream can stay at single precision between half-precision links.
    private readonly Dictionary<string, int[]> _shapes = new();
    private readonly Dictionary<string, HostDataType> _types = new();
    // The declared inputs' widths outlive a Drop: Run converts what it is
    // handed to them.
    private readonly Dictionary<string, HostDataType> _inputTypes = new();
    private readonly HashSet<string> _declared = new();
    private string? _result;
    private ulong _maxTemporary;

    /// <summary>How much of the weights the chain was built to keep in video
    /// memory. Null keeps everything there. Otherwise links are placed there
    /// in order until the budget is spent and the rest go to system memory,
    /// which costs their size across the bus per run and nothing else. Fixed
    /// at build time: where a link's weights live is where its persistent
    /// resource was created.</summary>
    private ulong? _residentBudget;
    private ulong _residentSoFar;

    public GraphChain(DmlDevice device, HostDataType dataType, ulong? residentBudget = null,
        bool int8Weights = false)
    {
        _device = device;
        _dataType = dataType;
        _int8Weights = int8Weights;
        _residentBudget = residentBudget;
        _current = new ModelBuilder(device, dataType, int8Weights);
    }

    private readonly bool _int8Weights;

    /// <summary>The graph under construction: where layers go until the next cut.</summary>
    public ModelBuilder Current => _current;

    /// <summary>Declare a value the caller supplies to <see cref="Run"/>, at
    /// the chain's precision unless <paramref name="dataType"/> says otherwise;
    /// the tensor handed to <see cref="Run"/> is converted to it. Nothing is
    /// added to the current link until <see cref="Get"/> reads the value: a
    /// graph input no node consumes is one DirectML binds and never reads,
    /// and the first link of a chain that declares everything up front would
    /// otherwise carry several of those.</summary>
    public void Input(string name, int[] shape, HostDataType? dataType = null)
    {
        if (!_declared.Add(name))
        {
            throw new ArgumentException($"{name} was declared twice");
        }
        _shapes[name] = shape;
        _types[name] = _inputTypes[name] = dataType ?? _dataType;
    }

    /// <summary>The value called <paramref name="name"/>, as an expression in
    /// the current link: the expression that produced it if that was here, or
    /// a placeholder fed from an earlier link otherwise.</summary>
    public DmlExpression Get(string name)
    {
        if (_values.TryGetValue(name, out DmlExpression? value))
        {
            return value;
        }
        if (!_shapes.ContainsKey(name))
        {
            throw new ArgumentException($"{name} has not been produced yet");
        }
        return Placeholder(name);
    }

    private DmlExpression Placeholder(string name)
    {
        DmlExpression placeholder = _current.Placeholder(_shapes[name], _types[name]);
        _values[name] = placeholder;
        _currentInputs.Add(name);
        return placeholder;
    }

    /// <summary>The shape of a carried value, without pulling it into the link.</summary>
    public int[] Shape(string name) => _shapes[name];

    /// <summary>Name a value so later links can read it. Naming it again
    /// replaces it, from this link on.</summary>
    public void Set(string name, DmlExpression value)
    {
        if (!ReferenceEquals(value.Graph, _current.Graph))
        {
            throw new ArgumentException($"{name} belongs to an earlier link");
        }
        _values[name] = value;
        _shapes[name] = value.Shape.Select(extent => checked((int)extent)).ToArray();
        _types[name] = DmlTensorDesc.ToHostDataType(value.Desc.DataType);
        if (!_produced.Contains(name))
        {
            _produced.Add(name);
        }
    }

    /// <summary>Nothing after this link reads <paramref name="name"/>: its
    /// tensor is released once the link has run, and it is not an output.</summary>
    public void Drop(string name)
    {
        if (!_shapes.Remove(name))
        {
            throw new ArgumentException($"{name} is not a live value");
        }
        _types.Remove(name);
        _values.Remove(name);
        _produced.Remove(name);
        if (_currentInputs.Contains(name))
        {
            _dropped.Add(name);
        }
    }

    /// <summary>End the current link and start the next. Every value named
    /// in this link since the last cut is an output; the placeholders it
    /// declared are its inputs.</summary>
    public void Cut()
    {
        var outputs = new List<DmlExpression>(_produced.Count);
        foreach (string name in _produced)
        {
            DmlExpression value = _values[name];
            // A graph input cannot be a graph output; an untouched pass-through
            // goes through an identity.
            outputs.Add(value.InputIndex >= 0 ? DmlOps.Identity(value, 1.0f, 0.0f) : value);
        }

        ModelBuilder model = _current;
        model.Compile(outputs, streamWhen: persistent =>
        {
            if (_residentBudget is null || _residentSoFar + persistent <= _residentBudget)
            {
                _residentSoFar += persistent;
                return false;
            }
            return true;
        });
        _maxTemporary = Math.Max(_maxTemporary, model.TemporarySize);
        // The link's host copies — weights merged or converted on the way in —
        // are garbage now that they are uploaded. Left to itself the GC lets
        // several graphs' worth pile up before it looks, and that pile is the
        // host peak of a LoRA build; collected here, the next link reuses the
        // same space.
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: false);

        _links.Add(new Link
        {
            Model = model,
            Inputs = _currentInputs.ToArray(),
            Outputs = _produced.ToArray(),
            DropAfter = _dropped.ToArray(),
        });

        _current = new ModelBuilder(_device, _dataType, _int8Weights);
        _currentInputs.Clear();
        _values.Clear();
        _produced.Clear();
        _dropped.Clear();
    }

    /// <summary>Cut the last link and say which value <see cref="Run"/> returns.</summary>
    public void Finish(string result)
    {
        Cut();
        if (!_shapes.ContainsKey(result))
        {
            throw new ArgumentException($"{result} is not a live value at the end");
        }
        _result = result;
    }

    /// <summary>Feed the declared inputs through every link in turn and bring
    /// the result back to the host. Everything else stays on the GPU: each
    /// tensor lives from the link that produced it to the last that reads it.</summary>
    public HostTensor Run(IReadOnlyDictionary<string, HostTensor> inputs)
    {
        if (_result is null)
        {
            throw new InvalidOperationException("Finish() the chain before running it");
        }

        var live = new Dictionary<string, DeviceTensor>();
        try
        {
            foreach ((string name, HostTensor tensor) in inputs)
            {
                if (!_declared.Contains(name))
                {
                    throw new ArgumentException($"{name} is not an input of this chain");
                }
                live[name] = _device.Upload(tensor.ConvertTo(_inputTypes[name]));
            }

            // Sized once for the largest link rather than re-created at every
            // link that wants a different size.
            _device.TemporaryFloor = _maxTemporary;
            foreach (Link link in _links)
            {
                var arguments = new DeviceTensor[link.Inputs.Length];
                for (int i = 0; i < arguments.Length; i++)
                {
                    if (!live.TryGetValue(link.Inputs[i], out DeviceTensor? argument))
                    {
                        throw new InvalidOperationException($"{link.Inputs[i]} was not supplied");
                    }
                    arguments[i] = argument;
                }

                DeviceTensor[] outputs = link.Model.RunOnDevice(arguments);
                for (int i = 0; i < outputs.Length; i++)
                {
                    // Renaming over a value replaces it; the dispatch that read
                    // the old one has completed, so it can go now.
                    if (live.TryGetValue(link.Outputs[i], out DeviceTensor? replaced))
                    {
                        replaced.Dispose();
                    }
                    live[link.Outputs[i]] = outputs[i];
                }
                foreach (string name in link.DropAfter)
                {
                    if (live.Remove(name, out DeviceTensor? dropped))
                    {
                        dropped.Dispose();
                    }
                }
            }

            return live[_result].Download();
        }
        finally
        {
            _device.TemporaryFloor = 0;
            foreach (DeviceTensor tensor in live.Values)
            {
                tensor.Dispose();
            }
        }
    }

    /// <summary>Video memory the links' weights would occupy all resident.</summary>
    public ulong PersistentBytes => _links.Aggregate(0UL, (total, link) => total + link.Model.PersistentBytes);

    /// <summary>Of those, the bytes kept in system memory and read across the
    /// bus on every run.</summary>
    public ulong StreamedBytes => _links.Where(link => link.Model.Compiled.Streamed)
        .Aggregate(0UL, (total, link) => total + link.Model.PersistentBytes);

    /// <summary>And the bytes that are in video memory.</summary>
    public ulong ResidentBytes => PersistentBytes - StreamedBytes;

    /// <summary>Scratch the largest link needs; the whole chain runs in that much.</summary>
    public ulong TemporaryBytes => _maxTemporary;

    public int LinkCount => _links.Count;

    public void Dispose()
    {
        foreach (Link link in _links)
        {
            link.Model.Dispose();
        }
        _links.Clear();
        _current.Dispose();
        // The activations' buffers are pooled by the device; nothing that
        // follows wants them at these sizes.
        _device.TrimBuffers();
    }
}
