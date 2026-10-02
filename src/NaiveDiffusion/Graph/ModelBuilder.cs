using NaiveDiffusion.Dml;
using NaiveDiffusion.Tensors;

namespace NaiveDiffusion.Graph;

/// <summary>A graph under construction, plus the tensors bound to its weights.
///
/// Weights are registered with <see cref="Constant"/> and carry their data with
/// them; anything fed per run is registered with <see cref="Placeholder"/> and
/// supplied to <see cref="Run"/>, in order.</summary>
public sealed class ModelBuilder : IDisposable
{
    public DmlGraph Graph { get; }

    private readonly HostDataType _dataType;
    private readonly Dictionary<DmlExpression, HostTensor> _weights = new();
    private readonly List<DmlExpression> _placeholders = new();
    private DmlCompiledModel? _operator;

    public ModelBuilder(DmlDevice device, HostDataType dataType = HostDataType.Float32,
        bool int8Weights = false)
    {
        Graph = new DmlGraph(device);
        _dataType = dataType;
        Int8Weights = int8Weights;
    }

    /// <summary>Whether <see cref="Weight"/> stores the large matrices as
    /// block-quantized int8 (see <see cref="BlockQuantizer"/>) and dequantizes
    /// them in the graph, which halves what the folded weights take. Only the
    /// storage changes: DirectML keeps the int8 in the persistent resource and
    /// writes the dequantized copy into the dispatch's scratch, so the gemm
    /// itself runs at the builder's precision as before.</summary>
    public bool Int8Weights { get; }

    /// <summary>Weights with fewer elements than this stay at full precision:
    /// they are not where the bytes are, and the small ones — the time
    /// embedding, the convolutions in and out — are the sensitive ones.</summary>
    private const long QuantizeFrom = BlockQuantizer.MinimumElements;

    /// <summary>Register a weight, reshaped to <paramref name="shape"/> if given,
    /// at the builder's precision unless <paramref name="dataType"/> says
    /// otherwise. Owned inputs are handed to DirectML once, at initialization,
    /// and stay on the GPU from then on.</summary>
    public DmlExpression Constant(HostTensor tensor, int[]? shape = null,
        HostDataType? dataType = null)
    {
        HostDataType type = dataType ?? _dataType;
        tensor = tensor.ConvertTo(type);
        if (shape is not null)
        {
            tensor = tensor.Reshape(shape);
        }
        DmlExpression expression = Graph.Input(type, tensor.Shape, owned: true);
        _weights[expression] = tensor;
        return expression;
    }

    /// <summary>A weight matrix as the graph sees it, in <paramref name="shape"/>:
    /// the fp16 constant itself, or, under <see cref="Int8Weights"/>, its int8
    /// blocks and scales dequantized back to that shape. <paramref name="rows"/>
    /// is the output extent — the blocks run along everything else, which is
    /// the input features of a linear layer and the input channels times the
    /// kernel of a convolution.</summary>
    public DmlExpression Weight(HostTensor tensor, int[] shape, int rows)
    {
        if (!Int8Weights || tensor.ElementCount < QuantizeFrom ||
            !BlockQuantizer.Fits(tensor.ElementCount, rows))
        {
            return Constant(tensor, shape);
        }

        (HostTensor quantized, HostTensor scales) = BlockQuantizer.Quantize(tensor, rows);

        // Dequantize wants the scales' rank to match, so both go in as
        // [1, 1, rows, columns] and the result is viewed as the asked shape.
        int[] quantizedShape = { 1, 1, rows, quantized.Shape[1] };
        int[] scaleShape = { 1, 1, rows, scales.Shape[1] };
        DmlExpression q = Graph.Input(HostDataType.Int8, quantizedShape, owned: true);
        DmlExpression s = Graph.Input(_dataType, scaleShape, owned: true);
        _weights[q] = quantized.Reshape(quantizedShape);
        _weights[s] = scales.ConvertTo(_dataType).Reshape(scaleShape);
        DmlExpression dequantized = DmlOps.Dequantize(q, s);
        uint[] sizes = shape.Select(extent => checked((uint)extent)).ToArray();
        return dequantized.Shape.SequenceEqual(sizes)
            ? dequantized
            : DmlOps.Reinterpret(dequantized, sizes);
    }

    /// <summary>Register an input whose data is supplied to <see cref="Run"/>,
    /// at the builder's precision unless <paramref name="dataType"/> says
    /// otherwise — a residual stream kept wider than the weights, say.</summary>
    public DmlExpression Placeholder(int[] shape, HostDataType? dataType = null)
    {
        DmlExpression expression = Graph.Input(dataType ?? _dataType, shape);
        _placeholders.Add(expression);
        return expression;
    }

    /// <summary>Compile the graph and upload every weight. The registered
    /// tensors are released here, so exactly one copy remains — on the GPU.
    /// <paramref name="streamWhen"/>, given the bytes the folded weights will
    /// take, says whether they are to live in system memory rather than video
    /// memory (<see cref="DmlCompiledModel.Streamed"/>).</summary>
    public ModelBuilder Compile(IReadOnlyList<DmlExpression> outputs,
        Func<ulong, bool>? streamWhen = null)
    {
        _operator = Graph.Compile(outputs);
        // Decided between compiling and initializing: the persistent size is
        // known by then, and a streamed graph's weights leave video memory
        // as part of initializing.
        if (streamWhen is not null)
        {
            _operator.Streamed = streamWhen(_operator.PersistentSize);
        }
        try
        {
            _operator.Initialize(_weights);
        }
        catch
        {
            // Out of memory while the weights go up is survivable — the
            // device stays in use — so what was allocated for this model goes
            // back here: the usual caller is `return model.Compile(...)`,
            // which has nothing to dispose when this throws.
            _operator.Dispose();
            _operator = null;
            _weights.Clear();
            throw;
        }
        _weights.Clear();
        return this;
    }

    /// <summary>Bind <paramref name="values"/> to the placeholders in order and
    /// execute the graph.</summary>
    public HostTensor[] Run(params HostTensor[] values)
    {
        if (_operator is null)
        {
            throw new InvalidOperationException("Compile() the model before running it");
        }
        if (values.Length != _placeholders.Count)
        {
            throw new ArgumentException(
                $"expected {_placeholders.Count} inputs, got {values.Length}");
        }
        var inputs = new Dictionary<DmlExpression, HostTensor>(values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            inputs[_placeholders[i]] = values[i];
        }
        return _operator.Dispatch(inputs);
    }

    /// <summary>Bind <paramref name="values"/>, already on the GPU, to the
    /// placeholders in order and execute the graph, leaving the outputs there.</summary>
    public DeviceTensor[] RunOnDevice(params DeviceTensor[] values)
    {
        if (_operator is null)
        {
            throw new InvalidOperationException("Compile() the model before running it");
        }
        if (values.Length != _placeholders.Count)
        {
            throw new ArgumentException(
                $"expected {_placeholders.Count} inputs, got {values.Length}");
        }
        var inputs = new Dictionary<DmlExpression, DeviceTensor>(values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            inputs[_placeholders[i]] = values[i];
        }
        return _operator.DispatchOnDevice(inputs);
    }

    /// <summary>The compiled graph, once <see cref="Compile"/> has run.</summary>
    public DmlCompiledModel Compiled =>
        _operator ?? throw new InvalidOperationException("Compile() the model first");

    /// <summary>Scratch bytes one dispatch needs — where the intermediates live.</summary>
    public ulong TemporarySize => _operator?.TemporarySize ?? 0;

    /// <summary>Video memory the folded weights occupy, once compiled.</summary>
    public ulong PersistentBytes => _operator?.PersistentBytes ?? 0;

    public void Dispose()
    {
        _operator?.Dispose();
        _weights.Clear();
    }
}
