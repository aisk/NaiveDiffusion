using NaiveDiffusion.Tensors;
using Vortice.DirectML;

namespace NaiveDiffusion.Dml;

/// <summary>A DirectML graph under construction, in the mold of DirectMLX:
/// declare inputs, build expressions with <see cref="DmlOps"/>, then compile
/// the outputs into a <see cref="DmlCompiledModel"/>.</summary>
public sealed class DmlGraph
{
    internal sealed record InputSlot(DmlTensorDesc Desc, bool Owned);

    internal sealed record Node(OperatorDescription Description, DmlExpression?[] Inputs);

    private readonly List<InputSlot> _inputs = new();
    private readonly List<Node> _nodes = new();

    public DmlDevice Device { get; }

    public DmlGraph(DmlDevice device) => Device = device;

    /// <summary>Add a packed graph input. <paramref name="owned"/> marks it
    /// DML_TENSOR_FLAG_OWNED_BY_DML: its data is handed over once, at
    /// initialization, and lives on the GPU from then on.</summary>
    public DmlExpression Input(TensorDataType dataType, uint[] sizes, bool owned = false)
    {
        var desc = DmlTensorDesc.Packed(dataType, sizes,
            owned ? TensorFlags.OwnedByDml : TensorFlags.None);
        var expression = new DmlExpression(this, _inputs.Count, -1, 0, desc);
        _inputs.Add(new InputSlot(desc, owned));
        return expression;
    }

    public DmlExpression Input(HostDataType dataType, int[] sizes, bool owned = false) =>
        Input(DmlTensorDesc.ToDataType(dataType),
            sizes.Select(extent => checked((uint)extent)).ToArray(), owned);

    /// <summary>Add one operator node. <paramref name="inputs"/> lists the
    /// operator's input tensors in descriptor order, with null holding the place
    /// of every optional tensor that is absent.</summary>
    internal DmlExpression Emit(OperatorDescription description, DmlExpression?[] inputs,
        DmlTensorDesc outputDesc)
    {
        foreach (DmlExpression? input in inputs)
        {
            if (input is not null && !ReferenceEquals(input.Graph, this))
            {
                throw new InvalidOperationException("expression belongs to a different graph");
            }
        }
        var expression = new DmlExpression(this, -1, _nodes.Count, 0, outputDesc);
        _nodes.Add(new Node(description, inputs));
        return expression;
    }

    /// <summary>Compile the graph. The outputs are fixed here, in this order.
    /// The model snapshots what it needs, so the graph can be dropped after.</summary>
    public DmlCompiledModel Compile(IReadOnlyList<DmlExpression> outputs,
        ExecutionFlags flags = ExecutionFlags.None)
    {
        var operators = new IDMLOperator[_nodes.Count];
        var nodes = new OperatorGraphNodeDescription[_nodes.Count];
        var inputEdges = new List<InputGraphEdgeDescription>();
        var intermediateEdges = new List<IntermediateGraphEdgeDescription>();

        try
        {
            for (int nodeIndex = 0; nodeIndex < _nodes.Count; nodeIndex++)
            {
                Node node = _nodes[nodeIndex];
                operators[nodeIndex] = Device.Device.CreateOperator(node.Description);
                nodes[nodeIndex] = new OperatorGraphNodeDescription { Operator = operators[nodeIndex] };

                for (int port = 0; port < node.Inputs.Length; port++)
                {
                    DmlExpression? source = node.Inputs[port];
                    if (source is null)
                    {
                        continue;
                    }
                    if (source.InputIndex >= 0)
                    {
                        inputEdges.Add(new InputGraphEdgeDescription
                        {
                            GraphInputIndex = source.InputIndex,
                            ToNodeIndex = nodeIndex,
                            ToNodeInputIndex = port,
                        });
                    }
                    else
                    {
                        intermediateEdges.Add(new IntermediateGraphEdgeDescription
                        {
                            FromNodeIndex = source.NodeIndex,
                            FromNodeOutputIndex = source.NodeOutputIndex,
                            ToNodeIndex = nodeIndex,
                            ToNodeInputIndex = port,
                        });
                    }
                }
            }

            var outputEdges = new OutputGraphEdgeDescription[outputs.Count];
            var outputDescs = new DmlTensorDesc[outputs.Count];
            for (int i = 0; i < outputs.Count; i++)
            {
                DmlExpression output = outputs[i];
                if (output.NodeIndex < 0)
                {
                    throw new InvalidOperationException(
                        "a graph input cannot be an output; run it through an identity first");
                }
                outputEdges[i] = new OutputGraphEdgeDescription
                {
                    FromNodeIndex = output.NodeIndex,
                    FromNodeOutputIndex = output.NodeOutputIndex,
                    GraphOutputIndex = i,
                };
                outputDescs[i] = output.Desc;
            }

            // DirectML compiles and initializes a graph with an input no node
            // reads, and then takes the device down on its first dispatch —
            // DXGI_ERROR_DEVICE_REMOVED with nothing pointing here. Said now
            // instead, with the slot. Only of the inputs bound per dispatch,
            // which is the case that was met; a weight nothing reads is
            // wasted memory, not a known crash.
            var read = new bool[_inputs.Count];
            foreach (InputGraphEdgeDescription edge in inputEdges)
            {
                read[edge.GraphInputIndex] = true;
            }
            for (int slot = 0; slot < read.Length; slot++)
            {
                if (!read[slot] && !_inputs[slot].Owned)
                {
                    throw new InvalidOperationException(
                        $"graph input {slot} [{string.Join(", ", _inputs[slot].Desc.Sizes)}] is read by no node; " +
                        "DirectML removes the device when such a graph is dispatched");
                }
            }

            var graphDescription = new GraphDescription
            {
                InputCount = _inputs.Count,
                OutputCount = outputs.Count,
                Nodes = nodes,
                InputEdges = inputEdges.ToArray(),
                OutputEdges = outputEdges,
                IntermediateEdges = intermediateEdges.ToArray(),
            };

            IDMLCompiledOperator compiled = Device.Device.CompileGraph(graphDescription, flags);
            return new DmlCompiledModel(Device, compiled, _inputs.ToArray(), outputDescs);
        }
        finally
        {
            foreach (IDMLOperator? op in operators)
            {
                op?.Dispose();
            }
        }
    }
}
