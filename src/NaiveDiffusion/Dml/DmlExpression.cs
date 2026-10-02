namespace NaiveDiffusion.Dml;

/// <summary>One tensor value inside a graph under construction: either a graph
/// input or the output of an operator node, seen through a tensor description.
/// Reinterpreting returns a new expression over the same source, so identity
/// (which input slot a binding matches) is the source, not the view.</summary>
public sealed class DmlExpression
{
    public DmlGraph Graph { get; }

    /// <summary>The graph input index, or -1 when this is a node output.</summary>
    public int InputIndex { get; }

    /// <summary>The operator node index, or -1 when this is a graph input.</summary>
    public int NodeIndex { get; }

    public int NodeOutputIndex { get; }

    public DmlTensorDesc Desc { get; }

    internal DmlExpression(DmlGraph graph, int inputIndex, int nodeIndex,
        int nodeOutputIndex, DmlTensorDesc desc)
    {
        Graph = graph;
        InputIndex = inputIndex;
        NodeIndex = nodeIndex;
        NodeOutputIndex = nodeOutputIndex;
        Desc = desc;
    }

    public uint[] Shape => Desc.Sizes;

    /// <summary>The same bytes through a different description.</summary>
    internal DmlExpression WithDesc(DmlTensorDesc desc) =>
        new(Graph, InputIndex, NodeIndex, NodeOutputIndex, desc);

    public static DmlExpression operator +(DmlExpression a, DmlExpression b) => DmlOps.Add(a, b);
    public static DmlExpression operator -(DmlExpression a, DmlExpression b) => DmlOps.Subtract(a, b);
    public static DmlExpression operator *(DmlExpression a, DmlExpression b) => DmlOps.Multiply(a, b);
    public static DmlExpression operator /(DmlExpression a, DmlExpression b) => DmlOps.Divide(a, b);

    // A float operand rides on the scale-bias of an elementwise identity,
    // exactly as DirectMLX writes it.
    public static DmlExpression operator +(DmlExpression a, float b) => DmlOps.Identity(a, 1.0f, b);
    public static DmlExpression operator -(DmlExpression a, float b) => DmlOps.Identity(a, 1.0f, -b);
    public static DmlExpression operator *(DmlExpression a, float b) => DmlOps.Identity(a, b, 0.0f);
    public static DmlExpression operator /(DmlExpression a, float b) => DmlOps.Identity(a, 1.0f / b, 0.0f);
    public static DmlExpression operator +(float a, DmlExpression b) => DmlOps.Identity(b, 1.0f, a);
    public static DmlExpression operator -(float a, DmlExpression b) => DmlOps.Identity(b, -1.0f, a);
    public static DmlExpression operator *(float a, DmlExpression b) => DmlOps.Identity(b, a, 0.0f);
    public static DmlExpression operator -(DmlExpression a) => DmlOps.Identity(a, -1.0f, 0.0f);

    public override string ToString() => $"<expr {Desc}>";
}
