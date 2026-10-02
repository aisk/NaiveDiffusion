using NaiveDiffusion.Dml;
using NaiveDiffusion.Graph;
using NaiveDiffusion.Tensors;

namespace NaiveDiffusion.Models.Sdxl;

/// <summary>The UNet's own compound layers, read under diffusers' weight
/// names: attention with its four projections and the gated feed-forward.</summary>
public static class UNetLayers
{
    /// <summary>Attention under diffusers' weight names, where only the output
    /// projection carries a bias. Passing <paramref name="context"/> makes it
    /// cross-attention: keys and values come from the text embeddings.</summary>
    public static DmlExpression DiffusersAttention(ModelBuilder model, DmlExpression x,
        IReadOnlyDictionary<string, HostTensor> parameters, string prefix, int heads,
        DmlExpression? context = null)
    {
        DmlExpression source = context ?? x;
        DmlExpression query = Layers.Linear(model, x, parameters[$"{prefix}.to_q.weight"]);
        DmlExpression key = Layers.Linear(model, source, parameters[$"{prefix}.to_k.weight"]);
        DmlExpression value = Layers.Linear(model, source, parameters[$"{prefix}.to_v.weight"]);

        DmlExpression attended = Layers.Attend(query, key, value, heads);
        return Layers.Linear(model, attended, parameters[$"{prefix}.to_out.0.weight"],
            parameters[$"{prefix}.to_out.0.bias"]);
    }

    /// <summary>The UNet's feed-forward: project to twice the width, gate one
    /// half by the exact GELU of the other.</summary>
    public static DmlExpression Geglu(ModelBuilder model, DmlExpression x,
        IReadOnlyDictionary<string, HostTensor> parameters, string prefix)
    {
        DmlExpression projected = Layers.Linear(model, x,
            parameters[$"{prefix}.net.0.proj.weight"], parameters[$"{prefix}.net.0.proj.bias"]);
        uint[] s = projected.Shape;
        (uint n, uint tokens, uint doubled) = (s[0], s[2], s[3]);
        uint inner = doubled / 2;

        DmlExpression Half(uint offset) => DmlOps.Slice(projected,
            offsets: new[] { 0, 0, 0, (int)offset },
            sizes: new[] { (int)n, 1, (int)tokens, (int)inner },
            strides: new[] { 1, 1, 1, 1 });

        DmlExpression gated = Half(0) * DmlOps.ActivationGelu(Half(inner));
        return Layers.Linear(model, gated, parameters[$"{prefix}.net.2.weight"],
            parameters[$"{prefix}.net.2.bias"]);
    }
}
