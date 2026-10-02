using NaiveDiffusion.Tensors;

namespace NaiveDiffusion.Text;

/// <summary>One tower's parameters, split by how they are consumed.
///
/// A matrix — a linear layer's weight, an embedding table — is read one row at a
/// time and stays at the width the checkpoint stored it; widening a whole tower
/// up front costs 4.8 GiB of host memory for 1.5 GiB of weights. A vector —
/// every bias and layer-norm scale — is read whole, and all of them together
/// come to a couple of megabytes, so they are widened once here rather than
/// repeatedly at the point of use.</summary>
public sealed class TowerWeights
{
    private readonly Dictionary<string, HostTensor> _matrices = new();
    private readonly Dictionary<string, float[]> _vectors = new();

    public TowerWeights(Dictionary<string, HostTensor> tensors)
    {
        foreach ((string name, HostTensor tensor) in tensors)
        {
            if (tensor.Shape.Length >= 2)
            {
                _matrices[name] = tensor;
            }
            else
            {
                _vectors[name] = tensor.ToFloats();
            }
        }
    }

    public HostTensor Matrix(string name) => _matrices[name];

    public float[] Vector(string name) => _vectors[name];

    /// <summary>Take a matrix out and hand it over widened — for the one weight
    /// that is neither read by row nor small, the pooled projection.</summary>
    public float[] TakeWidened(string name)
    {
        float[] widened = _matrices[name].ToFloats();
        _matrices.Remove(name);
        return widened;
    }

    public void Clear()
    {
        _matrices.Clear();
        _vectors.Clear();
    }
}
