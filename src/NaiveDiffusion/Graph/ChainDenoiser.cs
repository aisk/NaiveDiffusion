using NaiveDiffusion.Dml;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Tensors;

namespace NaiveDiffusion.Graph;

/// <summary>A denoiser that is a <see cref="GraphChain"/>: what the two
/// transformer families share around their graphs — the chain's memory
/// figures reported as the pipeline's, the build that releases the chain
/// when it fails part way, and the disposal. The subclass decides its
/// shape, builds the links and turns a latent into the chain's inputs.</summary>
public abstract class ChainDenoiser : IDenoiser
{
    private GraphChain? _chain;

    /// <summary>The chain, once <see cref="Build"/> has made it.</summary>
    protected GraphChain Chain =>
        _chain ?? throw new InvalidOperationException("the chain has not been built");

    /// <summary>Make the chain with the weights stored at
    /// <paramref name="stored"/> and build its links; a failure inside
    /// <paramref name="build"/> — out of memory, a shape that does not fit
    /// — releases what was made so the device goes on being usable.</summary>
    protected void Build(DmlDevice device, HostDataType stored, ulong? residentBudget,
        bool int8Weights, Action<GraphChain> build)
    {
        var chain = new GraphChain(device, stored, residentBudget, int8Weights);
        try
        {
            build(chain);
        }
        catch
        {
            chain.Dispose();
            throw;
        }
        _chain = chain;
    }

    public abstract HostTensor Predict(HostTensor latent, double timestep, double sigma,
        IReadOnlyDictionary<string, HostTensor> conditioning);

    public ulong PersistentBytes => Chain.PersistentBytes;

    public ulong StreamedBytes => Chain.StreamedBytes;

    public ulong ResidentBytes => Chain.ResidentBytes;

    public ulong TemporaryBytes => Chain.TemporaryBytes;

    public int GraphCount => Chain.LinkCount;

    public void Dispose() => _chain?.Dispose();
}
