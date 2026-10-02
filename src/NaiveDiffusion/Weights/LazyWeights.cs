using System.Collections;
using NaiveDiffusion.Tensors;

namespace NaiveDiffusion.Weights;

/// <summary>A model's weights, finished only as they are read: a tensor comes
/// out of the indexer with its LoRA deltas folded in and at the width the
/// graphs run at, made on the spot and not kept.
///
/// The UNet's builder reads each weight once, hands it to the graph being
/// built, and that graph uploads and forgets it when it is cut — so a copy
/// made here lives for one graph's worth of building. Merging everything up
/// front would hold every copy until the last graph was done: 4.3 GiB of
/// them at once on the UNet, on top of the mapped checkpoint. The same goes
/// for a checkpoint stored as bfloat16, which is converted weight by weight
/// rather than all at once.</summary>
public sealed class LazyWeights : IReadOnlyDictionary<string, HostTensor>, IDisposable
{
    private readonly Dictionary<string, HostTensor> _stored;
    private readonly LoraPatchSet? _patches;
    private readonly HostDataType _dataType;

    /// <param name="stored">The tensors as the checkpoint holds them.</param>
    /// <param name="patches">The LoRA deltas to fold in, or null for none;
    /// owned from here on and closed with this.</param>
    /// <param name="dataType">The width every tensor comes out at.</param>
    public LazyWeights(Dictionary<string, HostTensor> stored, LoraPatchSet? patches,
        HostDataType dataType)
    {
        _stored = stored;
        _patches = patches;
        _dataType = dataType;
    }

    public HostTensor this[string key] => TryGetValue(key, out HostTensor? value)
        ? value
        : throw new KeyNotFoundException(key);

    public bool TryGetValue(string key, out HostTensor value)
    {
        if (!_stored.TryGetValue(key, out HostTensor? stored))
        {
            value = null!;
            return false;
        }
        value = Unnarrowed(key).ConvertTo(_dataType);
        return true;
    }

    /// <summary>The width every tensor comes out at.</summary>
    public HostDataType DataType => _dataType;

    /// <summary>A tensor as it is before it is narrowed to
    /// <see cref="DataType"/> — the checkpoint's, with its LoRA deltas folded
    /// in — for a look at what the narrowing costs.</summary>
    public HostTensor Unnarrowed(string key)
    {
        HostTensor stored = _stored[key];
        return _patches is not null && _patches.Touches(key) ? _patches.Apply(key, stored) : stored;
    }

    public bool ContainsKey(string key) => _stored.ContainsKey(key);

    public IEnumerable<string> Keys => _stored.Keys;

    /// <summary>Every tensor finished, one after another — a full pass over the
    /// weights, so only something that really needs all of them at once
    /// should ask.</summary>
    public IEnumerable<HostTensor> Values => _stored.Keys.Select(key => this[key]);

    public int Count => _stored.Count;

    public IEnumerator<KeyValuePair<string, HostTensor>> GetEnumerator() =>
        _stored.Keys.Select(key => new KeyValuePair<string, HostTensor>(key, this[key])).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>Closes the LoRA files and forgets the checkpoint's tensors,
    /// which are windows onto its mapping and must not outlive it.</summary>
    public void Dispose()
    {
        _patches?.Dispose();
        _stored.Clear();
    }
}
