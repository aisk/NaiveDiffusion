using NaiveDiffusion.Tensors;
using NaiveDiffusion.Vae;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Models.Anima;

/// <summary>Where Anima's tensors come from: three files — the transformer
/// with its text adapter, the Qwen3 text encoder, and the Wan-family VAE —
/// mapped and owned by <see cref="SplitWeights"/>, read a model at a time.</summary>
public sealed class AnimaWeights : SplitWeights
{
    /// <param name="checkpointPath">The transformer file, under either prefix.</param>
    /// <param name="textEncoderPath">The Qwen3 file.</param>
    /// <param name="vaePath">The VAE file.</param>
    public AnimaWeights(string checkpointPath, string? textEncoderPath = null, string? vaePath = null)
        : base(checkpointPath, textEncoderPath, vaePath)
    {
    }

    protected override string? CheckpointPrefix(SafetensorsFile file) => AnimaCheckpoint.DitPrefix(file);

    protected override string CheckpointKind => "an Anima checkpoint";

    /// <summary>The transformer's tensors under their bare names — the
    /// adapter left out — at half precision, converted lazily as the graphs
    /// read them so no more than one graph's worth of copies exists, with
    /// the <paramref name="loras"/>' deltas folded in the same way. The files
    /// are matched against the whole transformer, adapter included, so one
    /// that fits nothing is refused here, before the build starts.</summary>
    public LazyWeights LoadDit(IReadOnlyList<LoraSpec>? loras = null)
    {
        Dictionary<string, HostTensor> raw = Checkpoint.ReadPrefix(Prefix);
        LoraPatchSet? patches = loras is { Count: > 0 }
            ? LoraPatchSet.Prepare(raw, AnimaLoras.Layout, AnimaLoras.Dit, loras)
            : null;
        foreach (string name in raw.Keys.Where(name =>
                     name.StartsWith(AnimaCheckpoint.AdapterPrefix, StringComparison.Ordinal)).ToArray())
        {
            raw.Remove(name);
        }
        return new LazyWeights(raw, patches, HostDataType.Float16);
    }

    /// <summary>The text adapter's tensors under their bare names, at the
    /// width the file stored them — or, for a weight a LoRA changes, at half
    /// precision with the delta folded in; the CPU forward widens rows as it
    /// reads. The adapter is part of the transformer to a LoRA, which names
    /// its layers <c>llm_adapter.…</c>, so the files are matched against the
    /// whole of it, as <see cref="LoadDit"/> matches them.</summary>
    public Dictionary<string, HostTensor> LoadAdapter(IReadOnlyList<LoraSpec>? loras = null)
    {
        Dictionary<string, HostTensor> adapter =
            Checkpoint.ReadPrefix(Prefix + AnimaCheckpoint.AdapterPrefix);
        if (loras is not { Count: > 0 })
        {
            return adapter;
        }
        using LoraPatchSet patches = LoraPatchSet.Prepare(Checkpoint.ReadPrefix(Prefix),
            AnimaLoras.Layout, AnimaLoras.Dit, loras);
        foreach (string name in adapter.Keys.ToArray())
        {
            adapter[name] = patches.Apply(AnimaCheckpoint.AdapterPrefix + name, adapter[name]);
        }
        return adapter;
    }

    /// <summary>The Qwen3 text encoder's tensors, as stored, but for the
    /// weights the <paramref name="loras"/> change, which come at half
    /// precision with the delta folded in.</summary>
    public Dictionary<string, HostTensor> LoadTextEncoder(IReadOnlyList<LoraSpec>? loras = null)
    {
        if (!AnimaCheckpoint.IsTextEncoder(TextEncoderFile))
        {
            throw new InvalidDataException($"{TextEncoderPath}: not the Qwen3-0.6B text encoder");
        }
        Dictionary<string, HostTensor> tensors = TextEncoderFile.ReadPrefix("");
        if (loras is { Count: > 0 })
        {
            LoraMerge.Apply(tensors, AnimaLoras.Layout, AnimaLoras.TextEncoder, loras,
                AnimaLoras.TextEncoderAliases);
        }
        return tensors;
    }

    /// <summary>The VAE's tensors as the single-frame 2-D graph wants them,
    /// at float32 — see <see cref="WanVae.SingleFrame"/>.</summary>
    public Dictionary<string, HostTensor> LoadVae()
    {
        if (!AnimaCheckpoint.IsVae(VaeFile))
        {
            throw new InvalidDataException($"{VaePath}: not the Wan 2.1 VAE");
        }
        return WanVae.SingleFrame(VaeFile.ReadPrefix(""));
    }
}
