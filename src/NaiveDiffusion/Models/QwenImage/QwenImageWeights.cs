using NaiveDiffusion.Tensors;
using NaiveDiffusion.Vae;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Models.QwenImage;

/// <summary>Where Qwen-Image 2.1's tensors come from: three files — the
/// transformer, the Qwen3-VL text encoder, and the VAE — mapped and owned
/// by <see cref="SplitWeights"/>, read a model at a time.</summary>
public sealed class QwenImageWeights : SplitWeights
{
    public QwenImageWeights(string checkpointPath, string? textEncoderPath = null, string? vaePath = null)
        : base(checkpointPath, textEncoderPath, vaePath)
    {
    }

    protected override string? CheckpointPrefix(SafetensorsFile file) => QwenImageCheckpoint.DitPrefix(file);

    protected override string CheckpointKind => "a Qwen-Image 2.1 checkpoint";

    /// <summary>The transformer's tensors under their bare names at half
    /// precision, converted lazily as the graphs read them so no more than
    /// one graph's worth of copies exists.</summary>
    public LazyWeights LoadDit() =>
        new(Checkpoint.ReadPrefix(Prefix), null, HostDataType.Float16);

    /// <summary>The language model's tensors, as stored; the vision tower
    /// and the output head, which no prompt goes through, are left in the
    /// file.</summary>
    public Dictionary<string, HostTensor> LoadTextEncoder()
    {
        if (!QwenImageCheckpoint.IsTextEncoder(TextEncoderFile))
        {
            throw new InvalidDataException($"{TextEncoderPath}: not the Qwen3-VL-8B text encoder");
        }
        Dictionary<string, HostTensor> tensors = TextEncoderFile.ReadPrefix("model.");
        foreach (string name in tensors.Keys.Where(name =>
                     name.StartsWith("visual.", StringComparison.Ordinal)).ToArray())
        {
            tensors.Remove(name);
        }
        return tensors.ToDictionary(pair => "model." + pair.Key, pair => pair.Value);
    }

    /// <summary>The VAE's tensors as the single-frame 2-D graph wants them,
    /// at float32 — see <see cref="WanVae.SingleFrame"/>.</summary>
    public Dictionary<string, HostTensor> LoadVae()
    {
        if (!QwenImageCheckpoint.IsVae(VaeFile))
        {
            throw new InvalidDataException($"{VaePath}: not the Qwen-Image 2.1 VAE");
        }
        return WanVae.SingleFrame(VaeFile.ReadPrefix(""));
    }
}
