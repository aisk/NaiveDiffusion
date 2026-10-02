using NaiveDiffusion.Tensors;

namespace NaiveDiffusion.Pipeline;

/// <summary>Everything the prompts turn into, as the denoiser wants it: one
/// set of named tensors per guidance branch, negative first. What the names
/// are is between the family's conditioner and its denoiser; the pipeline
/// only carries them across.</summary>
public abstract class Conditioning
{
    /// <summary>How many branches there are: what the pipeline asked the
    /// encoder for, the family's <see cref="GuidanceStrategy.Branches"/>
    /// at the run's guidance scale, and checked against it.</summary>
    public abstract int Branches { get; }

    /// <summary>The tensors for one branch, the prompt's own last.</summary>
    public abstract IReadOnlyDictionary<string, HostTensor> Branch(int index);

    /// <summary>What a compiled denoiser is specialized for beyond the image
    /// size — for SDXL the context width, which grows with the prompt — so a
    /// cached one is only reused where it fits. Part of the cache key.</summary>
    public abstract string Specialization { get; }
}

/// <summary>Prompt in, conditioning out — the text side of one model family,
/// in two steps so that the pipeline can say which one it is in: reading the
/// text encoder's weights off the files takes seconds — Qwen3-VL-8B is
/// sixteen gigabytes — and is not the same wait as running it.</summary>
public interface IConditioner
{
    /// <summary>Read the text encoder's weights for this run; nothing is
    /// encoded yet. Disposed by the caller once the prompts are.</summary>
    IPromptEncoder Open(GenerationOptions options);

    /// <summary>Whether the denoiser's conditioning is a plain sequence of
    /// context rows that can be handed over as numbers, with no text
    /// encoder run at all — a reference implementation's rows, so one
    /// forward pass of the diffusion model is compared on its own. A family
    /// whose conditioning is more than one such sequence says no.</summary>
    bool TakesContextRows { get; }

    /// <summary>One branch of conditioning, the prompt's, from raw context
    /// rows laid end to end at the width the family's denoiser projects;
    /// the row count follows from the length. Refused with a sentence when
    /// the length is not whole rows, and not offered at all when
    /// <see cref="TakesContextRows"/> is false.</summary>
    Conditioning FromContextRows(GenerationOptions options, float[] values);
}

/// <summary>A loaded text encoder: the run's prompts in, the conditioning
/// the denoiser reads out.</summary>
public interface IPromptEncoder : IDisposable
{
    /// <summary>The conditioning for <paramref name="branches"/> branches:
    /// the negative prompt's first when there are two, the prompt's own
    /// last, and only the prompt's when there is one.</summary>
    Conditioning Encode(GenerationOptions options, int branches);
}

public static class Conditioners
{
    /// <summary>Open, encode and release in one call, for a caller with no
    /// progress to report between the two.</summary>
    public static Conditioning Encode(this IConditioner conditioner, GenerationOptions options,
        int branches)
    {
        using IPromptEncoder encoder = conditioner.Open(options);
        return encoder.Encode(options, branches);
    }

    /// <summary>The branch count a run wants is 1 or 2, and what the
    /// encoder made has to have exactly that many: the sampler reads them
    /// by index.</summary>
    public static Conditioning CheckedFor(this Conditioning conditioning, int branches)
    {
        if (branches is not (1 or 2))
        {
            throw new ArgumentOutOfRangeException(nameof(branches), branches, "one or two branches");
        }
        if (conditioning.Branches != branches)
        {
            throw new InvalidOperationException(
                $"the conditioner made {conditioning.Branches} branches where the guidance runs {branches}");
        }
        return conditioning;
    }
}
