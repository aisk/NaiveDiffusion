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

/// <summary>One text for a family's text encoder, and the length its output
/// is brought to, in the family's own unit — SDXL's chunks, Anima's context
/// rows — where the prompts of a run have to come out the same shape; 0
/// where the family pads nothing at this point. The two together are what
/// the encoder's output depends on besides the encoder itself, which is
/// what lets a <see cref="PromptCache"/> keep it.</summary>
public readonly record struct PromptRequest(string Text, int Length);

/// <summary>Prompt in, conditioning out — the text side of one model family,
/// in steps so that the pipeline can say which one it is in — reading the
/// text encoder's weights off the files takes seconds, Qwen3-VL-8B being
/// sixteen gigabytes, and is not the same wait as running it — and can skip
/// the encoder altogether for prompts it has encoded before.</summary>
public interface IConditioner
{
    /// <summary>The texts a run encodes, the negative prompt's first when
    /// there are two branches and only the prompt's when there is one. From
    /// the tokenizers alone; no weight is read.</summary>
    IReadOnlyList<PromptRequest> Requests(GenerationOptions options, int branches);

    /// <summary>Read the text encoder's weights for this run; nothing is
    /// encoded yet. Disposed by the caller once the prompts are.</summary>
    IPromptEncoder Open(GenerationOptions options);

    /// <summary>The conditioning out of the encoder's output for each of
    /// <see cref="Requests"/>, in the same order, with whatever else the
    /// denoiser reads that follows from the run rather than the text — the
    /// image size, the compute precision. Reads the arrays and keeps none
    /// of them.</summary>
    Conditioning Condition(GenerationOptions options, IReadOnlyList<float[][]> encoded);

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

/// <summary>A loaded text encoder: one text in, what the encoder makes of
/// it out.</summary>
public interface IPromptEncoder : IDisposable
{
    /// <summary>The encoder's output for one text, as the arrays the
    /// family's <see cref="IConditioner.Condition"/> reads: a function of
    /// the request and of the encoder, and of nothing else.</summary>
    float[][] Encode(PromptRequest request);
}

public static class Conditioners
{
    /// <summary>The prompts a run encodes, the negative first when there
    /// are two branches.</summary>
    public static string[] Prompts(GenerationOptions options, int branches) => branches > 1
        ? new[] { options.Negative, options.Prompt }
        : new[] { options.Prompt };

    /// <summary>Open, encode and release in one call, for a caller with no
    /// progress to report between the two.</summary>
    public static Conditioning Encode(this IConditioner conditioner, GenerationOptions options,
        int branches)
    {
        using IPromptEncoder encoder = conditioner.Open(options);
        return conditioner.Condition(options, conditioner.Requests(options, branches)
            .Select(encoder.Encode).ToArray());
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
