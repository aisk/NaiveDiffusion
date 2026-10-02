using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Tensors;

namespace NaiveDiffusion.Models.Anima;

/// <summary>What Anima's transformer cross-attends to: one context sequence
/// per classifier-free-guidance branch, negative first, under the name
/// <see cref="AnimaDit"/> reads it by. Both branches are padded to the same
/// length, since both go through the same compiled graphs.</summary>
public sealed class AnimaConditioning : Conditioning
{
    public const string ContextName = "context";

    private readonly IReadOnlyDictionary<string, HostTensor>[] _branches;

    public AnimaConditioning(HostTensor[] contexts, int rows)
    {
        _branches = contexts.Select(context => (IReadOnlyDictionary<string, HostTensor>)
            new Dictionary<string, HostTensor> { [ContextName] = context }).ToArray();
        Rows = rows;
    }

    /// <summary>The context length the graphs are compiled for: 512, or a
    /// longer prompt's own length.</summary>
    public int Rows { get; }

    public override int Branches => _branches.Length;

    public override IReadOnlyDictionary<string, HostTensor> Branch(int index) => _branches[index];

    public override string Specialization => Rows.ToString();
}

/// <summary>Anima's prompt encoding: Qwen3 and the adapter over the prompt
/// and the negative prompt, on the CPU.</summary>
public sealed class AnimaConditioner : IConditioner
{
    public IPromptEncoder Open(GenerationOptions options) =>
        new Encoder(new AnimaTextEncoder(options.CheckpointPath,
            options.ComponentPath(AnimaFamily.TextEncoderComponent.Id)
            ?? throw new ArgumentException("Anima needs its text encoder file"),
            options.Loras));

    public bool TakesContextRows => true;

    /// <summary>One branch from the adapter's rows, padded with zeros to
    /// the context length as the encoder pads its own.</summary>
    public Conditioning FromContextRows(GenerationOptions options, float[] values)
    {
        if (values.Length == 0 || values.Length % LlmAdapter.Width != 0)
        {
            throw new ArgumentException(
                $"{values.Length} values is not a whole number of rows of {LlmAdapter.Width}");
        }
        int rows = Math.Max(AnimaPrompt.ContextLength, values.Length / LlmAdapter.Width);
        var context = new float[rows * LlmAdapter.Width];
        Array.Copy(values, context, values.Length);
        return new AnimaConditioning(new[] { HostTensor.FromFloats(context, 1, 1, rows, LlmAdapter.Width) }, rows);
    }

    private sealed class Encoder(AnimaTextEncoder encoder) : IPromptEncoder
    {
        public Conditioning Encode(GenerationOptions options, int branches)
        {
            // Past 512 tokens a prompt sets the context length; the other branch
            // is padded to match, so both attend against the same graphs.
            string[] prompts = branches > 1
                ? new[] { options.Negative, options.Prompt }
                : new[] { options.Prompt };
            int rows = prompts.Max(prompt => AnimaTextEncoder.ContextRows(encoder.Tokenize(prompt)));
            return new AnimaConditioning(prompts.Select(prompt =>
                HostTensor.FromFloats(encoder.Encode(prompt, rows), 1, 1, rows, LlmAdapter.Width)).ToArray(), rows);
        }

        public void Dispose() => encoder.Dispose();
    }
}
