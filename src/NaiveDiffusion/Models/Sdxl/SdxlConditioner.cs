using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Tensors;
using NaiveDiffusion.Text.Clip;

namespace NaiveDiffusion.Models.Sdxl;

/// <summary>What SDXL's UNet cross-attends to and adds in: one context
/// sequence and one add-vector per classifier-free-guidance branch, negative
/// first, under the names <see cref="UNetModel"/> reads them by.</summary>
public sealed class SdxlConditioning : Conditioning
{
    public const string ContextName = "context";
    public const string AddName = "add";

    private readonly IReadOnlyDictionary<string, HostTensor>[] _branches;

    /// <param name="tokens">The context sequence's length, which every
    /// branch shares: the UNet is compiled for it.</param>
    public SdxlConditioning(HostTensor[] contexts, HostTensor[] addInputs, int tokens)
    {
        _branches = new IReadOnlyDictionary<string, HostTensor>[contexts.Length];
        for (int i = 0; i < contexts.Length; i++)
        {
            _branches[i] = new Dictionary<string, HostTensor>
            {
                [ContextName] = contexts[i],
                [AddName] = addInputs[i],
            };
        }
        Tokens = tokens;
    }

    public int Tokens { get; }

    public override int Branches => _branches.Length;

    public override IReadOnlyDictionary<string, HostTensor> Branch(int index) => _branches[index];

    public override string Specialization => Tokens.ToString();
}

/// <summary>SDXL's prompt encoding: both CLIP towers over the prompt and the
/// negative prompt, on the CPU, plus the resolution conditioning SDXL adds
/// alongside the pooled vector.</summary>
public sealed class SdxlConditioner : IConditioner
{
    // Both towers run on the CPU; nothing in this stage reaches the GPU.
    public IPromptEncoder Open(GenerationOptions options) =>
        new Encoder(new SdxlTextEncoders(options.CheckpointPath, options.Loras, options.ClipSkip));

    /// <summary>The UNet reads a context and a pooled add-vector per
    /// branch, not one sequence; there is no file of rows to stand in.</summary>
    public bool TakesContextRows => false;

    public Conditioning FromContextRows(GenerationOptions options, float[] values) =>
        throw new NotSupportedException("SDXL's conditioning is a context and an add-vector, not rows alone");

    private sealed class Encoder(SdxlTextEncoders encoders) : IPromptEncoder
    {
        public Conditioning Encode(GenerationOptions options, int branches)
        {
            // A prompt over 75 tokens is encoded in several chunks laid end to end.
            // Both prompts cross-attend against the same UNet, so the shorter one is
            // padded out with empty chunks to the longer's length.
            string[] prompts = branches > 1
                ? new[] { options.Negative, options.Prompt }
                : new[] { options.Prompt };
            int chunks = prompts.Max(encoders.ChunkCount);
            int tokens = chunks * ClipTextEncoder.MaxTokens;

            // SDXL conditions on the resolution it is pretending to have been
            // cropped from as well as on the prompt.
            var resolution = (options.Height, options.Width);
            var contexts = new HostTensor[prompts.Length];
            var addInputs = new HostTensor[prompts.Length];
            for (int i = 0; i < prompts.Length; i++)
            {
                (float[] embeds, float[] pooled) = encoders.Encode(prompts[i], chunks);
                contexts[i] = HostTensor.FromFloats(embeds, 1, 1, tokens, UNetModel.ContextWidth);
                addInputs[i] = UNetModel.AddConditioning(pooled, resolution, (0, 0), resolution);
            }
            return new SdxlConditioning(contexts, addInputs, tokens);
        }

        public void Dispose() => encoders.Dispose();
    }
}
