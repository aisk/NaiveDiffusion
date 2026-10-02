using NaiveDiffusion.Models;
using NaiveDiffusion.Models.Anima;
using NaiveDiffusion.Models.QwenImage;
using NaiveDiffusion.Models.Sdxl;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Cli;

/// <summary>The text side over a prompt, reported as statistics for
/// eyeballing against the reference: SDXL's two CLIP towers, Anima's
/// Qwen3 through its adapter, or Qwen-Image's Qwen3-VL over its chat
/// template. A diagnostic, so it reaches past the family seam on purpose:
/// the tensors it dumps are the towers' own, which the conditioning the
/// pipeline sees does not expose.</summary>
internal static class EncodeCommand
{
    public static readonly string[] ValueOptions = new[]
    {
        "--checkpoint", "--dump", "--dump-hidden", "--clip-skip",
    }.Concat(CommandLine.ComponentOptions()).ToArray();

    // The text side runs on the CPU, so there is no device to open.
    public static int Run(CommandLine line)
    {
        string prompt = line.Argument ?? "an astronaut riding a horse on mars";
        IModelFamily family = line.Family(
            CheckpointInspector.CheckpointParts.TextEncoder
            | CheckpointInspector.CheckpointParts.TextEncoder2);
        return family switch
        {
            AnimaFamily => Anima(line, prompt),
            QwenImageFamily => QwenImage(line, prompt),
            SdxlFamily => Sdxl(line, prompt),
            _ => CommandLine.Fail<int>($"encode does not know {family.Name}'s text side"),
        };
    }

    private static int Sdxl(CommandLine line, string prompt)
    {
        line.Components(SdxlFamily.Instance);
        using var encoders = new SdxlTextEncoders(line.Checkpoint(), clipSkip: line.ClipSkip(SdxlFamily.Instance));
        float[] embeds, pooled;
        try
        {
            (embeds, pooled) = encoders.Encode(prompt);
        }
        catch (InvalidDataException exception)
        {
            // A checkpoint whose weights cannot give an answer at this clip
            // skip: the sentence is the whole story, no stack wanted.
            return CommandLine.Fail<int>(exception.Message);
        }

        // A prompt over 75 tokens encodes to more than one 77-token chunk.
        Console.WriteLine($"embeds [{embeds.Length / 2048}, 2048]: " +
                          $"mean {embeds.Average():+0.0000;-0.0000} " +
                          $"std {Statistics.Std(embeds):0.0000} first {embeds[0]:0.0000}");
        Console.WriteLine($"pooled [{pooled.Length}]: mean {pooled.Average():+0.0000;-0.0000} " +
                          $"std {Statistics.Std(pooled):0.0000} first {pooled[0]:0.0000}");

        // --dump writes both outputs raw, embeds then pooled, so two builds can
        // be compared bit for bit rather than to four decimals.
        if (line.Value("--dump") is string dump)
        {
            Dumps.Write(dump, embeds.Concat(pooled).ToArray());
        }
        return 0;
    }

    /// <summary>Anima's text side: the tokens both ways, the Qwen3 hidden
    /// states (--dump-hidden) and the adapter's padded context (--dump), the
    /// same tensors the ComfyUI reference script writes.</summary>
    private static int Anima(CommandLine line, string prompt)
    {
        Dictionary<string, string> components = line.Components(AnimaFamily.Instance,
            PipelineParts.Conditioner);
        line.ClipSkip(AnimaFamily.Instance);
        using var encoder = new AnimaTextEncoder(line.Checkpoint(),
            components[AnimaFamily.TextEncoderComponent.Id]);

        AnimaTokens tokens = encoder.Tokenize(prompt);
        Console.WriteLine($"qwen ids [{tokens.Qwen.Length}]: {string.Join(" ", tokens.Qwen)}");
        Console.WriteLine($"t5 ids [{tokens.T5.Length}]: {string.Join(" ", tokens.T5)}");
        float[] hidden = encoder.HiddenStates(tokens);
        Console.WriteLine($"qwen hidden [{tokens.Qwen.Length}, {LlmAdapter.Width}]: " +
                          $"mean {hidden.Average():+0.0000;-0.0000} " +
                          $"std {Statistics.Std(hidden):0.0000} first {hidden[0]:0.0000}");
        int rows = AnimaTextEncoder.ContextRows(tokens);
        float[] output = encoder.AdapterOutput(tokens, hidden);
        var context = new float[rows * LlmAdapter.Width];
        Array.Copy(output, context, output.Length);
        Console.WriteLine($"context [{rows}, {LlmAdapter.Width}]: " +
                          $"mean {context.Average():+0.00000;-0.00000} " +
                          $"std {Statistics.Std(context):0.00000} first {context[0]:0.0000}");

        if (line.Value("--dump-hidden") is string dumpHidden)
        {
            Dumps.Write(dumpHidden, hidden);
        }
        if (line.Value("--dump") is string dump)
        {
            Dumps.Write(dump, context);
        }
        return 0;
    }

    /// <summary>Qwen-Image's text side: the template's tokens, the hidden
    /// states of all of them (--dump-hidden) and the context past the
    /// system turn (--dump), the tensors the reference script writes.</summary>
    private static int QwenImage(CommandLine line, string prompt)
    {
        Dictionary<string, string> components = line.Components(QwenImageFamily.Instance,
            PipelineParts.Conditioner);
        line.ClipSkip(QwenImageFamily.Instance);
        using var encoder = new QwenImageTextEncoder(line.Checkpoint(),
            components[QwenImageFamily.TextEncoderComponent.Id]);

        (int[] ids, int drop) = encoder.Tokenize(prompt);
        Console.WriteLine($"ids [{ids.Length}], {drop} of the system turn: {string.Join(" ", ids)}");
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        float[] hidden = encoder.HiddenStates(ids);
        Console.WriteLine($"hidden [{ids.Length}, {QwenImageTextEncoder.Width}] in {stopwatch.Elapsed.TotalSeconds:0.0} s: " +
                          $"mean {hidden.Average():+0.0000;-0.0000} " +
                          $"std {Statistics.Std(hidden):0.0000} first {hidden[0]:0.0000}");
        float[] context = QwenImageTextEncoder.Context(hidden, drop);
        int rows = context.Length / QwenImageTextEncoder.Width;
        Console.WriteLine($"context [{rows}, {QwenImageTextEncoder.Width}]: " +
                          $"mean {context.Average():+0.0000;-0.0000} " +
                          $"std {Statistics.Std(context):0.0000} first {context[0]:0.0000}");

        if (line.Value("--dump-hidden") is string dumpHidden)
        {
            Dumps.Write(dumpHidden, hidden);
        }
        if (line.Value("--dump") is string dump)
        {
            Dumps.Write(dump, context);
        }
        return 0;
    }
}
