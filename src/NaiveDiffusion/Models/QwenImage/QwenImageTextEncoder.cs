using NaiveDiffusion.Text;
using NaiveDiffusion.Text.Qwen;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Models.QwenImage;

/// <summary>Qwen-Image 2.1's whole text side, prompt in and context rows
/// out: the chat template around the prompt through Qwen3-VL-8B's language
/// model on the CPU, read at its last layer before the final norm, with the
/// system turn's rows dropped. All of it off the mapped file, which stays
/// open for the life of this object; the weights are 16 GiB of bfloat16
/// read a row at a time, so a prompt takes some five seconds and the
/// first one after a cold start waits on the disk as well.</summary>
public sealed class QwenImageTextEncoder : IDisposable
{
    public const int Width = 4096;

    private readonly QwenImageWeights _weights;
    private readonly TowerWeights _textEncoder;
    private readonly Qwen2Tokenizer _tokenizer = Qwen2Tokenizer.Shared;

    public QwenImageTextEncoder(string checkpointPath, string textEncoderPath,
        IReadOnlyList<LoraSpec>? loras = null)
    {
        _weights = new QwenImageWeights(checkpointPath, textEncoderPath);
        try
        {
            _textEncoder = new TowerWeights(_weights.LoadTextEncoder(loras));
        }
        catch
        {
            _weights.Dispose();
            throw;
        }
    }

    public (int[] Ids, int Drop) Tokenize(string prompt) => QwenImagePrompt.Tokenize(prompt, _tokenizer);

    public int ContextRows(string prompt) => QwenImagePrompt.ContextRows(prompt, _tokenizer);

    /// <summary>The hidden states for every token of the template, [ids.Length, 4096].</summary>
    public float[] HiddenStates(int[] ids) =>
        Qwen3TextModel.Forward(_textEncoder, ids, Qwen3TextModel.Config.Qwen3Vl_8B, finalNorm: false);

    /// <summary>The transformer's context out of the template's hidden
    /// states: the rows past the system turn's <paramref name="drop"/>.</summary>
    public static float[] Context(float[] hidden, int drop)
    {
        var context = new float[hidden.Length - drop * Width];
        Array.Copy(hidden, drop * Width, context, 0, context.Length);
        return context;
    }

    /// <summary>The transformer's context for a prompt: the hidden states
    /// past the system turn, [<see cref="ContextRows"/>, 4096].</summary>
    public float[] Encode(string prompt)
    {
        (int[] ids, int drop) = Tokenize(prompt);
        return Context(HiddenStates(ids), drop);
    }

    public void Dispose()
    {
        _textEncoder.Clear();
        _weights.Dispose();
    }
}
