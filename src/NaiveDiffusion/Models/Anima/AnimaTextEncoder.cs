using NaiveDiffusion.Tensors;
using NaiveDiffusion.Text;
using NaiveDiffusion.Text.Qwen;
using NaiveDiffusion.Text.T5;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Models.Anima;

/// <summary>Anima's whole text side, prompt in and transformer context out:
/// the prompt cut for both tokenizers, Qwen3 over its ids, the adapter over
/// the T5 ids against those hidden states, the prompt's weights multiplied
/// onto the adapter's rows, and the result padded with zeros to the context
/// length the transformer expects. All of it on the CPU, off the mapped
/// files, which stay open for the life of this object, with any LoRA's
/// deltas for the encoder and the adapter folded in as they are read.</summary>
public sealed class AnimaTextEncoder : IDisposable
{
    private readonly AnimaWeights _weights;
    private readonly TowerWeights _textEncoder;
    private readonly TowerWeights _adapter;
    private readonly Qwen2Tokenizer _qwen = Qwen2Tokenizer.Shared;
    private readonly T5Tokenizer _t5 = T5Tokenizer.Shared;

    public AnimaTextEncoder(string checkpointPath, string textEncoderPath,
        IReadOnlyList<LoraSpec>? loras = null)
    {
        _weights = new AnimaWeights(checkpointPath, textEncoderPath);
        try
        {
            _textEncoder = new TowerWeights(_weights.LoadTextEncoder(loras));
            _adapter = new TowerWeights(_weights.LoadAdapter(loras));
        }
        catch
        {
            _weights.Dispose();
            throw;
        }
    }

    public AnimaTokens Tokenize(string prompt) => AnimaPrompt.Tokenize(prompt, _qwen, _t5);

    /// <summary>The Qwen3 hidden states for a prompt, [tokens, 1024].</summary>
    public float[] HiddenStates(AnimaTokens tokens) =>
        Qwen3TextModel.Forward(_textEncoder, tokens.Qwen, Qwen3TextModel.Config.Qwen3_0_6B);

    /// <summary>The adapter's output for a prompt, [T5 tokens, 1024], the
    /// prompt's weights applied and nothing padded.</summary>
    public float[] AdapterOutput(AnimaTokens tokens, float[] hiddenStates)
    {
        float[] output = LlmAdapter.Forward(_adapter, tokens.T5, hiddenStates, tokens.Qwen.Length);
        for (int row = 0; row < tokens.T5.Length; row++)
        {
            float weight = tokens.T5Weights[row];
            if (weight == 1f)
            {
                continue;
            }
            for (int i = row * LlmAdapter.Width; i < (row + 1) * LlmAdapter.Width; i++)
            {
                output[i] *= weight;
            }
        }
        return output;
    }

    /// <summary>How many context rows a prompt takes once padded: the
    /// context length, or the prompt's own T5 length past it.</summary>
    public static int ContextRows(AnimaTokens tokens) =>
        Math.Max(AnimaPrompt.ContextLength, tokens.T5.Length);

    /// <summary>The transformer's context for a prompt: the adapter output
    /// padded with zero rows to <paramref name="rows"/>, which is at least
    /// <see cref="ContextRows"/> of this prompt.</summary>
    public float[] Encode(string prompt, int rows)
    {
        AnimaTokens tokens = Tokenize(prompt);
        if (rows < tokens.T5.Length)
        {
            throw new ArgumentException($"{rows} context rows cannot hold {tokens.T5.Length} tokens");
        }
        float[] output = AdapterOutput(tokens, HiddenStates(tokens));
        var context = new float[rows * LlmAdapter.Width];
        Array.Copy(output, context, output.Length);
        return context;
    }

    public void Dispose()
    {
        _textEncoder.Clear();
        _adapter.Clear();
        _weights.Dispose();
    }
}
