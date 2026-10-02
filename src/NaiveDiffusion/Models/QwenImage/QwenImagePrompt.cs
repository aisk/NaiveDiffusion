using NaiveDiffusion.Text;
using NaiveDiffusion.Text.Qwen;

namespace NaiveDiffusion.Models.QwenImage;

/// <summary>How a prompt becomes Qwen3-VL's input for Qwen-Image 2.1: the
/// chat template the model was trained with — a system turn asking it to
/// comprehend the prompt, the prompt as the user turn, an open assistant
/// turn — tokenized with the chat markers as single tokens. The whole
/// template goes through the language model, since every token's hidden
/// state depends on what came before it; the system turn's rows are then
/// dropped and the rest is the transformer's context, as the reference
/// does (<c>drop_idx</c> in diffusers, the second <c>&lt;|im_start|&gt;</c>
/// in ComfyUI).
///
/// The prompt is text as written, less the backslashes A1111 prompts put
/// before brackets (<see cref="PromptTags.Unescape"/>, as the other
/// families read them and as ComfyUI does for this one): no weights in
/// brackets, no thinking block appended (the reference passes thinking on,
/// which means none), and an empty prompt is the template around nothing
/// — one token fewer than ComfyUI, which pads it to a space.</summary>
public static class QwenImagePrompt
{
    private const string SystemTurn =
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";

    private const string UserTurn = "<|im_start|>user\n{0}<|im_end|>\n<|im_start|>assistant\n";

    /// <summary>The token ids of the whole template around
    /// <paramref name="prompt"/>, and how many of them are the system turn.</summary>
    public static (int[] Ids, int Drop) Tokenize(string prompt, Qwen2Tokenizer tokenizer)
    {
        int drop = tokenizer.EncodeWithSpecials(SystemTurn).Count;
        List<int> ids = tokenizer.EncodeWithSpecials(
            SystemTurn + string.Format(UserTurn, PromptTags.Unescape(prompt)));
        return (ids.ToArray(), drop);
    }

    /// <summary>How many context rows a prompt gives the transformer: its
    /// tokens past the system turn.</summary>
    public static int ContextRows(string prompt, Qwen2Tokenizer tokenizer)
    {
        (int[] ids, int drop) = Tokenize(prompt, tokenizer);
        return ids.Length - drop;
    }
}
