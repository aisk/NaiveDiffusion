using System.Text.RegularExpressions;

namespace NaiveDiffusion.Text.Qwen;

/// <summary>The Qwen2 family's byte-level BPE tokenizer, matching transformers'
/// slow Qwen2Tokenizer over the same vocab.json and merges.txt: split with
/// Qwen's pattern, map every byte to a printable stand-in, merge by rank. No
/// lowercasing, no start or end marker, no whitespace cleanup — a space is a
/// character of the word after it, and two spaces are two characters.
///
/// The vocabulary rides along as an embedded resource: Qwen3-0.6B, which
/// Anima reads its prompt with, never carries it in its weights file, and
/// every Qwen2 and Qwen3 text model shares the one table. The special
/// tokens past its end — the chat markers and the vision placeholders —
/// are listed here, since the table does not have them, and only
/// <see cref="EncodeWithSpecials"/> looks for them.</summary>
public sealed partial class Qwen2Tokenizer
{
    /// <summary>What ComfyUI pads an empty prompt with, and the only special
    /// token <see cref="Encode"/> ever emits: Qwen's end-of-text.</summary>
    public const int PadToken = 151643;

    public const int ImStart = 151644;
    public const int ImEnd = 151645;

    /// <summary>The added tokens of Qwen2.5 and Qwen3, by their text: the
    /// ids after the vocabulary that transformers splits a text on before
    /// any byte-pair step, wherever they occur in it.</summary>
    private static readonly Dictionary<string, int> SpecialTokens = new()
    {
        ["<|endoftext|>"] = PadToken,
        ["<|im_start|>"] = ImStart,
        ["<|im_end|>"] = ImEnd,
        ["<|object_ref_start|>"] = 151646,
        ["<|object_ref_end|>"] = 151647,
        ["<|box_start|>"] = 151648,
        ["<|box_end|>"] = 151649,
        ["<|quad_start|>"] = 151650,
        ["<|quad_end|>"] = 151651,
        ["<|vision_start|>"] = 151652,
        ["<|vision_end|>"] = 151653,
        ["<|vision_pad|>"] = 151654,
        ["<|image_pad|>"] = 151655,
        ["<|video_pad|>"] = 151656,
        ["<tool_call>"] = 151657,
        ["</tool_call>"] = 151658,
        ["<|fim_prefix|>"] = 151659,
        ["<|fim_middle|>"] = 151660,
        ["<|fim_suffix|>"] = 151661,
        ["<|fim_pad|>"] = 151662,
        ["<|repo_name|>"] = 151663,
        ["<|file_sep|>"] = 151664,
        ["<tool_response>"] = 151665,
        ["</tool_response>"] = 151666,
        ["<think>"] = 151667,
        ["</think>"] = 151668,
    };

    [GeneratedRegex(@"<\|[a-z_]+\|>|</?tool_call>|</?tool_response>|</?think>")]
    private static partial Regex SpecialPattern();

    private static readonly Lazy<Qwen2Tokenizer> SharedInstance = new(() => new Qwen2Tokenizer());

    private readonly Dictionary<string, int> _vocab;
    private readonly BytePairEncoder _encoder;
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, int[]> _cache = new();

    /// <summary>The one instance: the tables are 150 thousand entries and
    /// the same for every run.</summary>
    public static Qwen2Tokenizer Shared => SharedInstance.Value;

    // transformers' PRETOKENIZE_REGEX for Qwen2, as .NET reads it: the
    // contractions, a letter run with one leading non-letter, one digit at a
    // time, a run of punctuation with an optional leading space, and the
    // whitespace rules that keep a run's last space for the word after it.
    [GeneratedRegex(
        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")]
    private static partial Regex TokenPattern();

    public Qwen2Tokenizer()
    {
        using (Stream vocab = BytePairEncoder.OpenResource("Qwen2Tokenizer.vocab.json"))
        {
            _vocab = BytePairEncoder.LoadVocabulary(vocab);
        }
        using (Stream merges = BytePairEncoder.OpenResource("Qwen2Tokenizer.merges.txt"))
        {
            _encoder = new BytePairEncoder(merges, "");
        }
    }

    /// <summary>The token ids of a text, nothing added at either end. Empty
    /// text gives no ids; a caller that needs at least one pads with
    /// <see cref="PadToken"/>, as ComfyUI does.</summary>
    public List<int> Encode(string text)
    {
        var ids = new List<int>();
        foreach (Match match in TokenPattern().Matches(text))
        {
            ids.AddRange(EncodeWord(match.Value));
        }
        return ids;
    }

    /// <summary>The token ids of a text that may carry special tokens, as
    /// transformers reads a chat template: each special token is one id and
    /// the text between them is encoded on its own, so a piece never merges
    /// across one. A name in the special form that is not a special token
    /// is text.</summary>
    public List<int> EncodeWithSpecials(string text)
    {
        var ids = new List<int>();
        int start = 0;
        foreach (Match match in SpecialPattern().Matches(text))
        {
            if (!SpecialTokens.TryGetValue(match.Value, out int id))
            {
                continue;
            }
            ids.AddRange(Encode(text[start..match.Index]));
            ids.Add(id);
            start = match.Index + match.Length;
        }
        ids.AddRange(Encode(text[start..]));
        return ids;
    }

    private int[] EncodeWord(string word)
    {
        if (_cache.TryGetValue(word, out int[]? cached))
        {
            return cached;
        }
        List<string> pieces = _encoder.Encode(_encoder.MapBytes(word));
        var ids = new int[pieces.Count];
        for (int i = 0; i < ids.Length; i++)
        {
            // Every single byte is in the vocabulary, so a piece that is
            // not is a fault in the table, not in the text.
            ids[i] = _vocab.TryGetValue(pieces[i], out int id)
                ? id
                : throw new InvalidDataException($"{pieces[i]} is not in the Qwen2 vocabulary");
        }
        _cache[word] = ids;
        return ids;
    }
}
