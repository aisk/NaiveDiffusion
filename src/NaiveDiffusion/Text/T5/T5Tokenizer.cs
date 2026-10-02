using System.Text;
using System.Text.Json;

namespace NaiveDiffusion.Text.T5;

/// <summary>T5's SentencePiece unigram tokenizer, matching transformers'
/// T5TokenizerFast over the same tokenizer.json: normalize, turn spaces into
/// the ▁ mark with one in front, then the Viterbi segmentation over the
/// 32,000 scored pieces. Anima's text side uses it not to read a T5 — there
/// is none — but because its adapter's embedding table was trained on T5's
/// ids, so the prompt has to be cut the way T5 cuts it.
///
/// The normalizer is the one place this is not the reference bit for bit:
/// SentencePiece's nmt_nfkc is a compiled character map, and what is
/// reproduced here is NFKC plus its whitespace and control-character rules,
/// which is what it comes to on anything a prompt is likely to hold.</summary>
public sealed class T5Tokenizer
{
    private static readonly Lazy<T5Tokenizer> SharedInstance = new(() => new T5Tokenizer());

    /// <summary>The one instance: the model is the same for every run.</summary>
    public static T5Tokenizer Shared => SharedInstance.Value;

    /// <summary>The end-of-sequence id the reference appends to every text.</summary>
    public const int EndToken = 1;

    private const int UnknownToken = 2;
    private const char Metaspace = '▁';

    private readonly Dictionary<string, (int Id, double Score)> _pieces = new(StringComparer.Ordinal);
    private readonly int _longestPiece;
    private readonly double _unknownScore;

    public T5Tokenizer()
    {
        using Stream stream = typeof(T5Tokenizer).Assembly.GetManifestResourceStream("T5Tokenizer.tokenizer.json")
            ?? throw new InvalidOperationException("embedded resource T5Tokenizer.tokenizer.json is missing");
        using JsonDocument json = JsonDocument.Parse(stream);
        JsonElement model = json.RootElement.GetProperty("model");
        if (model.GetProperty("type").GetString() != "Unigram")
        {
            throw new InvalidDataException("the T5 tokenizer file is not a unigram model");
        }

        double lowest = 0;
        int id = 0;
        foreach (JsonElement entry in model.GetProperty("vocab").EnumerateArray())
        {
            string piece = entry[0].GetString()!;
            double score = entry[1].GetDouble();
            _pieces[piece] = (id++, score);
            _longestPiece = Math.Max(_longestPiece, piece.Length);
            lowest = Math.Min(lowest, score);
        }
        // tokenizers' unk_penalty: an unknown character scores ten below the
        // rarest piece, so it is only ever taken where nothing else fits.
        _unknownScore = lowest - 10.0;
    }

    /// <summary>The ids of a text, without the end marker. Empty text gives
    /// no ids.</summary>
    public List<int> Encode(string text)
    {
        var ids = new List<int>();
        string normalized = Normalize(text);
        if (normalized.Length == 0)
        {
            return ids;
        }

        // Viterbi over character positions: best[i] is the score of the best
        // segmentation of the first i characters, from[i] where its last
        // piece began. A position no piece reaches is an unknown character.
        int length = normalized.Length;
        var best = new double[length + 1];
        var from = new int[length + 1];
        var unknown = new bool[length + 1];
        Array.Fill(best, double.NegativeInfinity);
        best[0] = 0;
        for (int start = 0; start < length; start++)
        {
            if (double.IsNegativeInfinity(best[start]))
            {
                continue;
            }
            bool matched = false;
            int longest = Math.Min(_longestPiece, length - start);
            for (int count = 1; count <= longest; count++)
            {
                // A piece never ends between the two halves of a surrogate pair.
                if (char.IsHighSurrogate(normalized[start + count - 1]))
                {
                    continue;
                }
                if (_pieces.TryGetValue(normalized.Substring(start, count), out (int Id, double Score) piece))
                {
                    matched = true;
                    Relax(start, start + count, piece.Score, false);
                }
            }
            if (!matched)
            {
                int width = char.IsHighSurrogate(normalized[start]) && start + 1 < length ? 2 : 1;
                Relax(start, start + width, _unknownScore, true);
            }
        }

        // Walk back from the end, then reverse. Runs of unknown characters
        // fuse into one unknown id, as tokenizers does for this model.
        var reversed = new List<int>();
        bool lastUnknown = false;
        for (int end = length; end > 0; end = from[end])
        {
            if (unknown[end])
            {
                if (!lastUnknown)
                {
                    reversed.Add(UnknownToken);
                }
                lastUnknown = true;
            }
            else
            {
                reversed.Add(_pieces[normalized[from[end]..end]].Id);
                lastUnknown = false;
            }
        }
        reversed.Reverse();
        ids.AddRange(reversed);
        return ids;

        void Relax(int start, int end, double score, bool isUnknown)
        {
            double total = best[start] + score;
            if (total > best[end])
            {
                best[end] = total;
                from[end] = start;
                unknown[end] = isUnknown;
            }
        }
    }

    /// <summary>nmt_nfkc as far as prompts go — NFKC, every kind of space to
    /// a plain one, control characters dropped — then tokenizers' own steps:
    /// trailing space stripped, runs of spaces collapsed, spaces turned into
    /// ▁ with one in front.</summary>
    private static string Normalize(string text)
    {
        var mapped = new StringBuilder(text.Length + 1);
        foreach (char c in text.Normalize(NormalizationForm.FormKC))
        {
            if (IsSpace(c))
            {
                mapped.Append(' ');
            }
            else if (!IsDropped(c))
            {
                mapped.Append(c);
            }
        }
        string stripped = mapped.ToString().TrimEnd(' ');
        if (stripped.Length == 0)
        {
            return "";
        }

        var result = new StringBuilder(stripped.Length + 1);
        if (stripped[0] != ' ' && stripped[0] != Metaspace)
        {
            result.Append(Metaspace);
        }
        bool lastSpace = false;
        foreach (char c in stripped)
        {
            if (c == ' ')
            {
                if (!lastSpace)
                {
                    result.Append(Metaspace);
                }
                lastSpace = true;
            }
            else
            {
                result.Append(c);
                lastSpace = false;
            }
        }
        return result.ToString();
    }

    /// <summary>The whitespace nmt_nfkc turns into a space: the ASCII
    /// controls that break lines, the Unicode spaces, and the zero-width
    /// space, which it treats as one. Code points rather than character
    /// literals, so the table reads as the standard lists it.</summary>
    private static bool IsSpace(char c) => (int)c is 0x20 or 0x09 or 0x0A or 0x0B or 0x0C
        or 0x0D or 0x85 or 0xA0 or 0x1680 or (>= 0x2000 and <= 0x200B) or 0x2028 or 0x2029
        or 0x202F or 0x205F or 0x3000;

    /// <summary>Control and formatting characters nmt_nfkc removes.</summary>
    private static bool IsDropped(char c) => (int)c is (>= 0x00 and <= 0x08)
        or (>= 0x0E and <= 0x1F) or (>= 0x7F and <= 0x84) or (>= 0x86 and <= 0x9F)
        or (>= 0x200C and <= 0x200F) or (>= 0x2060 and <= 0x2064) or 0xFEFF;
}
