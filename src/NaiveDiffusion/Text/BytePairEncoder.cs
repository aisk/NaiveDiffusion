using System.Text;
using System.Text.Json;

namespace NaiveDiffusion.Text;

/// <summary>GPT-2's byte-level byte-pair encoding, which CLIP and Qwen both
/// build their tokenizers on: every byte of the UTF-8 text is mapped to a
/// printable stand-in character, and a word of those is merged, lowest-ranked
/// adjacent pair first, until no pair has a rank in the merges table. What
/// differs between the two — the pre-tokenizing pattern, the case folding,
/// the markers — stays in each tokenizer; this is the table and the loop.</summary>
public sealed class BytePairEncoder
{
    private readonly Dictionary<(string, string), int> _ranks = new();
    private readonly char[] _byteEncoder = new char[256];
    private readonly string _endOfWord;

    /// <param name="merges">The merges.txt: one pair per line, in rank order,
    /// after a "#version" comment.</param>
    /// <param name="endOfWord">What CLIP hangs on a word's last character
    /// before merging, "&lt;/w&gt;"; empty for Qwen, whose words carry their
    /// leading space instead.</param>
    public BytePairEncoder(Stream merges, string endOfWord)
    {
        _endOfWord = endOfWord;
        using (var reader = new StreamReader(merges))
        {
            int rank = 0;
            while (reader.ReadLine() is string rawLine)
            {
                string line = rawLine.TrimEnd('\r', '\n');
                if (line.Length == 0 || line.StartsWith("#version", StringComparison.Ordinal))
                {
                    continue;
                }
                int space = line.IndexOf(' ');
                _ranks[(line[..space], line[(space + 1)..])] = rank++;
            }
        }

        // GPT-2's printable byte alphabet: printable bytes map to themselves,
        // the rest to code points from 256 up.
        var identity = new List<byte>();
        for (int b = '!'; b <= '~'; b++) identity.Add((byte)b);
        for (int b = 0xA1; b <= 0xAC; b++) identity.Add((byte)b);
        for (int b = 0xAE; b <= 0xFF; b++) identity.Add((byte)b);
        var mapped = new bool[256];
        foreach (byte b in identity)
        {
            _byteEncoder[b] = (char)b;
            mapped[b] = true;
        }
        int next = 256;
        for (int b = 0; b < 256; b++)
        {
            if (!mapped[b])
            {
                _byteEncoder[b] = (char)next++;
            }
        }
    }

    /// <summary>A vocab.json — token text to id — as a dictionary.</summary>
    public static Dictionary<string, int> LoadVocabulary(Stream json)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        var vocabulary = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (JsonProperty property in document.RootElement.EnumerateObject())
        {
            vocabulary[property.Name] = property.Value.GetInt32();
        }
        return vocabulary;
    }

    /// <summary>An embedded resource of this assembly by its logical name.</summary>
    public static Stream OpenResource(string name) =>
        typeof(BytePairEncoder).Assembly.GetManifestResourceStream(name)
        ?? throw new InvalidOperationException($"embedded resource {name} is missing");

    /// <summary>One word's UTF-8 bytes as the stand-in characters the merges
    /// table is written in.</summary>
    public string MapBytes(string word)
    {
        var mapped = new StringBuilder(word.Length);
        foreach (byte b in Encoding.UTF8.GetBytes(word))
        {
            mapped.Append(_byteEncoder[b]);
        }
        return mapped.ToString();
    }

    /// <summary>The pieces of one mapped word: characters at first, with the
    /// end-of-word marker riding on the last one where the tokenizer has
    /// one, merged until none of the adjacent pairs has a rank. Every
    /// occurrence of the chosen pair is merged at once, as the reference
    /// does.</summary>
    public List<string> Encode(string mappedWord)
    {
        var word = new List<string>(mappedWord.Length);
        for (int i = 0; i < mappedWord.Length; i++)
        {
            word.Add(i == mappedWord.Length - 1 ? mappedWord[i] + _endOfWord : mappedWord[i].ToString());
        }

        while (word.Count > 1)
        {
            int bestRank = int.MaxValue;
            int bestIndex = -1;
            for (int i = 0; i < word.Count - 1; i++)
            {
                if (_ranks.TryGetValue((word[i], word[i + 1]), out int rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestIndex = i;
                }
            }
            if (bestIndex < 0)
            {
                break;
            }

            string first = word[bestIndex];
            string second = word[bestIndex + 1];
            string merged = first + second;
            var next = new List<string>(word.Count);
            for (int i = 0; i < word.Count;)
            {
                if (i < word.Count - 1 && word[i] == first && word[i + 1] == second)
                {
                    next.Add(merged);
                    i += 2;
                }
                else
                {
                    next.Add(word[i]);
                    i += 1;
                }
            }
            word = next;
        }
        return word;
    }
}
