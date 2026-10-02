using System.Text.RegularExpressions;

namespace NaiveDiffusion.Text.Clip;

/// <summary>One chunk of a prompt as CLIP takes it: <see
/// cref="ClipTokenizer.MaxTokens"/> ids, and what each of those positions is
/// worth. The markers and the padding are always worth 1, and so is every
/// position of an unweighted prompt.</summary>
/// <param name="Ids">The token ids, start marker through padding.</param>
/// <param name="Weights">One multiplier per id, applied to the hidden states
/// the tower produces rather than to anything in here — see
/// <c>SdxlTextEncoders.Reweight</c>.</param>
public sealed record TokenChunk(uint[] Ids, float[] Weights);

/// <summary>CLIP's byte-pair-encoding tokenizer, matching transformers'
/// CLIPTokenizer: lowercase, collapse whitespace, split with CLIP's pattern,
/// then BPE over a byte-to-unicode alphabet with a word-final marker.
///
/// The vocabulary and merges ride along as embedded resources — they are
/// OpenAI CLIP's standard BPE files, the same for every SDXL model, and a
/// checkpoint never carries them. Only the pad token differs per tower.</summary>
public sealed partial class ClipTokenizer
{
    public const int MaxTokens = 77;

    /// <summary>The body tokens one chunk carries between its start and end
    /// markers.</summary>
    public const int ChunkTokens = MaxTokens - 2;

    /// <summary>Where a prompt stops being encoded at all. CLIP has no opinion
    /// past the first chunk — the cost of another one is that the UNet
    /// cross-attends to a longer sequence at every layer — and eight of them is
    /// six hundred tokens, which is past what anyone writes.</summary>
    public const int MaxChunks = 8;

    private static readonly Dictionary<string, ClipTokenizer> Shared = new(StringComparer.Ordinal);

    private readonly Dictionary<string, int> _vocab;
    private readonly BytePairEncoder _encoder;
    private readonly int _startToken;
    private readonly int _endToken;
    private readonly int _padToken;

    [GeneratedRegex(
        @"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
        RegexOptions.IgnoreCase)]
    private static partial Regex TokenPattern();

    /// <summary>The tokenizer for a tower, built once: the tables take a
    /// moment to read and are the same for every run.</summary>
    public static ClipTokenizer For(string padTokenText)
    {
        lock (Shared)
        {
            if (!Shared.TryGetValue(padTokenText, out ClipTokenizer? tokenizer))
            {
                Shared[padTokenText] = tokenizer = new ClipTokenizer(padTokenText);
            }
            return tokenizer;
        }
    }

    public ClipTokenizer(string padTokenText)
    {
        using (Stream vocab = BytePairEncoder.OpenResource("ClipTokenizer.vocab.json"))
        {
            _vocab = BytePairEncoder.LoadVocabulary(vocab);
        }
        using (Stream merges = BytePairEncoder.OpenResource("ClipTokenizer.merges.txt"))
        {
            _encoder = new BytePairEncoder(merges, "</w>");
        }
        _startToken = _vocab["<|startoftext|>"];
        _endToken = _vocab["<|endoftext|>"];
        _padToken = _vocab[padTokenText];
    }

    /// <summary>Encode a prompt of one weight throughout — an empty one, or
    /// any text nobody has weighted.</summary>
    public List<TokenChunk> EncodeChunks(string prompt) =>
        EncodeChunks(new[] { new PromptSegment(prompt, 1f) });

    /// <summary>Encode a prompt into as many chunks of <see cref="MaxTokens"/>
    /// ids as it takes, each a complete start/body/end sequence CLIP can be run
    /// on by itself. This is what ComfyUI and A1111 do with a prompt over the
    /// 75-token limit: cut it on the token grid, encode the pieces separately,
    /// and let the UNet cross-attend to them laid end to end. The cut falls
    /// wherever the count lands, so a phrase can straddle two chunks — the same
    /// as in the implementations this follows. Always at least one chunk: an
    /// empty prompt still encodes to start, end and padding.
    ///
    /// The segments are encoded one after another and the boundaries between
    /// them do not disturb anything: every piece the token pattern matches lies
    /// inside one segment, since a segment ends after its separator. So the ids
    /// are the ones the concatenated text would have given, and each carries
    /// the weight of the segment it came from.</summary>
    public List<TokenChunk> EncodeChunks(IReadOnlyList<PromptSegment> segments)
    {
        List<(int Id, float Weight)> body = Tokenize(segments);
        var chunks = new List<TokenChunk>();
        for (int start = 0; start < body.Count; start += ChunkTokens)
        {
            chunks.Add(Pack(body, start, Math.Min(ChunkTokens, body.Count - start)));
        }
        if (chunks.Count == 0)
        {
            chunks.Add(Pack(body, 0, 0));
        }
        return chunks;
    }

    /// <summary>The prompt's body tokens, each with the weight of the segment
    /// it came from, up to what <see cref="MaxChunks"/> holds.</summary>
    private List<(int Id, float Weight)> Tokenize(IReadOnlyList<PromptSegment> segments)
    {
        int limit = MaxChunks * ChunkTokens;
        var body = new List<(int, float)>();

        foreach (PromptSegment segment in segments)
        {
            string cleaned = Regex.Replace(segment.Text, @"\s+", " ").Trim().ToLowerInvariant();
            foreach (Match match in TokenPattern().Matches(cleaned))
            {
                foreach (string piece in _encoder.Encode(_encoder.MapBytes(match.Value)))
                {
                    body.Add((_vocab.TryGetValue(piece, out int id) ? id : _endToken,
                        segment.Weight));
                }
                if (body.Count >= limit)
                {
                    break;
                }
            }
            if (body.Count >= limit)
            {
                break;
            }
        }

        if (body.Count > limit)
        {
            body.RemoveRange(limit, body.Count - limit);
        }
        return body;
    }

    /// <summary>One chunk's worth of body tokens wrapped in the markers and
    /// padded out to <see cref="MaxTokens"/>.</summary>
    private TokenChunk Pack(List<(int Id, float Weight)> body, int start, int count)
    {
        var ids = new uint[MaxTokens];
        var weights = new float[MaxTokens];
        // The markers and the padding are the prompt's frame rather than part
        // of what it asks for, so nothing the user wrote weights them.
        Array.Fill(weights, 1f);

        ids[0] = (uint)_startToken;
        for (int i = 0; i < count; i++)
        {
            ids[i + 1] = (uint)body[start + i].Id;
            weights[i + 1] = body[start + i].Weight;
        }
        ids[count + 1] = (uint)_endToken;
        for (int i = count + 2; i < MaxTokens; i++)
        {
            ids[i] = (uint)_padToken;
        }
        return new TokenChunk(ids, weights);
    }
}
