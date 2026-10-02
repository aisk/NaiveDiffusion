using NaiveDiffusion.Text;
using NaiveDiffusion.Text.Qwen;
using NaiveDiffusion.Text.T5;

namespace NaiveDiffusion.Models.Anima;

/// <summary>A prompt cut two ways, as Anima's text side wants it: Qwen ids
/// for the text encoder to read, and T5 ids — with a weight each — for the
/// adapter to embed. Every id is a whole sequence: no markers on the Qwen
/// side, T5's end marker on the other.</summary>
/// <param name="Qwen">Never empty: an empty prompt is Qwen's pad token
/// alone, which is what the reference feeds too.</param>
/// <param name="T5">Ends with <see cref="T5Tokenizer.EndToken"/>.</param>
/// <param name="T5Weights">One per T5 id, the end marker's 1; multiplied
/// onto the adapter's output row for that token.</param>
public sealed record AnimaTokens(int[] Qwen, int[] T5, float[] T5Weights);

/// <summary>ComfyUI's tokenization of a prompt for Anima: the text split
/// into weighted pieces, each piece tokenized on its own by both tokenizers
/// and the pieces laid end to end. The weights are carried by the T5 side
/// only — the Qwen side is read at weight 1, as the reference does —
/// because the adapter output is where a per-token multiplier can be applied.</summary>
public static class AnimaPrompt
{
    /// <summary>What ComfyUI pads an adapter output to: the context length
    /// the transformer was trained against. A longer prompt is longer.</summary>
    public const int ContextLength = 512;

    /// <summary>The prompt as the pieces ComfyUI's tokenizer cuts it into: a
    /// weighted group, '(text:1.2)', is a piece of its own at that weight,
    /// and everything between two groups is one piece at weight 1 — commas,
    /// spaces and all, because the tokenizers read a space as part of the
    /// word after it and cutting at every tag would change the ids. The
    /// groups are the ones <see cref="PromptTags.Unwrap"/> recognizes, so a
    /// bare '(vocaloid)' stays text here as it does everywhere else in this
    /// library, where ComfyUI would read it as emphasis.</summary>
    public static IReadOnlyList<PromptSegment> Segments(string prompt)
    {
        var segments = new List<PromptSegment>();
        var pending = new System.Text.StringBuilder();
        for (int i = 0; i < prompt.Length; i++)
        {
            char c = prompt[i];
            if (c == '\\' && i + 1 < prompt.Length)
            {
                pending.Append(c).Append(prompt[++i]);
                continue;
            }
            if (c != '(')
            {
                pending.Append(c);
                continue;
            }
            int close = MatchingClose(prompt, i);
            string group = close < 0 ? "" : prompt[i..(close + 1)];
            (string content, float weight) = PromptTags.Unwrap(group);
            if (close < 0 || weight == 1f && content == group)
            {
                // Not a weight group: the bracket is text.
                pending.Append(c);
                continue;
            }
            Flush();
            segments.Add(new PromptSegment(PromptTags.Unescape(content), weight));
            i = close;
        }
        Flush();
        return segments;

        void Flush()
        {
            if (pending.Length > 0)
            {
                segments.Add(new PromptSegment(PromptTags.Unescape(pending.ToString()), 1f));
                pending.Clear();
            }
        }
    }

    /// <summary>The index of the ')' closing the '(' at <paramref name="open"/>,
    /// escapes skipped, or −1 when it never closes.</summary>
    private static int MatchingClose(string prompt, int open)
    {
        int depth = 0;
        for (int i = open; i < prompt.Length; i++)
        {
            char c = prompt[i];
            if (c == '\\')
            {
                i++;
            }
            else if (c == '(')
            {
                depth++;
            }
            else if (c == ')' && --depth == 0)
            {
                return i;
            }
        }
        return -1;
    }

    public static AnimaTokens Tokenize(string prompt, Qwen2Tokenizer qwen, T5Tokenizer t5)
    {
        var qwenIds = new List<int>();
        var t5Ids = new List<int>();
        var weights = new List<float>();
        foreach (PromptSegment segment in Segments(prompt))
        {
            if (segment.Text.Length == 0)
            {
                continue;
            }
            qwenIds.AddRange(qwen.Encode(segment.Text));
            List<int> piece = t5.Encode(segment.Text);
            t5Ids.AddRange(piece);
            weights.AddRange(Enumerable.Repeat(segment.Weight, piece.Count));
        }
        if (qwenIds.Count == 0)
        {
            qwenIds.Add(Qwen2Tokenizer.PadToken);
        }
        t5Ids.Add(T5Tokenizer.EndToken);
        weights.Add(1f);
        return new AnimaTokens(qwenIds.ToArray(), t5Ids.ToArray(), weights.ToArray());
    }
}
