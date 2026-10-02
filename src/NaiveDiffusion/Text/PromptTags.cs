using System.Globalization;
using System.Text;

namespace NaiveDiffusion.Text;

/// <summary>What is suspicious about one tag. Every one of these is a typo the
/// prompt survives — CLIP encodes the text whatever it says — which is exactly
/// why they are worth pointing at: nothing else ever complains.</summary>
public enum PromptIssue
{
    /// <summary>A full-width comma inside a tag. The prompt was written on an
    /// IME still in Chinese/Japanese mode, so what reads as a separator is a
    /// word to CLIP, and the two tags around it fused into one.</summary>
    FullWidthComma,

    /// <summary>Other full-width punctuation. Harmless to the tokenizer but
    /// almost never meant, and full-width brackets in particular do not do what
    /// the ASCII ones do.</summary>
    FullWidthPunctuation,

    /// <summary>A '(' or '[' the tag never closes — the usual shape of a
    /// literal bracket that wanted a backslash in front of it, and of a tag
    /// pasted with half of its weight syntax.</summary>
    UnclosedBracket,

    /// <summary>A ')' or ']' with nothing open before it.</summary>
    StrayBracket,

    /// <summary>The same tag already appeared earlier in the prompt.</summary>
    Duplicate,

    /// <summary>An A1111 '&lt;lora:name:weight&gt;' tag, which names a file
    /// rather than saying anything to CLIP. A run takes it out of the prompt
    /// and puts the LoRA on the list, if it can find the file — see
    /// <see cref="LoraTags"/>.</summary>
    LoraTag,
}

/// <summary>How a family's encoder reads a prompt, which decides what a
/// prompt box is for. <see cref="Tags"/>: comma-separated tags, each with a
/// weight of its own — what CLIP and Anima's encoder take, and what the
/// typo checks are built around. <see cref="Sentences"/>: prose handed to
/// the encoder as written — Qwen-Image's Qwen3-VL — where a comma is
/// punctuation, a full-width one in a Chinese sentence is the right one,
/// and a bracket or a weight group is just more text.</summary>
public enum PromptStyle
{
    Tags,
    Sentences,
}

/// <summary>One comma-separated piece of a prompt, as written.</summary>
/// <param name="Text">The tag's own text, trimmed, with its brackets,
/// backslashes and weight group left exactly as typed.</param>
/// <param name="Issues">What <see cref="PromptTags.Parse"/> found wrong with
/// it, empty for a tag with nothing to report.</param>
/// <param name="Content">The tag with its weight group taken off: the words
/// themselves, which is what a front end shows and what reaches CLIP.</param>
/// <param name="Weight">What the tag is worth, 1 for a tag that never asked
/// for anything — see <see cref="PromptTags.Unwrap"/> for what asking looks
/// like.</param>
public sealed record PromptTag(
    string Text, IReadOnlyList<PromptIssue> Issues, string Content, float Weight);

/// <summary>One run of prompt text carrying a single weight, which is the form
/// the tokenizer wants: it encodes the runs one after another and gives every
/// token the weight of the run it came out of.
///
/// The separators are part of the text, so concatenating the segments spells
/// the prompt out again as <see cref="PromptTags.Join"/> would write it — one
/// ", " between tags, whatever separated them when typed.</summary>
public readonly record struct PromptSegment(string Text, float Weight);

/// <summary>A prompt cut into tags, the way everyone writing for SDXL already
/// reads one: comma-separated pieces, with brackets holding their contents
/// together.
///
/// This is mostly a text-level view: the pieces reach CLIP as the one string
/// they came from. It exists so a front end can show them separately and say
/// which of them look mistyped, since the pipeline itself accepts anything and
/// a wrong comma only shows up in the image.
///
/// The one thing in here that changes what the model sees is weight, and it is
/// read in a single form: a tag that is entirely '(text:number)' is worth that
/// number, and every other bracket is literal text. A1111 additionally reads a
/// bare '(x)' as 1.1, which is why prompts written for it have to put a
/// backslash in front of both brackets of a tag like 'hatsune miku (vocaloid)'.
/// Nothing here needs that, and anything written with a number means the same
/// in both. <see cref="Segments"/> is where a prompt turns into the weighted
/// text the tokenizer runs on.</summary>
public static class PromptTags
{
    /// <summary>Full-width forms that are separators in prose but words to
    /// CLIP. Both are reported as <see cref="PromptIssue.FullWidthComma"/> and
    /// both are what <see cref="Repair"/> rewrites.</summary>
    private const string FullWidthCommas = "，、";

    /// <summary>The rest of the full-width block worth naming. The ideographic
    /// space is in here because it does not separate anything either.</summary>
    private const string FullWidthPunctuation = "（）［］｛｝〔〕【】「」『』：；！？。．　·＂＇＜＞";

    /// <summary>Cut a prompt into tags and inspect each one.
    ///
    /// Separators are the ASCII comma and the line break, both only at bracket
    /// depth zero — '(cat, dog:1.2)' stays a single tag, because splitting it
    /// would put half a weight group in each piece. Empty pieces (a doubled or
    /// trailing comma) are dropped rather than reported: joining the tags back
    /// together is itself the fix.</summary>
    public static IReadOnlyList<PromptTag> Parse(string prompt) => Parse(prompt, PromptStyle.Tags);

    /// <summary>The same cut, inspected for the style the prompt is read in.
    /// A sentence is not checked for what only a tag can get wrong: its
    /// punctuation, its brackets and a repeated clause are all text to the
    /// encoder, so only a '&lt;lora:&gt;' tag — which no run accepts in a
    /// prompt — is still reported.</summary>
    public static IReadOnlyList<PromptTag> Parse(string prompt, PromptStyle style)
    {
        var tags = new List<PromptTag>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (string text in Split(prompt))
        {
            List<PromptIssue> issues = Inspect(text);
            (string content, float weight) = Unwrap(text);
            // Two tags are the same when CLIP would see the same thing, so
            // neither the weight nor a stray escape makes them different:
            // '1girl' twice is twice through CLIP whatever either copy says.
            if (!seen.Add(Normalize(Unescape(content))))
            {
                issues.Add(PromptIssue.Duplicate);
            }
            if (style == PromptStyle.Sentences)
            {
                issues.RemoveAll(issue => issue != PromptIssue.LoraTag);
            }
            tags.Add(new PromptTag(text, issues, content, weight));
        }
        return tags;
    }

    /// <summary>Split text that is still being typed. Everything up to the last
    /// separator is finished and comes back as tags; what follows it is
    /// <paramref name="pending"/>, the piece the caret is still inside.
    ///
    /// A separator inside brackets does not count, so '(cat, ' leaves the whole
    /// thing pending — the group is not finished until its ')' arrives.</summary>
    public static IReadOnlyList<string> SplitTyped(string text, out string pending)
    {
        int cut = LastSeparator(text);
        if (cut < 0)
        {
            pending = text;
            return Array.Empty<string>();
        }
        pending = text[(cut + 1)..];
        return Split(text[..cut]);
    }

    /// <summary>Tags back into a prompt: one comma and one space between them,
    /// which is the separator the tokenizer sees as a single token. Blank tags
    /// drop out, so this is also how a prompt gets its stray commas cleaned
    /// up.</summary>
    public static string Join(IEnumerable<string> tags) =>
        string.Join(", ", tags.Select(tag => tag.Trim()).Where(tag => tag.Length > 0));

    /// <summary>A prompt as the weighted text the tokenizer runs on: one
    /// segment per tag, carrying that tag's weight, with ", " on the end of
    /// all but the last so the segments still spell the whole prompt.
    ///
    /// This is also where the backslashes come off. A1111 prompts escape a
    /// literal bracket as '\(' because a bare one means emphasis there; here
    /// it never did, but a prompt pasted in from elsewhere still carries them,
    /// and a backslash left in is a token CLIP would otherwise read as part of
    /// the word.</summary>
    public static IReadOnlyList<PromptSegment> Segments(string prompt)
    {
        IReadOnlyList<PromptTag> tags = Parse(prompt);
        var segments = new List<PromptSegment>(tags.Count);
        for (int i = 0; i < tags.Count; i++)
        {
            string text = Unescape(tags[i].Content) + (i + 1 < tags.Count ? ", " : "");
            segments.Add(new PromptSegment(text, tags[i].Weight));
        }
        return segments;
    }

    /// <summary>One tag split into what it says and what it is worth.
    ///
    /// A weight group is the whole tag and nothing less: '(' first, the ')' it
    /// opens last, and a number after the final colon inside it. Everything
    /// else comes back as it went in at weight 1 — '(vocaloid)' has no number,
    /// '(re:zero)' has no number after its colon, and '(a:1.2) b' does not
    /// reach the end. That strictness is the whole point: it is what lets a
    /// danbooru tag keep its brackets.
    ///
    /// Nesting multiplies, so '((a:1.1):1.2)' is worth 1.32 — the same as
    /// A1111 makes it.</summary>
    public static (string Content, float Weight) Unwrap(string tag)
    {
        string text = tag.Trim();
        if (text.Length < 2 || text[0] != '(' || text[^1] != ')')
        {
            return (text, 1f);
        }

        int depth = 0, colon = -1;
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (c == '\\' && i + 1 < text.Length)
            {
                i++;
                continue;
            }
            switch (c)
            {
                case '(' or '[':
                    depth++;
                    break;
                case ')' or ']':
                    depth--;
                    // Closed before the end: the opening bracket is not the one
                    // the last character closes, so this is two groups or worse.
                    if (depth == 0 && i != text.Length - 1)
                    {
                        return (text, 1f);
                    }
                    break;
                // The last one at the group's own level, so the number in
                // '(re:zero kara:1.2)' is the 1.2 and not the zero.
                case ':' when depth == 1:
                    colon = i;
                    break;
            }
        }
        if (depth != 0 || colon < 0)
        {
            return (text, 1f);
        }

        if (!float.TryParse(text[(colon + 1)..^1].Trim(), NumberStyles.Float,
                CultureInfo.InvariantCulture, out float weight) || !float.IsFinite(weight))
        {
            return (text, 1f);
        }
        string content = text[1..colon].Trim();
        if (content.Length == 0)
        {
            return (text, 1f);
        }
        (string inner, float nested) = Unwrap(content);
        return (inner, weight * nested);
    }

    /// <summary>One tag written out at a given weight. A weight of 1 is written
    /// as no weight at all rather than as '(tag:1)': the brackets would then be
    /// something to look at in every tag that had ever been nudged.
    ///
    /// A tag whose own brackets do not balance cannot be wrapped: 'a)' would
    /// come out as '(a):1.2)', which reads back as literal text at weight 1.
    /// Such a tag comes back unweighted, for its author to fix first.</summary>
    public static string WithWeight(string tag, float weight)
    {
        (string content, _) = Unwrap(tag);
        if (weight == 1f)
        {
            return content;
        }
        string wrapped = $"({content}:{weight.ToString("0.##", CultureInfo.InvariantCulture)})";
        return Unwrap(wrapped).Content == content ? wrapped : content;
    }

    /// <summary>Whether <see cref="Repair"/> would change anything — that is,
    /// whether the prompt holds a full-width comma. Nothing else is repaired
    /// automatically: an unclosed bracket has no single right fix (escape it or
    /// close it), and only the person who typed it knows which.</summary>
    public static bool NeedsRepair(string prompt) => NeedsRepair(prompt, PromptStyle.Tags);

    /// <summary>Never for a sentence: its full-width commas are its own.</summary>
    public static bool NeedsRepair(string prompt, PromptStyle style) =>
        style == PromptStyle.Tags && prompt.AsSpan().IndexOfAny(FullWidthCommas) >= 0;

    /// <summary>Every full-width comma turned into an ASCII one. The text is
    /// otherwise untouched — line breaks and spacing survive, so a prompt the
    /// user laid out by hand still looks like theirs afterwards.</summary>
    public static string Repair(string prompt)
    {
        var repaired = new StringBuilder(prompt.Length);
        for (int i = 0; i < prompt.Length; i++)
        {
            char c = prompt[i];
            if (!FullWidthCommas.Contains(c))
            {
                repaired.Append(c);
                continue;
            }
            // The space belongs to the separator, so one that is already there
            // does not become two.
            repaired.Append(',');
            if (i + 1 >= prompt.Length || prompt[i + 1] != ' ')
            {
                repaired.Append(' ');
            }
        }
        return repaired.ToString();
    }

    /// <summary>The pieces, trimmed, with the empty ones left out.</summary>
    private static List<string> Split(string prompt)
    {
        var pieces = new List<string>();
        var current = new StringBuilder();
        int depth = 0;

        void Flush()
        {
            string piece = current.ToString().Trim();
            if (piece.Length > 0)
            {
                pieces.Add(piece);
            }
            current.Clear();
        }

        for (int i = 0; i < prompt.Length; i++)
        {
            char c = prompt[i];
            // A backslash takes the next character with it, brackets included:
            // '\(' is a literal one and must not open a group.
            if (c == '\\' && i + 1 < prompt.Length)
            {
                current.Append(c).Append(prompt[i + 1]);
                i++;
                continue;
            }
            switch (c)
            {
                case '(' or '[':
                    depth++;
                    break;
                case ')' or ']':
                    // An unbalanced closer would otherwise push the depth
                    // negative and swallow every later comma.
                    depth = Math.Max(0, depth - 1);
                    break;
            }
            if (depth == 0 && (c == ',' || c == '\n' || c == '\r'))
            {
                Flush();
                continue;
            }
            current.Append(c);
        }
        Flush();
        return pieces;
    }

    /// <summary>Where the last top-level separator sits, or -1 when the text
    /// holds none.</summary>
    private static int LastSeparator(string text)
    {
        int cut = -1, depth = 0;
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (c == '\\' && i + 1 < text.Length)
            {
                i++;
                continue;
            }
            switch (c)
            {
                case '(' or '[':
                    depth++;
                    break;
                case ')' or ']':
                    depth = Math.Max(0, depth - 1);
                    break;
                case ',' or '\n' or '\r' when depth == 0:
                    cut = i;
                    break;
            }
        }
        return cut;
    }

    /// <summary>What one tag's own text has wrong with it. Duplicates are not
    /// visible from here and are added by <see cref="Parse"/>.</summary>
    private static List<PromptIssue> Inspect(string tag)
    {
        var issues = new List<PromptIssue>();
        int parens = 0, brackets = 0;
        bool stray = false, fullComma = false, fullPunctuation = false;

        for (int i = 0; i < tag.Length; i++)
        {
            char c = tag[i];
            if (c == '\\' && i + 1 < tag.Length)
            {
                i++;
                continue;
            }
            switch (c)
            {
                case '(':
                    parens++;
                    break;
                case '[':
                    brackets++;
                    break;
                case ')':
                    stray |= parens == 0;
                    parens = Math.Max(0, parens - 1);
                    break;
                case ']':
                    stray |= brackets == 0;
                    brackets = Math.Max(0, brackets - 1);
                    break;
                default:
                    fullComma |= FullWidthCommas.Contains(c);
                    fullPunctuation |= FullWidthPunctuation.Contains(c);
                    break;
            }
        }

        if (fullComma)
        {
            issues.Add(PromptIssue.FullWidthComma);
        }
        if (fullPunctuation)
        {
            issues.Add(PromptIssue.FullWidthPunctuation);
        }
        if (parens > 0 || brackets > 0)
        {
            issues.Add(PromptIssue.UnclosedBracket);
        }
        if (stray)
        {
            issues.Add(PromptIssue.StrayBracket);
        }
        if (LoraTags.Contains(tag))
        {
            issues.Add(PromptIssue.LoraTag);
        }
        return issues;
    }

    /// <summary>Text with the escapes resolved: a backslash in front of a
    /// bracket or another backslash is dropped and the character it protected
    /// kept. Anything else keeps its backslash, because '\m/' is a tag.</summary>
    public static string Unescape(string text)
    {
        if (!text.Contains('\\'))
        {
            return text;
        }
        var plain = new StringBuilder(text.Length);
        for (int i = 0; i < text.Length; i++)
        {
            if (text[i] == '\\' && i + 1 < text.Length && "()[]\\".Contains(text[i + 1]))
            {
                i++;
            }
            plain.Append(text[i]);
        }
        return plain.ToString();
    }

    /// <summary>The form two tags are compared in: case and inner spacing do
    /// not make '1girl' and '1  Girl' different tags.</summary>
    private static string Normalize(string tag)
    {
        var normalized = new StringBuilder(tag.Length);
        bool space = false;
        foreach (char c in tag)
        {
            if (char.IsWhiteSpace(c))
            {
                space = normalized.Length > 0;
                continue;
            }
            if (space)
            {
                normalized.Append(' ');
                space = false;
            }
            normalized.Append(c);
        }
        return normalized.ToString();
    }
}
