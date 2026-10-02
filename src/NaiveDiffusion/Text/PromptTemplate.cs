using System.Text;
using System.Text.RegularExpressions;

namespace NaiveDiffusion.Text;

/// <summary>What expanding a prompt came to: the text with every reference
/// it could resolve replaced, and the names it could not — the ones nothing
/// answered to and the ones that led back to themselves. A prompt with either
/// list non-empty is not one to generate from: the reference is still in the
/// text, and CLIP would read it as the word in braces.</summary>
public sealed record Expansion(string Text, IReadOnlyList<string> Missing,
    IReadOnlyList<string> Cyclic)
{
    public bool Complete => Missing.Count == 0 && Cyclic.Count == 0;
}

/// <summary>References inside a prompt: '{name}' stands for a piece of prompt
/// kept under that name — a character with its LoRA tag, a setting, a style
/// — so the piece is written once and put wherever it is wanted, in the
/// position it is wanted in. What a name resolves to is the caller's: a
/// front end may keep a library of snippets, the CLI takes them from
/// options. A snippet may refer to others; one that refers to itself,
/// however indirectly, is reported rather than followed forever.
///
/// One name is reserved: <see cref="StepName"/>. '{step}' is where each
/// step of a sequence goes — see <see cref="WithStep"/> — and is never a
/// snippet.
///
/// A name is letters, digits, spaces, '-' and '_'. Braces around anything
/// else are not a reference and pass through as text, so a NovelAI prompt's
/// '{tag:1.2}' is not mistaken for one; a bare '{masterpiece}' pasted from
/// one is, and is reported as a snippet nobody has, which is the truthful
/// answer — this library never read those braces as emphasis.</summary>
public static partial class PromptTemplate
{
    public const string StepName = "step";

    /// <summary>Names compare without case: a library keeps one snippet per
    /// spelling-insensitive name, as the files under it would anyway.</summary>
    public static readonly StringComparer NameComparer = StringComparer.OrdinalIgnoreCase;

    [GeneratedRegex(@"\{\s*([\p{L}\p{N}_\-][\p{L}\p{N}_\- ]*?)\s*\}", RegexOptions.CultureInvariant)]
    private static partial Regex Reference();

    [GeneratedRegex(@"^[\p{L}\p{N}_\-][\p{L}\p{N}_\- ]*$", RegexOptions.CultureInvariant)]
    private static partial Regex Name();

    /// <summary>Whether the text would do as a snippet's name: something a
    /// reference can spell, and not the reserved one.</summary>
    public static bool IsValidName(string name) =>
        Name().IsMatch(name.Trim()) && !NameComparer.Equals(name.Trim(), StepName);

    /// <summary>The reference written for a name.</summary>
    public static string Format(string name) => "{" + name.Trim() + "}";

    /// <summary>Whether one tag is nothing but a reference, and to what.</summary>
    public static bool IsReference(string tag, out string name)
    {
        Match match = Reference().Match(tag.Trim());
        if (match.Success && match.Index == 0 && match.Length == tag.Trim().Length)
        {
            name = match.Groups[1].Value;
            return true;
        }
        name = "";
        return false;
    }

    /// <summary>Every name the text refers to, once each, in order of first
    /// use. Only the text's own: what the names expand to is not read.</summary>
    public static IReadOnlyList<string> References(string text)
    {
        var names = new List<string>();
        foreach (Match match in Reference().Matches(text))
        {
            string name = match.Groups[1].Value;
            if (!names.Contains(name, NameComparer))
            {
                names.Add(name);
            }
        }
        return names;
    }

    /// <summary>The text with every reference replaced by what
    /// <paramref name="resolve"/> gives for its name, and those replacements
    /// expanded in turn. A name it gives null for is left as written and
    /// reported in <see cref="Expansion.Missing"/>; one that is already being
    /// expanded — a snippet that names itself, or two that name each other —
    /// is left as written and reported in <see cref="Expansion.Cyclic"/>.
    /// <paramref name="keep"/> is a name to leave alone without reporting it,
    /// which is how '{step}' survives until the sequence fills it.</summary>
    public static Expansion Expand(string text, Func<string, string?> resolve, string? keep = null)
    {
        var missing = new List<string>();
        var cyclic = new List<string>();
        var stack = new List<string>();
        string expanded = ExpandInto(text, resolve, keep, stack, missing, cyclic);
        return new Expansion(expanded, missing, cyclic);
    }

    private static string ExpandInto(string text, Func<string, string?> resolve, string? keep,
        List<string> stack, List<string> missing, List<string> cyclic)
    {
        if (!Reference().IsMatch(text))
        {
            return text;
        }
        var result = new StringBuilder(text.Length);
        int copied = 0;
        foreach (Match match in Reference().Matches(text))
        {
            string name = match.Groups[1].Value;
            result.Append(text, copied, match.Index - copied);
            copied = match.Index + match.Length;

            if (keep is not null && NameComparer.Equals(name, keep))
            {
                result.Append(match.Value);
                continue;
            }
            if (stack.Contains(name, NameComparer))
            {
                Note(cyclic, name);
                result.Append(match.Value);
                continue;
            }
            string? replacement = resolve(name);
            if (replacement is null)
            {
                Note(missing, name);
                result.Append(match.Value);
                continue;
            }
            stack.Add(name);
            result.Append(ExpandInto(replacement, resolve, keep, stack, missing, cyclic));
            stack.RemoveAt(stack.Count - 1);
        }
        result.Append(text, copied, text.Length - copied);
        return result.ToString();
    }

    private static void Note(List<string> names, string name)
    {
        if (!names.Contains(name, NameComparer))
        {
            names.Add(name);
        }
    }

    /// <summary>Whether the text has a place for a step.</summary>
    public static bool HasStep(string text) =>
        References(text).Contains(StepName, NameComparer);

    /// <summary>The prompt with one step of a sequence in it: in place of
    /// every '{step}' when the prompt has one, and otherwise on the end, after
    /// a comma. A prompt of nothing but the step is the step itself, so a
    /// sequence of whole prompts needs no prompt above it.</summary>
    public static string WithStep(string prompt, string step)
    {
        if (HasStep(prompt))
        {
            return Reference().Replace(prompt, match =>
                NameComparer.Equals(match.Groups[1].Value, StepName) ? step : match.Value);
        }
        string trimmed = prompt.Trim();
        if (trimmed.Length == 0)
        {
            return step;
        }
        if (step.Trim().Length == 0)
        {
            return prompt;
        }
        return trimmed.EndsWith(',') ? trimmed + " " + step : trimmed + ", " + step;
    }
}
