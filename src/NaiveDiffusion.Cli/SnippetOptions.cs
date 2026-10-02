using NaiveDiffusion.Text;

namespace NaiveDiffusion.Cli;

/// <summary>Where '{name}' in a prompt gets its text from on the command
/// line: --set name=text, any number of times, and --snippet-dir, a folder
/// of snippet JSON files. A --set wins over a file of the same name. A name
/// nothing answers to is an error, not a prompt with braces in it.</summary>
internal static class SnippetOptions
{
    public static readonly string[] ValueOptions = { "--set", "--snippet-dir" };

    public static Func<string, string?> Resolver(CommandLine line)
    {
        var snippets = new List<Snippet>();
        foreach (string text in line.Values("--set"))
        {
            int equals = text.IndexOf('=');
            string name = equals < 0 ? text : text[..equals];
            if (equals < 0 || !PromptTemplate.IsValidName(name))
            {
                CommandLine.Fail($"--set {text}: name=text, the name letters, digits, spaces, - and _" +
                                 (PromptTemplate.NameComparer.Equals(name.Trim(), PromptTemplate.StepName)
                                     ? $"; {PromptTemplate.StepName} is the sequence's own, given with --step"
                                     : ""));
            }
            snippets.Add(new Snippet { Name = name.Trim(), Text = text[(equals + 1)..] });
        }
        foreach (string folder in line.Values("--snippet-dir"))
        {
            if (!Directory.Exists(folder))
            {
                CommandLine.Fail($"--snippet-dir {folder}: no such folder");
            }
            snippets.AddRange(SnippetFile.LoadAll(folder));
        }
        return SnippetFile.Resolver(snippets);
    }

    /// <summary>The text with its references expanded, or exit code 2 with
    /// the first name that could not be.</summary>
    public static string Expand(string text, Func<string, string?> resolve, string what,
        string? keep = null)
    {
        Expansion expansion = PromptTemplate.Expand(text, resolve, keep);
        if (expansion.Missing.Count > 0)
        {
            string name = expansion.Missing[0];
            return CommandLine.Fail<string>(PromptTemplate.NameComparer.Equals(name, PromptTemplate.StepName)
                ? $"the {what} has {PromptTemplate.Format(name)} but no --step was given"
                : $"the {what} refers to {PromptTemplate.Format(name)}, which nothing defines; " +
                  $"give it with --set {name}=text or a folder with --snippet-dir");
        }
        if (expansion.Cyclic.Count > 0)
        {
            return CommandLine.Fail<string>(
                $"{PromptTemplate.Format(expansion.Cyclic[0])} refers to itself, through however many others");
        }
        return expansion.Text;
    }
}
