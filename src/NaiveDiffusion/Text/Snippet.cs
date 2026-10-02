using System.Text.Json;
using System.Text.Json.Serialization;
using NaiveDiffusion.Files;

namespace NaiveDiffusion.Text;

/// <summary>A piece of prompt kept under a name, for '{name}' in a prompt to
/// stand for — see <see cref="PromptTemplate"/>. Plain prompt text: tags with
/// their weights, '&lt;lora:…&gt;' tags, references to other snippets. Not a
/// list of tags, because what people keep is a run of text they would
/// otherwise type again, brackets and all.
///
/// Settable rather than init-only for the same reason <c>Preset</c> is: the
/// generated deserializer hands every property a value, so a missing string
/// would land as null instead of its initializer.</summary>
public sealed record Snippet
{
    public string Name { get; set; } = "";
    public string Text { get; set; } = "";
}

/// <summary>Snippets as files, one JSON document each — see
/// <see cref="NamedFiles{T}"/> for the folder's rules. A document whose
/// name is not one a reference can spell is passed over, since nothing
/// could ever refer to it, and refused when saving.</summary>
public static class SnippetFile
{
    public const string Extension = NamedFiles<Snippet>.Extension;

    private static readonly NamedFiles<Snippet> Files = new("snippet", Parse, Serialize, snippet => snippet.Name);

    public static string FileName(string name) => Files.FileName(name);

    public static string Serialize(Snippet snippet) =>
        JsonSerializer.Serialize(snippet, SnippetJsonContext.Default.Snippet);

    /// <summary>The snippet in a document, or null when it is not one or its
    /// name would not do. An unnamed document takes the name it is given,
    /// which the loader makes the file's own.</summary>
    public static Snippet? Parse(string json, string fallbackName = "")
    {
        try
        {
            Snippet? snippet = JsonSerializer.Deserialize(json, SnippetJsonContext.Default.Snippet);
            if (snippet is null)
            {
                return null;
            }
            string name = (snippet.Name ?? "").Trim();
            if (name.Length == 0)
            {
                name = fallbackName.Trim();
            }
            if (!PromptTemplate.IsValidName(name))
            {
                return null;
            }
            return snippet with { Name = name, Text = snippet.Text ?? "" };
        }
        catch (JsonException)
        {
            return null;
        }
    }

    public static IReadOnlyList<Snippet> LoadAll(string folder) => Files.LoadAll(folder);

    /// <summary>Throws as the file system does, and for a name a reference
    /// cannot spell — before anything is written.</summary>
    public static string Save(string folder, Snippet snippet)
    {
        if (!PromptTemplate.IsValidName(snippet.Name))
        {
            throw new ArgumentException($"'{snippet.Name}' is not a name a prompt can refer to");
        }
        return Files.Save(folder, snippet with { Name = snippet.Name.Trim() });
    }

    public static bool Delete(string folder, string name) => Files.Delete(folder, name);

    /// <summary>A resolver over a set of snippets, for
    /// <see cref="PromptTemplate.Expand"/>.</summary>
    public static Func<string, string?> Resolver(IEnumerable<Snippet> snippets)
    {
        var byName = new Dictionary<string, string>(PromptTemplate.NameComparer);
        foreach (Snippet snippet in snippets)
        {
            byName.TryAdd(snippet.Name, snippet.Text);
        }
        return name => byName.GetValueOrDefault(name);
    }
}

[JsonSourceGenerationOptions(WriteIndented = true)]
[JsonSerializable(typeof(Snippet))]
internal sealed partial class SnippetJsonContext : JsonSerializerContext
{
}
