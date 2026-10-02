using NaiveDiffusion.Text;

namespace NaiveDiffusion.Tests;

/// <summary>Snippets as files: what goes in comes back out, a folder's worth
/// loads with the bad ones passed over, saving replaces by name whatever
/// the spelling, and a name a reference cannot spell is refused.</summary>
public class SnippetFileTests
{
    private string _folder = "";

    [SetUp]
    public void MakeFolder()
    {
        _folder = Path.Combine(Path.GetTempPath(), "NaiveDiffusion.SnippetFileTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_folder);
    }

    [TearDown]
    public void RemoveFolder()
    {
        try
        {
            Directory.Delete(_folder, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    [Test]
    public void RoundTripsThroughJson()
    {
        var snippet = new Snippet { Name = "miku", Text = "hatsune miku, (twintails:1.2), <lora:miku_il:0.8>\n{light}" };
        Snippet? read = SnippetFile.Parse(SnippetFile.Serialize(snippet));
        Assert.That(read, Is.EqualTo(snippet));
    }

    [Test]
    public void ParseFillsInTheFileNameAndRefusesABadName()
    {
        Assert.That(SnippetFile.Parse("""{"Text": "a, b"}""", "from file")!.Name, Is.EqualTo("from file"));
        Assert.That(SnippetFile.Parse("""{"Name": "step", "Text": "a"}"""), Is.Null);
        Assert.That(SnippetFile.Parse("""{"Name": "a{b}", "Text": "a"}"""), Is.Null);
        Assert.That(SnippetFile.Parse("""{"Name": " a ", "Text": null}""")!, Is.EqualTo(new Snippet { Name = "a" }));
        Assert.That(SnippetFile.Parse("not json"), Is.Null);
        Assert.That(SnippetFile.Parse("null"), Is.Null);
    }

    [Test]
    public void SaveLoadDeleteByName()
    {
        SnippetFile.Save(_folder, new Snippet { Name = "Miku", Text = "one" });
        SnippetFile.Save(_folder, new Snippet { Name = "classroom", Text = "two" });
        // Another spelling of a saved name replaces it, file included.
        SnippetFile.Save(_folder, new Snippet { Name = "miku", Text = "three" });
        File.WriteAllText(Path.Combine(_folder, "junk.json"), "{");
        File.WriteAllText(Path.Combine(_folder, "reserved.json"), """{"Name": "step", "Text": "x"}""");

        IReadOnlyList<Snippet> loaded = SnippetFile.LoadAll(_folder);
        Assert.That(loaded.Select(snippet => snippet.Name), Is.EqualTo(new[] { "classroom", "miku" }));
        Assert.That(loaded[1].Text, Is.EqualTo("three"));
        Assert.That(Directory.GetFiles(_folder, "*.json"), Has.Length.EqualTo(4));

        Assert.That(SnippetFile.Delete(_folder, "MIKU"), Is.True);
        Assert.That(SnippetFile.Delete(_folder, "miku"), Is.False);
        Assert.That(SnippetFile.LoadAll(_folder).Select(snippet => snippet.Name), Is.EqualTo(new[] { "classroom" }));
    }

    [Test]
    public void SaveRefusesANameNothingCouldReferTo()
    {
        Assert.Throws<ArgumentException>(() => SnippetFile.Save(_folder, new Snippet { Name = "step", Text = "x" }));
        Assert.Throws<ArgumentException>(() => SnippetFile.Save(_folder, new Snippet { Name = "", Text = "x" }));
        Assert.That(Directory.GetFiles(_folder), Is.Empty);
    }

    [Test]
    public void ResolverAnswersByNameWithoutCase()
    {
        Func<string, string?> resolve = SnippetFile.Resolver(new[]
        {
            new Snippet { Name = "Miku", Text = "one" },
            new Snippet { Name = "miku", Text = "shadowed" },
        });
        Assert.That(resolve("MIKU"), Is.EqualTo("one"));
        Assert.That(resolve("nobody"), Is.Null);
    }

    [Test]
    public void MissingFolderHoldsNone()
    {
        Assert.That(SnippetFile.LoadAll(Path.Combine(_folder, "nowhere")), Is.Empty);
        Assert.That(SnippetFile.Delete(Path.Combine(_folder, "nowhere"), "x"), Is.False);
    }
}
