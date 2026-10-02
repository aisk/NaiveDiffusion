using NaiveDiffusion.Text;

namespace NaiveDiffusion.Tests;

/// <summary>'{name}' in a prompt: what counts as a reference, how one expands
/// through a library of snippets, what happens to a name nobody has or one
/// that names itself, and where a sequence's step lands.</summary>
public class PromptTemplateTests
{
    private static readonly Func<string, string?> Library = SnippetFile.Resolver(new[]
    {
        new Snippet { Name = "miku", Text = "hatsune miku, twintails, <lora:miku_il:0.8>" },
        new Snippet { Name = "classroom", Text = "classroom, window, {light}" },
        new Snippet { Name = "light", Text = "afternoon light" },
        new Snippet { Name = "loop a", Text = "{loop b}" },
        new Snippet { Name = "loop b", Text = "{loop a}" },
        new Snippet { Name = "self", Text = "x, {self}" },
    });

    [TestCase("{miku}", "miku")]
    [TestCase("  { Miku } ", "Miku")]
    [TestCase("{loop a}", "loop a")]
    [TestCase("{a-b_c1}", "a-b_c1")]
    public void AWholeTagCanBeAReference(string tag, string name)
    {
        Assert.That(PromptTemplate.IsReference(tag, out string found), Is.True);
        Assert.That(found, Is.EqualTo(name));
    }

    [TestCase("{miku:1.2}")]
    [TestCase("{}")]
    [TestCase("{ }")]
    [TestCase("{(miku)}")]
    [TestCase("miku {classroom}")]
    [TestCase("1girl")]
    public void OtherBracesAreText(string tag)
    {
        Assert.That(PromptTemplate.IsReference(tag, out _), Is.False);
    }

    [Test]
    public void ReferencesAreListedOnceEachInOrder()
    {
        Assert.That(PromptTemplate.References("{classroom}, {miku}, {Classroom}, {step}"),
            Is.EqualTo(new[] { "classroom", "miku", "step" }));
        Assert.That(PromptTemplate.References("nothing here, {a:1.1}"), Is.Empty);
    }

    [Test]
    public void ExpandFollowsReferencesThroughSnippets()
    {
        Expansion result = PromptTemplate.Expand("1girl, {classroom}, {miku}, smile", Library);
        Assert.That(result.Complete, Is.True);
        Assert.That(result.Text, Is.EqualTo(
            "1girl, classroom, window, afternoon light, hatsune miku, twintails, <lora:miku_il:0.8>, smile"));
    }

    [Test]
    public void ExpandLeavesAnUnknownNameAndReportsIt()
    {
        Expansion result = PromptTemplate.Expand("{miku}, {nobody}, {nobody}", Library);
        Assert.That(result.Complete, Is.False);
        Assert.That(result.Missing, Is.EqualTo(new[] { "nobody" }));
        Assert.That(result.Cyclic, Is.Empty);
        Assert.That(result.Text, Is.EqualTo("hatsune miku, twintails, <lora:miku_il:0.8>, {nobody}, {nobody}"));
    }

    [Test]
    public void ExpandStopsAtACycleAndReportsIt()
    {
        Expansion result = PromptTemplate.Expand("{loop a}, {self}", Library);
        Assert.That(result.Cyclic, Is.EqualTo(new[] { "loop a", "self" }));
        Assert.That(result.Missing, Is.Empty);
        // The reference that closed the loop stays as written, one level in.
        Assert.That(result.Text, Is.EqualTo("{loop a}, x, {self}"));
    }

    [Test]
    public void ExpandKeepsTheStepForTheSequence()
    {
        Expansion result = PromptTemplate.Expand("{miku}, {step}", Library, keep: PromptTemplate.StepName);
        Assert.That(result.Complete, Is.True);
        Assert.That(result.Text, Is.EqualTo("hatsune miku, twintails, <lora:miku_il:0.8>, {step}"));

        // Not kept, it is a name like any other, and nobody has it.
        Assert.That(PromptTemplate.Expand("{step}", Library).Missing, Is.EqualTo(new[] { "step" }));
    }

    [Test]
    public void ExpandIsCaseInsensitiveAndTolerantOfSpaces()
    {
        Assert.That(PromptTemplate.Expand("{ LIGHT }", Library).Text, Is.EqualTo("afternoon light"));
    }

    [TestCase("a, {step}, b", "sitting", "a, sitting, b")]
    [TestCase("a, {step}, {Step}", "x", "a, x, x")]
    [TestCase("a, b", "sitting", "a, b, sitting")]
    [TestCase("a, b,", "sitting", "a, b, sitting")]
    [TestCase("a, b\n", "sitting", "a, b, sitting")]
    [TestCase("", "sitting", "sitting")]
    [TestCase("a, b", "", "a, b")]
    public void WithStepReplacesOrAppends(string prompt, string step, string expected)
    {
        Assert.That(PromptTemplate.WithStep(prompt, step), Is.EqualTo(expected));
    }

    [TestCase("miku", true)]
    [TestCase("loop a", true)]
    [TestCase("  a  ", true)]
    [TestCase("日本語", true)]
    [TestCase("", false)]
    [TestCase("step", false)]
    [TestCase("STEP", false)]
    [TestCase("a{b}", false)]
    [TestCase("a:b", false)]
    public void NamesAreWhatAReferenceCanSpell(string name, bool valid)
    {
        Assert.That(PromptTemplate.IsValidName(name), Is.EqualTo(valid));
    }
}
