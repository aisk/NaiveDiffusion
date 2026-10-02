using System.Globalization;
using NaiveDiffusion.Text;

namespace NaiveDiffusion.Tests;

/// <summary>The prompt-as-tags view: how a prompt is cut up, what a weight
/// group is and is not, and which typos get pointed at. Every case here is
/// one CLIP would have swallowed without a word.</summary>
public class PromptTagsTests
{
    [Test]
    public void SplitsOnCommasAndLineBreaksOutsideBrackets()
    {
        IReadOnlyList<PromptTag> tags = PromptTags.Parse("1girl, smile,\n(cat, dog:1.2),, ");
        Assert.That(tags.Select(tag => tag.Text),
            Is.EqualTo(new[] { "1girl", "smile", "(cat, dog:1.2)" }));
        Assert.That(tags.All(tag => tag.Issues.Count == 0));
    }

    [TestCase("(smile:1.3)", "smile", 1.3f)]
    [TestCase("( smile : 1.3 )", "smile", 1.3f)]
    [TestCase("(vocaloid)", "(vocaloid)", 1f)]
    [TestCase("(re:zero)", "(re:zero)", 1f)]
    [TestCase("(re:zero kara:1.2)", "re:zero kara", 1.2f)]
    [TestCase("(a:1.2) b", "(a:1.2) b", 1f)]
    [TestCase("(a:1.2) (b:1.3)", "(a:1.2) (b:1.3)", 1f)]
    [TestCase("((a:1.1):1.2)", "a", 1.32f)]
    [TestCase("(a:)", "(a:)", 1f)]
    [TestCase("(:1.2)", "(:1.2)", 1f)]
    [TestCase("(a:1,2)", "(a:1,2)", 1f)]
    [TestCase("plain", "plain", 1f)]
    public void UnwrapReadsOnlyTheOneWeightForm(string tag, string content, float weight)
    {
        (string unwrapped, float read) = PromptTags.Unwrap(tag);
        Assert.That(unwrapped, Is.EqualTo(content));
        Assert.That(read, Is.EqualTo(weight).Within(1e-5f));
    }

    [Test]
    [SetCulture("de-DE")]
    public void WithWeightWritesInvariantAndDropsWeightOne()
    {
        Assert.That(PromptTags.WithWeight("smile", 1.25f), Is.EqualTo("(smile:1.25)"));
        Assert.That(PromptTags.WithWeight("(smile:1.3)", 1f), Is.EqualTo("smile"));
        Assert.That(PromptTags.WithWeight("(smile:1.3)", 0.8f), Is.EqualTo("(smile:0.8)"));
    }

    [Test]
    public void WithWeightLeavesAnUnbalancedTagAlone()
    {
        // '(a):1.2)' would read back as literal text, so the tag comes back
        // unweighted for its author to fix first.
        Assert.That(PromptTags.WithWeight("a)", 1.2f), Is.EqualTo("a)"));
    }

    [Test]
    public void SegmentsCarryWeightsAndDropEscapes()
    {
        IReadOnlyList<PromptSegment> segments =
            PromptTags.Segments(@"1girl, (smile:1.3), hatsune miku \(vocaloid\)");
        Assert.That(segments, Is.EqualTo(new[]
        {
            new PromptSegment("1girl, ", 1f),
            new PromptSegment("smile, ", 1.3f),
            new PromptSegment("hatsune miku (vocaloid)", 1f),
        }));
        Assert.That(string.Concat(segments.Select(segment => segment.Text)),
            Is.EqualTo("1girl, smile, hatsune miku (vocaloid)"));
    }

    [Test]
    public void FlagsFullWidthCommasAndRepairsThem()
    {
        const string prompt = "1girl，smile、 blue eyes";
        IReadOnlyList<PromptTag> tags = PromptTags.Parse(prompt);
        Assert.That(tags, Has.Count.EqualTo(1));
        Assert.That(tags[0].Issues, Does.Contain(PromptIssue.FullWidthComma));
        Assert.That(PromptTags.NeedsRepair(prompt));
        Assert.That(PromptTags.Repair(prompt), Is.EqualTo("1girl, smile, blue eyes"));
        Assert.That(PromptTags.NeedsRepair(PromptTags.Repair(prompt)), Is.False);
    }

    /// <summary>A sentence for an encoder that takes it as written has no
    /// typos of the tag kind: its full-width commas, its brackets and a
    /// clause said twice are all just text. The cut is the same, so the
    /// pieces still show, and a LoRA tag is still the one thing pointed at.</summary>
    [Test]
    public void SentencesKeepTheirPunctuationAndOnlyLoraTagsAreFlagged()
    {
        const string prompt = "一只猫坐在窗台上，窗外下着雨（黄昏），一只猫坐在窗台上, <lora:rain:0.8>";
        IReadOnlyList<PromptTag> sentences = PromptTags.Parse(prompt, PromptStyle.Sentences);
        Assert.That(sentences.Select(tag => tag.Text),
            Is.EqualTo(new[] { "一只猫坐在窗台上，窗外下着雨（黄昏），一只猫坐在窗台上", "<lora:rain:0.8>" }));
        Assert.That(sentences[0].Issues, Is.Empty);
        Assert.That(sentences[1].Issues, Is.EqualTo(new[] { PromptIssue.LoraTag }));
        Assert.That(PromptTags.NeedsRepair(prompt, PromptStyle.Sentences), Is.False);

        IReadOnlyList<PromptTag> tags = PromptTags.Parse(prompt, PromptStyle.Tags);
        Assert.That(tags[0].Issues, Does.Contain(PromptIssue.FullWidthComma));
        Assert.That(tags[0].Issues, Does.Contain(PromptIssue.FullWidthPunctuation));
        Assert.That(PromptTags.NeedsRepair(prompt, PromptStyle.Tags));
    }

    [Test]
    public void FlagsBracketsPunctuationAndDuplicates()
    {
        // The unclosed one has to come last: a '(' swallows every comma
        // after it, which is the point of it being flagged.
        IReadOnlyList<PromptTag> tags =
            PromptTags.Parse("close), 1girl, 1GIRL , （cat）, <lora:x:0.8>, (open");
        Assert.That(tags.Select(tag => tag.Issues), Is.EqualTo(new[]
        {
            new[] { PromptIssue.StrayBracket },
            Array.Empty<PromptIssue>(),
            new[] { PromptIssue.Duplicate },
            new[] { PromptIssue.FullWidthPunctuation },
            new[] { PromptIssue.LoraTag },
            new[] { PromptIssue.UnclosedBracket },
        }));
    }

    [Test]
    public void EscapedBracketsAreLiteralEverywhere()
    {
        IReadOnlyList<PromptTag> tags = PromptTags.Parse(@"\(a, b\), c");
        Assert.That(tags.Select(tag => tag.Text), Is.EqualTo(new[] { @"\(a", @"b\)", "c" }));
        Assert.That(tags.All(tag => tag.Issues.Count == 0));
    }

    [Test]
    public void SplitTypedKeepsThePieceUnderTheCaret()
    {
        Assert.That(PromptTags.SplitTyped("a, b, c", out string pending),
            Is.EqualTo(new[] { "a", "b" }));
        Assert.That(pending, Is.EqualTo(" c"));

        Assert.That(PromptTags.SplitTyped("(cat, ", out pending), Is.Empty);
        Assert.That(pending, Is.EqualTo("(cat, "));
    }

    [Test]
    public void JoinDropsBlanks()
    {
        Assert.That(PromptTags.Join(new[] { " a ", "", "b", "  " }), Is.EqualTo("a, b"));
    }
}
