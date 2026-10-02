using NaiveDiffusion.Tests.Fixtures;
using NaiveDiffusion.Models.QwenImage;
using NaiveDiffusion.Text.Qwen;

namespace NaiveDiffusion.Tests;

/// <summary>Qwen-Image 2.1's chat template around a prompt, against the ids
/// ComfyUI's QwenImage21Tokenizer writes for the same prompts
/// (Fixtures/QwenImageTokens.json): the special tokens as single ids, the
/// text between them cut as transformers cuts it, and the system turn's
/// length, which is how many rows are dropped from the context. The one
/// known divergence is kept out of the fixture: an empty prompt, which
/// ComfyUI pads to a space and this library leaves empty, as diffusers does.</summary>
public class QwenImagePromptTests
{
    public sealed record Sample(string Prompt, int[] Ids, int Drop);

    private static IEnumerable<Sample> Samples() =>
        TokenFixture.Entries("QwenImageTokens.json").Select(entry => new Sample(
            entry.GetProperty("prompt").GetString()!,
            entry.GetProperty("ids").EnumerateArray().Select(id => id.GetInt32()).ToArray(),
            entry.GetProperty("drop").GetInt32()));

    private static readonly Qwen2Tokenizer Tokenizer = Qwen2Tokenizer.Shared;

    private static IEnumerable<TestCaseData> Prompts() => TokenFixture.Cases(Samples(), sample => sample.Prompt);

    [TestCaseSource(nameof(Prompts))]
    public void TheTemplateTokenizesAsComfyUiDoes(Sample sample)
    {
        (int[] ids, int drop) = QwenImagePrompt.Tokenize(sample.Prompt, Tokenizer);
        Assert.That(ids, Is.EqualTo(sample.Ids), "ids");
        Assert.That(drop, Is.EqualTo(sample.Drop), "the system turn");
        Assert.That(QwenImagePrompt.ContextRows(sample.Prompt, Tokenizer),
            Is.EqualTo(sample.Ids.Length - sample.Drop));
    }

    [Test]
    public void TheFixtureHasSixteenPrompts() =>
        Assert.That(Samples().Count(), Is.EqualTo(16));

    [Test]
    public void AnEmptyPromptIsTheTemplateAroundNothing()
    {
        (int[] ids, int drop) = QwenImagePrompt.Tokenize("", Tokenizer);
        // The system turn, then <|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n.
        Assert.That(drop, Is.EqualTo(14));
        Assert.That(ids.Skip(drop), Is.EqualTo(new[]
        {
            Qwen2Tokenizer.ImStart, 872, 198, Qwen2Tokenizer.ImEnd, 198, Qwen2Tokenizer.ImStart, 77091, 198,
        }));
    }

    [Test]
    public void SpecialTokensAreOneIdEachAndOnlyTheKnownOnes()
    {
        Assert.That(Tokenizer.EncodeWithSpecials("<|im_start|>"), Is.EqualTo(new[] { Qwen2Tokenizer.ImStart }));
        Assert.That(Tokenizer.EncodeWithSpecials("a<|im_end|>b"),
            Is.EqualTo(new[] { 64, Qwen2Tokenizer.ImEnd, 65 }));
        Assert.That(Tokenizer.EncodeWithSpecials("<|not_one|>"), Is.EqualTo(Tokenizer.Encode("<|not_one|>")));
        Assert.That(Tokenizer.Encode("<|im_start|>"), Has.Count.GreaterThan(1),
            "Encode never emits a special token");
    }
}
