using NaiveDiffusion.Tests.Fixtures;
using NaiveDiffusion.Models.Anima;
using NaiveDiffusion.Text.Qwen;
using NaiveDiffusion.Text.T5;

namespace NaiveDiffusion.Tests;

/// <summary>Both of Anima's tokenizers, and the way the prompt is cut for
/// them, against ids written by ComfyUI's AnimaTokenizer for the same
/// prompts (Fixtures/AnimaTokens.json): the Qwen ids are what Qwen3 reads,
/// the T5 ids what the adapter embeds, and a prompt cut differently is a
/// different image. The one known divergence is kept out of the fixture: a
/// bare '(word)' is emphasis at 1.1 over there and literal text here.</summary>
public class AnimaTokenizerTests
{
    public sealed record Sample(string Prompt, int[] Qwen, int[] T5, float[] T5Weights);

    private static IEnumerable<Sample> Samples() =>
        TokenFixture.Entries("AnimaTokens.json").Select(entry => new Sample(
            entry.GetProperty("prompt").GetString()!,
            entry.GetProperty("qwen").EnumerateArray().Select(pair => pair[0].GetInt32()).ToArray(),
            entry.GetProperty("t5").EnumerateArray().Select(pair => pair[0].GetInt32()).ToArray(),
            entry.GetProperty("t5").EnumerateArray().Select(pair => (float)pair[1].GetDouble()).ToArray()));

    private static readonly Qwen2Tokenizer Qwen = Qwen2Tokenizer.Shared;
    private static readonly T5Tokenizer T5 = T5Tokenizer.Shared;

    private static IEnumerable<TestCaseData> Prompts() => TokenFixture.Cases(Samples(), sample => sample.Prompt);

    [TestCaseSource(nameof(Prompts))]
    public void ThePromptTokenizesAsComfyUiDoes(Sample sample)
    {
        AnimaTokens tokens = AnimaPrompt.Tokenize(sample.Prompt, Qwen, T5);
        Assert.That(tokens.Qwen, Is.EqualTo(sample.Qwen), "qwen");
        Assert.That(tokens.T5, Is.EqualTo(sample.T5), "t5");
        Assert.That(tokens.T5Weights, Is.EqualTo(sample.T5Weights).Within(1e-6f), "t5 weights");
    }

    [Test]
    public void TheFixtureHasTwentyPrompts() =>
        Assert.That(Samples().Count(), Is.EqualTo(20));

    [Test]
    public void AnEmptyPromptIsThePadTokenAndTheEndMarker()
    {
        AnimaTokens tokens = AnimaPrompt.Tokenize("", Qwen, T5);
        Assert.That(tokens.Qwen, Is.EqualTo(new[] { Qwen2Tokenizer.PadToken }));
        Assert.That(tokens.T5, Is.EqualTo(new[] { T5Tokenizer.EndToken }));
    }

    [Test]
    public void AWeightRidesOnTheT5TokensOfItsSegmentOnly()
    {
        AnimaTokens tokens = AnimaPrompt.Tokenize("1girl, (smile:1.3), city", Qwen, T5);
        int smile = T5.Encode("smile").Count;
        Assert.That(tokens.T5Weights.Count(weight => weight == 1.3f), Is.EqualTo(smile));
        Assert.That(tokens.T5Weights[^1], Is.EqualTo(1f));
    }
}
