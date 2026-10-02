using NaiveDiffusion.Text;
using NaiveDiffusion.Text.Clip;

namespace NaiveDiffusion.Tests;

/// <summary>The BPE tokenizer against ids transformers' CLIPTokenizer gives,
/// and the chunking a long prompt gets.</summary>
public class ClipTokenizerTests
{
    private const uint Start = 49406, End = 49407;

    /// <summary>CLIP-L pads with the end marker; CLIP-G pads with "!", id 0.</summary>
    private static readonly ClipTokenizer ClipL = new("<|endoftext|>");
    private static readonly ClipTokenizer ClipG = new("!");

    [Test]
    public void KnownIds()
    {
        // The ids transformers gives for the same text.
        Assert.That(Body(ClipL, "a photo of a cat"), Is.EqualTo(new uint[] { 320, 1125, 539, 320, 2368 }));
        Assert.That(Body(ClipL, "Hello,  WORLD!"), Is.EqualTo(new uint[] { 3306, 267, 1002, 256 }));
        Assert.That(Body(ClipL, "1girl"), Is.EqualTo(new uint[] { 272, 1611 }));
    }

    [Test]
    public void FramesAndPadsPerTower()
    {
        TokenChunk chunk = ClipL.EncodeChunks("a cat").Single();
        Assert.That(chunk.Ids, Has.Length.EqualTo(ClipTokenizer.MaxTokens));
        Assert.That(chunk.Ids[0], Is.EqualTo(Start));
        Assert.That(chunk.Ids[3], Is.EqualTo(End));
        Assert.That(chunk.Ids[4..], Is.All.EqualTo(End));
        Assert.That(chunk.Weights, Is.All.EqualTo(1f));

        chunk = ClipG.EncodeChunks("a cat").Single();
        Assert.That(chunk.Ids[3], Is.EqualTo(End));
        Assert.That(chunk.Ids[4..], Is.All.EqualTo(0u));

        // An empty prompt is still one chunk: start, end, padding.
        chunk = ClipL.EncodeChunks("").Single();
        Assert.That(chunk.Ids[0], Is.EqualTo(Start));
        Assert.That(chunk.Ids[1..], Is.All.EqualTo(End));
    }

    [Test]
    public void WeightsFollowTheSegments()
    {
        TokenChunk chunk = ClipL.EncodeChunks(PromptTags.Segments("1girl, (smile:1.3), cat")).Single();
        // 1 girl , smile , cat — the separator rides with the tag before it,
        // so the comma after smile carries smile's weight.
        Assert.That(chunk.Ids[1..7], Is.EqualTo(new uint[] { 272, 1611, 267, 3490, 267, 2368 }));
        Assert.That(chunk.Weights[1..7], Is.EqualTo(new[] { 1f, 1f, 1f, 1.3f, 1.3f, 1f }));
        // The frame and the padding are never weighted.
        Assert.That(chunk.Weights[0], Is.EqualTo(1f));
        Assert.That(chunk.Weights[7..], Is.All.EqualTo(1f));
    }

    [Test]
    public void LongPromptsCutOnTheTokenGrid()
    {
        string prompt = string.Join(" ", Enumerable.Repeat("cat", 100));
        List<TokenChunk> chunks = ClipL.EncodeChunks(prompt);
        Assert.That(chunks, Has.Count.EqualTo(2));
        Assert.That(chunks[0].Ids[1..76], Is.All.EqualTo(2368u));
        Assert.That(chunks[0].Ids[76], Is.EqualTo(End));
        Assert.That(chunks[1].Ids[1..26], Is.All.EqualTo(2368u));
        Assert.That(chunks[1].Ids[26], Is.EqualTo(End));

        // Eight chunks is where a prompt stops being read.
        prompt = string.Join(" ", Enumerable.Repeat("cat", 1000));
        Assert.That(ClipL.EncodeChunks(prompt), Has.Count.EqualTo(ClipTokenizer.MaxChunks));
    }

    private static uint[] Body(ClipTokenizer tokenizer, string text)
    {
        uint[] ids = tokenizer.EncodeChunks(text).Single().Ids;
        int end = Array.IndexOf(ids, End, 1);
        return ids[1..end];
    }
}
