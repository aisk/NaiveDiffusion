using NaiveDiffusion.Graph;
using NaiveDiffusion.Models.Anima;
using NaiveDiffusion.Models.QwenImage;
using NaiveDiffusion.Models.Sdxl;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Tensors;
using NaiveDiffusion.Tests.Fixtures;

namespace NaiveDiffusion.Tests;

/// <summary>What the two transformers are fed from the CPU, which the GPU
/// comparisons against ComfyUI check only as a whole: the sinusoid of the
/// time, the rotary tables' placement of text and image tokens, and
/// Qwen-Image's per-branch masks — the causal table closed over the
/// padding, the count of keys the image rows may read — and the padding
/// of a shorter branch to the longer one. And the way a conditioner takes
/// rows from a file in place of its text encoder.</summary>
public class TransformerInputsTests
{
    [Test]
    public void TheSinusoidIsHalfCosinesThenHalfSines()
    {
        float[] at0 = Transformer.Sinusoid(0f, 8);
        Assert.That(at0, Is.EqualTo(new[] { 1f, 1f, 1f, 1f, 0f, 0f, 0f, 0f }));
        float[] at1 = Transformer.Sinusoid(1f, 8);
        // Frequencies 10000^(-i/4): 1, 0.1, 0.01, 0.001.
        Assert.That(at1[0], Is.EqualTo(MathF.Cos(1f)).Within(1e-6f));
        Assert.That(at1[1], Is.EqualTo(MathF.Cos(0.1f)).Within(1e-6f));
        Assert.That(at1[4], Is.EqualTo(MathF.Sin(1f)).Within(1e-6f));
        Assert.That(at1[7], Is.EqualTo(MathF.Sin(0.001f)).Within(1e-6f));
        // Qwen-Image embeds a thousand times the level.
        Assert.That(QwenImageDit.TimeEmbedding(0.5, 8), Is.EqualTo(Transformer.Sinusoid(500f, 8)));
    }

    /// <summary>Text token p sits at (p, p, p); image cell (i, j) at (T,
    /// i − ⌈h/2⌉, j − ⌈w/2⌉) with T the branch's own text length.</summary>
    [Test]
    public void QwenImagesRotaryTablesPlaceTextAfterTheImage()
    {
        const int rows = 2, columns = 3, textLength = 5, contextRows = 7;
        (HostTensor cos, HostTensor sin) = QwenImageDit.RopeTables(rows, columns, textLength, contextRows);
        int tokens = rows * columns + contextRows;
        Assert.That(cos.Shape, Is.EqualTo(new[] { 1, tokens, 1, 64 }));
        Assert.That(sin.Shape, Is.EqualTo(new[] { 1, tokens, 1, 64 }));
        float[] c = cos.ToFloats(), s = sin.ToFloats();

        // Pair 0 turns with the first axis at frequency 1: the image's
        // frame is the text length, a text token's is its position.
        Assert.That(c[0], Is.EqualTo(Math.Cos(textLength)).Within(1e-6));
        Assert.That(s[rows * columns * 64], Is.EqualTo(0f), "text token 0 at position 0");
        Assert.That(s[(rows * columns + 3) * 64], Is.EqualTo(Math.Sin(3)).Within(1e-6));
        // Pair 8 is the second axis at frequency 1: the row, centred.
        // Two rows: ⌈2/2⌉ = 1, so the rows sit at -1 and 0.
        Assert.That(s[0 * 64 + 8], Is.EqualTo(Math.Sin(-1)).Within(1e-6));
        Assert.That(s[columns * 64 + 8], Is.EqualTo(0f).Within(1e-6));
        // Pair 36 is the third axis at frequency 1: the column, centred at
        // ⌈3/2⌉ = 2, so -2, -1, 0.
        Assert.That(s[0 * 64 + 36], Is.EqualTo(Math.Sin(-2)).Within(1e-6));
        Assert.That(s[2 * 64 + 36], Is.EqualTo(0f).Within(1e-6));
        // A text token's three axes all count its position.
        int p = rows * columns + 4;
        Assert.That(s[p * 64 + 8], Is.EqualTo(Math.Sin(4)).Within(1e-6));
        Assert.That(s[p * 64 + 36], Is.EqualTo(Math.Sin(4)).Within(1e-6));
    }

    [Test]
    public void AnimasRotaryTablesLeaveTheTemporalPairsStill()
    {
        (HostTensor cos, HostTensor sin) = AnimaDit.RopeTables(2, 2, 128);
        Assert.That(cos.Shape, Is.EqualTo(new[] { 1, 4, 1, 64 }));
        float[] c = cos.ToFloats(), s = sin.ToFloats();
        // 22 temporal pairs at cos 1, sin 0 for every token.
        for (int token = 0; token < 4; token++)
        {
            Assert.That(c.Skip(token * 64).Take(22), Is.All.EqualTo(1f));
            Assert.That(s.Skip(token * 64).Take(22), Is.All.EqualTo(0f));
        }
        // Row 1's first spatial pair turns by one radian; column 1's by one
        // radian in the last 21 pairs.
        Assert.That(s[2 * 64 + 22], Is.EqualTo(MathF.Sin(1f)).Within(1e-6f), "row 1, column 0");
        Assert.That(s[1 * 64 + 22 + 21], Is.EqualTo(MathF.Sin(1f)).Within(1e-6f), "row 0, column 1");
        Assert.That(s[1 * 64 + 22], Is.EqualTo(0f), "row 0 does not turn with the row");
    }

    /// <summary>The shorter branch is padded to the longer's rows; its
    /// causal table stops at its own real tokens, its key count is the
    /// image's tokens plus its own real text, and its rotary tables place
    /// the image at its own text length.</summary>
    [Test]
    public void QwenImagesBranchesArePaddedAndMaskedToTheirOwnLength()
    {
        const int width = 4, latentRows = 2, latentColumns = 2;
        float[] negative = Enumerable.Repeat(1f, 2 * width).ToArray();
        float[] positive = Enumerable.Repeat(2f, 3 * width).ToArray();
        var conditioning = new QwenImageConditioning(new[] { negative, positive }, width, latentRows, latentColumns);

        Assert.That(conditioning.Branches, Is.EqualTo(2));
        Assert.That(conditioning.Rows, Is.EqualTo(3));
        Assert.That(conditioning.Specialization, Is.EqualTo("3"));

        IReadOnlyDictionary<string, HostTensor> shorter = conditioning.Branch(0);
        Assert.That(shorter.Keys, Is.EquivalentTo(QwenImageConditioning.Names));
        Assert.That(shorter[QwenImageConditioning.ContextName].Shape, Is.EqualTo(new[] { 1, 1, 3, width }));
        Assert.That(shorter[QwenImageConditioning.ContextName].ToFloats().Skip(2 * width), Is.All.Zero, "padded");
        Assert.That(shorter[QwenImageConditioning.KeyLengthName].ToInt32s(),
            Is.EqualTo(new[] { latentRows * latentColumns + 2 }));
        Assert.That(shorter[QwenImageConditioning.TextMaskName].ToInt32s(), Is.EqualTo(new[]
        {
            1, 0, 0,
            1, 1, 0,
            1, 1, 0, // the padding row reads the real tokens, nothing reads it
        }));
        Assert.That(shorter[QwenImageConditioning.CosName].Shape, Is.EqualTo(new[] { 1, 4 + 3, 1, 64 }));
        // The image's frame axis: at the branch's own text length, 2.
        Assert.That(shorter[QwenImageConditioning.CosName].ToFloats()[0], Is.EqualTo(Math.Cos(2)).Within(1e-6));

        IReadOnlyDictionary<string, HostTensor> longer = conditioning.Branch(1);
        Assert.That(longer[QwenImageConditioning.KeyLengthName].ToInt32s(), Is.EqualTo(new[] { 4 + 3 }));
        Assert.That(longer[QwenImageConditioning.TextMaskName].ToInt32s(), Is.EqualTo(new[]
        {
            1, 0, 0,
            1, 1, 0,
            1, 1, 1,
        }));
        Assert.That(longer[QwenImageConditioning.CosName].ToFloats()[0], Is.EqualTo(Math.Cos(3)).Within(1e-6));
    }

    /// <summary>Under an alignment the rows make the sequence up to a
    /// multiple of it, and the padding is masked as any padding is.</summary>
    [Test]
    public void QwenImagesSequenceIsPaddedToTheAlignmentAsked()
    {
        const int width = 4, latentRows = 2, latentColumns = 2;
        float[] positive = Enumerable.Repeat(2f, 3 * width).ToArray();
        var conditioning = new QwenImageConditioning(new[] { positive }, width, latentRows, latentColumns,
            alignment: 16);

        Assert.That(conditioning.Rows, Is.EqualTo(12), "4 image tokens + 3 rows, up to 16");
        IReadOnlyDictionary<string, HostTensor> branch = conditioning.Branch(0);
        Assert.That(branch[QwenImageConditioning.ContextName].Shape, Is.EqualTo(new[] { 1, 1, 12, width }));
        Assert.That(branch[QwenImageConditioning.ContextName].ToFloats().Skip(3 * width), Is.All.Zero);
        Assert.That(branch[QwenImageConditioning.KeyLengthName].ToInt32s(), Is.EqualTo(new[] { 4 + 3 }));
        Assert.That(branch[QwenImageConditioning.TextMaskName].ToInt32s(),
            Is.EqualTo(QwenImageConditioning.CausalTable(real: 3, rows: 12)));
        Assert.That(branch[QwenImageConditioning.CosName].Shape, Is.EqualTo(new[] { 1, 16, 1, 64 }));
        Assert.That(branch[QwenImageConditioning.CosName].ToFloats()[0], Is.EqualTo(Math.Cos(3)).Within(1e-6),
            "the image still sits at the real text length");

        var unaligned = new QwenImageConditioning(new[] { positive }, width, latentRows, latentColumns);
        Assert.That(unaligned.Rows, Is.EqualTo(3));
    }

    [Test]
    public void TheCausalTableIsATriangleClosedOverThePadding()
    {
        Assert.That(QwenImageConditioning.CausalTable(real: 1, rows: 1), Is.EqualTo(new[] { 1 }));
        Assert.That(QwenImageConditioning.CausalTable(real: 2, rows: 4), Is.EqualTo(new[]
        {
            1, 0, 0, 0,
            1, 1, 0, 0,
            1, 1, 0, 0,
            1, 1, 0, 0,
        }));
    }

    /// <summary>A file of rows stands in for the text encoder: Anima pads
    /// them to its context length, Qwen-Image reads their width off the
    /// checkpoint's header, SDXL — a context and an add-vector — takes none.</summary>
    [Test]
    public void AConditionerTakesRowsInPlaceOfItsTextEncoder()
    {
        using var folder = new TempFolder();
        var files = new FakeSafetensors(folder);
        string qwenImage = files.Write("qwen21", FakeSafetensors.QwenImageDit());
        var options = new GenerationOptions { Width = 32, Height = 32, CheckpointPath = qwenImage };

        Assert.That(SdxlFamily.Instance.Conditioner.TakesContextRows, Is.False);
        Assert.That(() => SdxlFamily.Instance.Conditioner.FromContextRows(options, new float[4]),
            Throws.InstanceOf<NotSupportedException>());

        Assert.That(AnimaFamily.Instance.Conditioner.TakesContextRows);
        Conditioning anima = AnimaFamily.Instance.Conditioner.FromContextRows(options,
            Enumerable.Repeat(1f, 3 * LlmAdapter.Width).ToArray());
        Assert.That(anima.Branches, Is.EqualTo(1));
        Assert.That(anima.Specialization, Is.EqualTo(AnimaPrompt.ContextLength.ToString()), "padded to 512 rows");
        HostTensor context = anima.Branch(0)[AnimaConditioning.ContextName];
        Assert.That(context.Shape, Is.EqualTo(new[] { 1, 1, AnimaPrompt.ContextLength, LlmAdapter.Width }));
        Assert.That(context.ToFloats().Take(3 * LlmAdapter.Width), Is.All.EqualTo(1f));
        Assert.That(context.ToFloats().Skip(3 * LlmAdapter.Width), Is.All.Zero);
        Assert.That(() => AnimaFamily.Instance.Conditioner.FromContextRows(options, new float[5]),
            Throws.ArgumentException.With.Message.Contains("rows"));

        Assert.That(QwenImageFamily.Instance.Conditioner.TakesContextRows);
        Conditioning qwen = QwenImageFamily.Instance.Conditioner.FromContextRows(options, new float[2 * 4096]);
        Assert.That(qwen.Branches, Is.EqualTo(1));
        Assert.That(qwen.Specialization, Is.EqualTo("2"));
        Assert.That(qwen.Branch(0)[QwenImageConditioning.KeyLengthName].ToInt32s(), Is.EqualTo(new[] { 4 + 2 }));
        Assert.That(() => QwenImageFamily.Instance.Conditioner.FromContextRows(options, new float[4096 + 1]),
            Throws.ArgumentException.With.Message.Contains("4096"));
    }

    [Test]
    public void TheBranchCountIsCheckedAgainstTheGuidance()
    {
        var one = new AnimaConditioning(new[] { HostTensor.FromFloats(new float[4], 1, 1, 1, 4) }, 1);
        Assert.That(one.CheckedFor(1), Is.SameAs(one));
        Assert.That(() => one.CheckedFor(2), Throws.InvalidOperationException.With.Message.Contains("1 branches"));
        Assert.That(() => one.CheckedFor(3), Throws.InstanceOf<ArgumentOutOfRangeException>());
    }
}
