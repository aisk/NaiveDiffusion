using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Tensors;
using NaiveDiffusion.Vae;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Models.QwenImage;

/// <summary>What Qwen-Image 2.1's transformer reads for one branch of the
/// guidance, under the names <see cref="QwenImageDit"/> reads them by: the
/// text encoder's rows padded with zeros to the longer branch's length,
/// since both branches go through the same compiled graphs; the causal
/// table the text rows attend under, closed over the padding; the count of
/// keys the image rows may read, its own tokens plus the real text; and
/// the rotary tables, whose image rows are placed after this branch's own
/// text length, as the reference places them with no padding at all. The
/// prompt's branch is last; with the guidance off it is the only one.
/// With the blocks computing at half precision the padding also brings
/// the whole sequence to a multiple of
/// <see cref="QwenImageDit.HalfComputeAlignment"/> rows; it changes
/// nothing the model reads, since the count of real keys and the causal
/// table close over it.</summary>
public sealed class QwenImageConditioning : Conditioning
{
    public const string ContextName = "context";
    public const string TextMaskName = "text_mask";
    public const string KeyLengthName = "key_length";
    public const string CosName = "cos";
    public const string SinName = "sin";

    public static readonly string[] Names = { ContextName, TextMaskName, KeyLengthName, CosName, SinName };

    private readonly IReadOnlyDictionary<string, HostTensor>[] _branches;

    /// <param name="contexts">Each branch's rows, [rows, width] packed, negative first.</param>
    /// <param name="width">The rows' width, the text encoder's.</param>
    /// <param name="latentRows">The latent's height in cells.</param>
    /// <param name="latentColumns">And its width.</param>
    /// <param name="alignment">What the sequence's length — the image's
    /// tokens and the rows — is padded to a multiple of; 1 pads only the
    /// shorter branch.</param>
    public QwenImageConditioning(IReadOnlyList<float[]> contexts, int width, int latentRows, int latentColumns,
        int alignment = 1)
    {
        int imageTokens = latentRows * latentColumns;
        int longest = contexts.Max(context => context.Length / width);
        Rows = (imageTokens + longest + alignment - 1) / alignment * alignment - imageTokens;
        _branches = contexts.Select(context =>
        {
            int rows = context.Length / width;
            var padded = new float[Rows * width];
            Array.Copy(context, padded, context.Length);
            (HostTensor cos, HostTensor sin) = QwenImageDit.RopeTables(latentRows, latentColumns, rows, Rows);
            return (IReadOnlyDictionary<string, HostTensor>)new Dictionary<string, HostTensor>
            {
                [ContextName] = HostTensor.FromFloats(padded, 1, 1, Rows, width),
                [TextMaskName] = HostTensor.FromInt32(CausalTable(rows, Rows), 1, 1, Rows, Rows),
                [KeyLengthName] = HostTensor.FromInt32(new[] { imageTokens + rows }, 1, 1, 1, 1),
                [CosName] = cos,
                [SinName] = sin,
            };
        }).ToArray();
    }

    /// <summary>The boolean table the text rows attend under: row i may
    /// read key j where j ≤ i and j is a real token — the causal triangle,
    /// closed over the padding rows past <paramref name="real"/>. A padding
    /// row's own query reads the real tokens, which keeps its softmax
    /// finite; nothing reads the row back.</summary>
    public static int[] CausalTable(int real, int rows)
    {
        var mask = new int[rows * rows];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j <= i && j < real; j++)
            {
                mask[i * rows + j] = 1;
            }
        }
        return mask;
    }

    /// <summary>The context length the graphs are compiled for: the longer
    /// branch's own.</summary>
    public int Rows { get; }

    public override int Branches => _branches.Length;

    public override IReadOnlyDictionary<string, HostTensor> Branch(int index) => _branches[index];

    public override string Specialization => Rows.ToString();
}

/// <summary>Qwen-Image 2.1's prompt encoding: Qwen3-VL over the prompt, and
/// over the negative prompt when the guidance is on, on the CPU.</summary>
public sealed class QwenImageConditioner : IConditioner
{
    public IPromptEncoder Open(GenerationOptions options) =>
        new Encoder(new QwenImageTextEncoder(options.CheckpointPath,
            options.ComponentPath(QwenImageFamily.TextEncoderComponent.Id)
            ?? throw new ArgumentException("Qwen-Image needs its text encoder file"),
            options.Loras));

    public bool TakesContextRows => true;

    /// <summary>One branch from rows at the width this checkpoint's text
    /// input projects, read off its header — a small transformer of the
    /// real shape has a small width.</summary>
    public Conditioning FromContextRows(GenerationOptions options, float[] values)
    {
        int width;
        using (var checkpoint = new SafetensorsFile(options.CheckpointPath))
        {
            width = QwenImageCheckpoint.ContextWidth(checkpoint);
        }
        if (values.Length == 0 || values.Length % width != 0)
        {
            throw new ArgumentException($"{values.Length} values is not a whole number of rows of {width}");
        }
        return Conditioning(new[] { values }, width, options);
    }

    private static QwenImageConditioning Conditioning(IReadOnlyList<float[]> contexts, int width,
        GenerationOptions options)
    {
        int scale = Wan22Vae.Latent.ScaleFactor;
        return new QwenImageConditioning(contexts, width, options.Height / scale, options.Width / scale,
            options.DenoiserCompute == ComputePrecision.Float16 ? QwenImageDit.HalfComputeAlignment : 1);
    }

    private sealed class Encoder(QwenImageTextEncoder encoder) : IPromptEncoder
    {
        public Conditioning Encode(GenerationOptions options, int branches)
        {
            var contexts = new List<float[]>();
            if (branches > 1)
            {
                contexts.Add(encoder.Encode(options.Negative));
            }
            contexts.Add(encoder.Encode(options.Prompt));
            return Conditioning(contexts, QwenImageTextEncoder.Width, options);
        }

        public void Dispose() => encoder.Dispose();
    }
}
