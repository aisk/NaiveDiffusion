using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Tensors;
using NaiveDiffusion.Text;
using NaiveDiffusion.Text.Clip;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Models.Sdxl;

/// <summary>SDXL's two CLIP towers plus their tokenizers: prompt in, conditioning
/// out. The concatenation, the pooled vector and the chunk alignment are SDXL's
/// conventions; the towers themselves are <see cref="ClipTextEncoder"/>.</summary>
public sealed class SdxlTextEncoders : IDisposable
{
    private readonly List<(ClipTextEncoder.Config Config, ClipTokenizer Tokenizer,
        TowerWeights Weights)> _towers = new();
    private readonly float[] _projection;   // [1280, 1280]
    private readonly int _projectionWidth;

    // The towers' matrices are windows onto this mapping and are read straight
    // off it, so it stays open for the life of this object rather than only for
    // the load. That is the point: nothing is copied into the heap.
    private readonly SdxlWeights _checkpoint;

    /// <summary>Which layer's output the prompt is read from, counted from
    /// the end; see <see cref="GenerationOptions.ClipSkip"/>.</summary>
    private readonly int _clipSkip;

    public SdxlTextEncoders(string checkpointPath, IReadOnlyList<LoraSpec>? loras = null,
        int clipSkip = GenerationOptions.DefaultClipSkip)
    {
        if (clipSkip < 1 || clipSkip > ClipTextEncoder.MaxClipSkip)
        {
            throw new ArgumentOutOfRangeException(nameof(clipSkip), clipSkip,
                $"between 1 and {ClipTextEncoder.MaxClipSkip}");
        }
        _clipSkip = clipSkip;
        _checkpoint = new SdxlWeights(checkpointPath);
        try
        {
            float[]? projection = null;
            foreach (ClipTextEncoder.Config config in ClipTextEncoder.Configs)
            {
                var weights = new TowerWeights(
                    _checkpoint.LoadTextEncoder(config.Name, loras));

                if (config.WantsPooled)
                {
                    projection = weights.TakeWidened("text_projection.weight");
                    _projectionWidth = projection.Length / config.Width;
                }
                _towers.Add((config, ClipTokenizer.For(config.PadToken), weights));
            }
            _projection = projection ?? throw new InvalidDataException(
                "no text encoder carries the projection SDXL pools from");
        }
        catch
        {
            // A LoRA that does not fit is found here, and the caller never
            // gets an object to dispose: the checkpoint's mapping must not
            // outlive the attempt.
            Dispose();
            throw;
        }
    }

    /// <summary>How many 77-token chunks a prompt takes. The towers share a
    /// vocabulary and differ only in what they pad with, so one count is both
    /// towers' count.</summary>
    public static int ChunkCount(string prompt) =>
        ClipTokenizer.For(ClipTextEncoder.Configs[0].PadToken)
            .EncodeChunks(PromptTags.Segments(prompt)).Count;

    /// <summary>Returns (prompt embeddings [77 * chunks, 2048], pooled embedding
    /// [1280]). <paramref name="chunks"/> is a floor, for when another prompt in
    /// the same run is longer and both have to reach the UNet as one shape; the
    /// chunks past this prompt's own length are empty ones.</summary>
    public (float[] Embeddings, float[] Pooled) Encode(string prompt, int chunks = 1)
    {
        // The prompt stops being a string here: from now on it is text with
        // weights on it, which is the only form the towers are given.
        IReadOnlyList<PromptSegment> segments = PromptTags.Segments(prompt);
        int count = Math.Max(chunks, ChunkCount(prompt));
        int tokens = count * ClipTextEncoder.MaxTokens;
        var sequences = new List<(float[] Data, int Width)>();
        float[]? pooled = null;

        foreach ((ClipTextEncoder.Config config, ClipTokenizer tokenizer,
                  TowerWeights weights) in _towers)
        {
            List<TokenChunk> chunked = tokenizer.EncodeChunks(segments);
            TokenChunk empty = tokenizer.EncodeChunks("")[0];
            var data = new float[tokens * config.Width];
            // The padding chunks are all the same empty one, so the tower
            // runs it once however many of them a short prompt needs.
            float[]? padding = null;

            for (int chunk = 0; chunk < count; chunk++)
            {
                if (chunk >= chunked.Count && padding is not null)
                {
                    Array.Copy(padding, 0, data, chunk * padding.Length, padding.Length);
                    continue;
                }
                TokenChunk piece = chunk < chunked.Count ? chunked[chunk] : empty;
                (float[] hidden, float[]? final) =
                    ClipTextEncoder.Forward(weights, piece.Ids, config, _clipSkip);
                if (Array.Exists(hidden, value => !float.IsFinite(value)))
                {
                    // Finetunes trained on the penultimate layer often ship
                    // the last one as NaN: it never ran, so it was never
                    // saved with a value. SDXL reads it at clip skip 1 and at
                    // no other setting; the image would come out black, so
                    // this stops here, before the UNet is built.
                    throw new InvalidDataException(
                        $"{config.Name} gives NaN at clip skip {_clipSkip}: the checkpoint's " +
                        "last text encoder layers hold no usable weights, as finetunes trained " +
                        "on the penultimate layer often do not; use clip skip 2 or more");
                }
                Reweight(hidden, piece.Weights, config.Width);
                if (chunk >= chunked.Count)
                {
                    padding = hidden;
                }
                Array.Copy(hidden, 0, data, chunk * hidden.Length, hidden.Length);

                // SDXL conditions on a single pooled vector however long the
                // prompt is, and the towers were never trained on more than one,
                // so the first chunk's is the one — as in ComfyUI and A1111.
                if (config.WantsPooled && chunk == 0)
                {
                    pooled = Pool(final!, piece.Ids, config.Width);
                }
            }
            sequences.Add((data, config.Width));
        }

        // Concatenate the towers' per-token outputs along the feature axis.
        int totalWidth = sequences.Sum(sequence => sequence.Width);
        var embeddings = new float[tokens * totalWidth];
        int offset = 0;
        foreach ((float[] data, int width) in sequences)
        {
            for (int token = 0; token < tokens; token++)
            {
                Array.Copy(data, token * width, embeddings, token * totalWidth + offset, width);
            }
            offset += width;
        }
        // The constructor established that one tower pools, so its first chunk
        // has already been through Pool by here.
        return (embeddings, pooled ?? throw new InvalidOperationException(
            "no tower produced a pooled embedding"));
    }

    /// <summary>Scale one chunk's per-token states by what the prompt said each
    /// token was worth, then put the chunk's mean back where it was.
    ///
    /// Both halves matter. The scaling is the weight itself; the correction is
    /// what keeps a weight meaning "more of this" rather than "louder
    /// everything". Without it the whole chunk grows with every emphasis in the
    /// prompt, the UNet reads that as a stronger signal across the board, and
    /// CFG multiplies the difference — the image comes out burnt at a weight
    /// nobody would call extreme. A1111 and ComfyUI both correct the mean this
    /// way, so a prompt written over there lands in the same place here.
    ///
    /// The pooled vector is deliberately left out of this: SDXL conditions on
    /// one pooled vector for the whole prompt, and neither reference
    /// implementation weights it either.</summary>
    private static void Reweight(float[] hidden, float[] weights, int width)
    {
        if (Array.TrueForAll(weights, weight => weight == 1f))
        {
            return;
        }

        double before = 0;
        foreach (float value in hidden)
        {
            before += value;
        }

        for (int token = 0; token < weights.Length; token++)
        {
            if (weights[token] == 1f)
            {
                continue;
            }
            for (int i = token * width; i < (token + 1) * width; i++)
            {
                hidden[i] *= weights[token];
            }
        }

        double after = 0;
        foreach (float value in hidden)
        {
            after += value;
        }
        // A chunk whose states happen to cancel out has no mean to restore, and
        // the ratio would be an infinity on its way into every element.
        if (after == 0)
        {
            return;
        }
        float scale = (float)(before / after);
        for (int i = 0; i < hidden.Length; i++)
        {
            hidden[i] *= scale;
        }
    }

    /// <summary>The pooled vector for one chunk: CLIP's final hidden state at
    /// the end-of-text token, through the projection.</summary>
    private float[] Pool(float[] final, uint[] ids, int width)
    {
        // The end-of-text token is the highest id in the vocabulary, so it is
        // the one argmax finds.
        int endOfText = 0;
        for (int i = 1; i < ids.Length; i++)
        {
            if (ids[i] > ids[endOfText])
            {
                endOfText = i;
            }
        }

        var pooled = new float[_projectionWidth];
        for (int o = 0; o < _projectionWidth; o++)
        {
            double sum = 0;
            for (int i = 0; i < width; i++)
            {
                sum += final[endOfText * width + i] * _projection[o * width + i];
            }
            pooled[o] = (float)sum;
        }
        return pooled;
    }

    /// <summary>Drop the towers and unmap the checkpoint. Every matrix they
    /// hold points into that mapping, so nothing may touch them after this.</summary>
    public void Dispose()
    {
        foreach ((_, _, TowerWeights weights) in _towers)
        {
            weights.Clear();
        }
        _towers.Clear();
        _checkpoint.Dispose();
    }
}
