using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace NaiveDiffusion.Tensors;

/// <summary>What converting a stored tensor to the width the graphs run at
/// does to its values: how many overflow to infinity, how many land among
/// the subnormals and keep fewer bits, how many round to zero, and what
/// share of the tensor's energy (Σw²) each of those holds, with the error
/// energy the rounding leaves over the values that stay finite.
///
/// bfloat16 to float16 is exact inside float16's normal range and nowhere
/// else — bfloat16 has the exponent of a float32 — so a weight past 65504
/// or under 6.1e-5 changes silently; float32 to float16 also rounds the
/// mantissa. A model whose weights cross those lines draws a worse image
/// with nothing to say why, and this is the saying.</summary>
public sealed record NarrowingLoss
{
    /// <summary>float16's smallest normal magnitude, 2^-14.</summary>
    public const float SmallestNormalHalf = 6.1035156e-5f;

    public long Elements { get; init; }

    /// <summary>NaN or infinity in the file itself: nothing the conversion
    /// did, and left out of every energy below.</summary>
    public long NonFinite { get; init; }

    /// <summary>Finite in the file, infinite after.</summary>
    public long Overflowed { get; init; }

    /// <summary>Nonzero after, but under the smallest normal: fewer mantissa bits.</summary>
    public long Subnormal { get; init; }

    /// <summary>Nonzero in the file, zero after.</summary>
    public long Flushed { get; init; }

    public double Energy { get; init; }
    public double OverflowedEnergy { get; init; }
    public double SubnormalEnergy { get; init; }
    public double FlushedEnergy { get; init; }

    /// <summary>Σ(w − narrowed w)² over the values that stay finite.</summary>
    public double ErrorEnergy { get; init; }

    /// <summary>The rounding error relative to the tensor, as an RMS ratio:
    /// 0 when the conversion is exact.</summary>
    public double RelativeError => Energy > 0 ? Math.Sqrt(ErrorEnergy / Energy) : 0;

    /// <summary>Whether anything crossed float16's range, as against the
    /// mantissa rounding every float32 source has.</summary>
    public bool OutOfRange => Overflowed + Subnormal + Flushed > 0;

    public static readonly NarrowingLoss None = new();

    public static NarrowingLoss operator +(NarrowingLoss a, NarrowingLoss b) => new()
    {
        Elements = a.Elements + b.Elements,
        NonFinite = a.NonFinite + b.NonFinite,
        Overflowed = a.Overflowed + b.Overflowed,
        Subnormal = a.Subnormal + b.Subnormal,
        Flushed = a.Flushed + b.Flushed,
        Energy = a.Energy + b.Energy,
        OverflowedEnergy = a.OverflowedEnergy + b.OverflowedEnergy,
        SubnormalEnergy = a.SubnormalEnergy + b.SubnormalEnergy,
        FlushedEnergy = a.FlushedEnergy + b.FlushedEnergy,
        ErrorEnergy = a.ErrorEnergy + b.ErrorEnergy,
    };

    /// <summary>Elements per piece of work: large enough that the per-piece
    /// cost vanishes, small enough that a 200-million-element matrix keeps
    /// every core busy and no piece allocates much.</summary>
    private const int Chunk = 1 << 20;

    /// <summary>The loss of converting <paramref name="stored"/> to
    /// <paramref name="target"/>, which is float16 or float32; to float32
    /// nothing is lost, and only the file's own non-finite values count.</summary>
    public static NarrowingLoss Measure(HostTensor stored, HostDataType target)
    {
        if (target is not (HostDataType.Float16 or HostDataType.Float32))
        {
            throw new NotSupportedException($"narrowing to {target}");
        }
        bool narrows = target == HostDataType.Float16 && stored.DataType != HostDataType.Float16;
        long elements = stored.ElementCount;
        int itemSize = HostTensor.BytesPerElement(stored.DataType);
        int chunks = checked((int)((elements + Chunk - 1) / Chunk));
        ReadOnlyMemory<byte> data = stored.Data;
        HostDataType type = stored.DataType;

        NarrowingLoss total = None;
        object gate = new();
        Parallel.For(0, chunks, () => (Loss: None, Wide: new float[Chunk], Half: new Half[Chunk]),
            (chunk, _, state) =>
            {
                long start = (long)chunk * Chunk;
                int count = (int)Math.Min(Chunk, elements - start);
                Span<float> wide = state.Wide.AsSpan(0, count);
                HostTensor.Widen(type, data.Span.Slice(checked((int)(start * itemSize)), count * itemSize), wide);
                Span<Half> half = state.Half.AsSpan(0, count);
                if (narrows)
                {
                    TensorPrimitives.ConvertToHalf(wide, half);
                }
                state.Loss += Classify(wide, half, narrows);
                return state;
            },
            state =>
            {
                lock (gate)
                {
                    total += state.Loss;
                }
            });
        return total;
    }

    private static NarrowingLoss Classify(ReadOnlySpan<float> wide, ReadOnlySpan<Half> half, bool narrows)
    {
        long nonFinite = 0, overflowed = 0, subnormal = 0, flushed = 0;
        double energy = 0, overflowedEnergy = 0, subnormalEnergy = 0, flushedEnergy = 0, error = 0;
        for (int i = 0; i < wide.Length; i++)
        {
            float w = wide[i];
            if (!float.IsFinite(w))
            {
                nonFinite++;
                continue;
            }
            double square = (double)w * w;
            energy += square;
            if (!narrows)
            {
                continue;
            }
            float back = (float)half[i];
            if (float.IsInfinity(back))
            {
                overflowed++;
                overflowedEnergy += square;
                continue;
            }
            if (back == 0)
            {
                if (w != 0)
                {
                    flushed++;
                    flushedEnergy += square;
                }
            }
            else if (MathF.Abs(back) < SmallestNormalHalf)
            {
                subnormal++;
                subnormalEnergy += square;
            }
            double difference = (double)w - back;
            error += difference * difference;
        }
        return new NarrowingLoss
        {
            Elements = wide.Length,
            NonFinite = nonFinite,
            Overflowed = overflowed,
            Subnormal = subnormal,
            Flushed = flushed,
            Energy = energy,
            OverflowedEnergy = overflowedEnergy,
            SubnormalEnergy = subnormalEnergy,
            FlushedEnergy = flushedEnergy,
            ErrorEnergy = error,
        };
    }

    /// <summary>The error block-wise int8 leaves on <paramref name="weight"/>
    /// viewed as [rows, columns] — what <see cref="BlockQuantizer.Quantize"/>
    /// and the dequantize op give back, measured on the float16 values the
    /// quantizer is handed — as Σ(w − dequantized w)² beside Σw², so a
    /// whole model's can be summed before the ratio is taken.</summary>
    public static (double ErrorEnergy, double Energy) Quantization(HostTensor weight, int rows)
    {
        HostTensor narrow = weight.ConvertTo(HostDataType.Float16);
        (HostTensor quantized, HostTensor scales) = BlockQuantizer.Quantize(narrow, rows);
        float[] original = narrow.ToFloats();
        float[] scale = scales.ToFloats();
        ReadOnlySpan<sbyte> values = MemoryMarshal.Cast<byte, sbyte>(quantized.Data.Span);
        double energy = 0, error = 0;
        for (int i = 0; i < original.Length; i++)
        {
            // The scales are [rows, columns / BlockSize] and a row is whole
            // blocks, so element i's block is i / BlockSize.
            double back = values[i] * (double)scale[i / BlockQuantizer.BlockSize];
            double difference = original[i] - back;
            energy += (double)original[i] * original[i];
            error += difference * difference;
        }
        return (error, energy);
    }
}
