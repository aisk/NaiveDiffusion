using System.Diagnostics;
using NaiveDiffusion.Models;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Tensors;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Cli;

/// <summary>What converting a checkpoint's diffusion model to the width
/// the graphs run at costs, tensor by tensor, before anything is built:
/// every family narrows those weights to float16, which bfloat16 survives
/// only inside float16's normal range and float32 never survives exactly.
/// A weight past the range comes out as infinity, zero or a subnormal
/// with no error anywhere, and the image is only worse; this is the
/// command to run once when a new model is taken on. With --weights int8 it also
/// measures what the block quantization would leave. The text encoders
/// are read at their stored width on the CPU and the VAEs are widened,
/// so neither loses anything and neither is read. Opens no device.</summary>
internal static class InspectCommand
{
    public static readonly string[] ValueOptions = { "--checkpoint", "--weights" };

    /// <summary>How many of the worst tensors a warning names.</summary>
    private const int Worst = 5;

    /// <summary>A tensor's relative error past which values leaving the
    /// range are worth a warning: twice float16's own rounding of a float32
    /// value. Below it, the subnormals every bfloat16 model has — Anima's
    /// fifteen million hold a millionth of a percent of the energy — keep
    /// their few bits and change nothing that shows.</summary>
    private const double Noticeable = 1e-3;

    public static int Run(CommandLine line)
    {
        IModelFamily family = line.Family(CheckpointInspector.CheckpointParts.Unet);
        string path = line.Checkpoint(CheckpointInspector.CheckpointParts.Unet);
        WeightStorage storage = line.Weights();
        if (!family.SupportsWeights(storage))
        {
            CommandLine.Fail($"--weights {storage.Label()}: {family.Name} cannot store its weights so");
        }
        bool int8 = storage == WeightStorage.Int8;

        var clock = Stopwatch.StartNew();
        var losses = new List<(string Name, NarrowingLoss Loss)>();
        var quantized = new List<(string Name, double Error, double Energy)>();
        var stored = new Dictionary<HostDataType, long>();
        HostDataType target = HostDataType.Float32;
        family.ReadDenoiserWeights(path, weights =>
        {
            target = weights.DataType;
            foreach (string name in weights.Keys.Order(StringComparer.Ordinal))
            {
                HostTensor tensor = weights.Unnarrowed(name);
                stored[tensor.DataType] = stored.GetValueOrDefault(tensor.DataType) + tensor.ElementCount;
                losses.Add((name, NarrowingLoss.Measure(tensor, target)));
                if (int8 && Quantizes(tensor) is int rows)
                {
                    (double error, double energy) = NarrowingLoss.Quantization(tensor, rows);
                    quantized.Add((name, error, energy));
                }
            }
        });

        NarrowingLoss total = losses.Aggregate(NarrowingLoss.None, (sum, entry) => sum + entry.Loss);
        Console.WriteLine($"{family.Name} diffusion model: {losses.Count} tensors, {total.Elements:N0} values " +
                          $"stored as {string.Join(", ", stored.Select(pair => $"{pair.Key} ({pair.Value:N0})"))}, " +
                          $"read at {target} ({clock.Elapsed.TotalSeconds:0.0} s)");
        (string worstName, NarrowingLoss worstLoss) = losses.MaxBy(entry => entry.Loss.RelativeError);
        Console.WriteLine($"  relative rms error {total.RelativeError:0.00e+00}" + (worstLoss.RelativeError > 0
            ? $", the worst tensor's {worstLoss.RelativeError:0.00e+00} ({worstName})"
            : ""));
        Console.WriteLine($"  overflowed to infinity {Count(total.Overflowed, total.OverflowedEnergy, total)}");
        Console.WriteLine($"  subnormal              {Count(total.Subnormal, total.SubnormalEnergy, total)}");
        Console.WriteLine($"  rounded to zero        {Count(total.Flushed, total.FlushedEnergy, total)}");
        Console.WriteLine($"  NaN or infinite in the file {total.NonFinite:N0}");

        // An overflow is an infinity in the graph whatever its share; a
        // value under the range matters only when it moves the tensor.
        var harmed = losses
            .Where(entry => entry.Loss.Overflowed > 0
                            || (entry.Loss.OutOfRange && entry.Loss.RelativeError > Noticeable))
            .ToList();
        if (harmed.Count > 0)
        {
            Console.Error.WriteLine($"warning: {harmed.Count} tensors are changed by leaving {target}'s " +
                                    "normal range; the worst:");
            foreach ((string name, NarrowingLoss loss) in harmed
                         .OrderByDescending(entry => entry.Loss.Overflowed)
                         .ThenByDescending(entry => entry.Loss.RelativeError)
                         .Take(Worst))
            {
                Console.Error.WriteLine($"  {name}: relative error {loss.RelativeError:0.00e+00}, " +
                                        $"{OutOfRangeShare(loss):P4} of its energy out of range, " +
                                        $"{loss.Overflowed:N0} overflowed, {loss.Subnormal:N0} subnormal, " +
                                        $"{loss.Flushed:N0} to zero");
            }
        }
        if (total.NonFinite > 0)
        {
            Console.Error.WriteLine($"warning: {total.NonFinite:N0} values are NaN or infinite in the file itself:");
            foreach ((string name, NarrowingLoss loss) in losses
                         .Where(entry => entry.Loss.NonFinite > 0)
                         .OrderByDescending(entry => entry.Loss.NonFinite)
                         .Take(Worst))
            {
                Console.Error.WriteLine($"  {name}: {loss.NonFinite:N0}");
            }
        }

        if (int8)
        {
            double error = quantized.Sum(entry => entry.Error), energy = quantized.Sum(entry => entry.Energy);
            Console.WriteLine($"int8 blocks of {BlockQuantizer.BlockSize}: {quantized.Count} of {losses.Count} " +
                              $"tensors, relative rms error {Rms(error, energy):0.00000} " +
                              $"({clock.Elapsed.TotalSeconds:0.0} s in all); the worst:");
            foreach ((string name, double tensorError, double tensorEnergy) in quantized
                         .OrderByDescending(entry => Rms(entry.Error, entry.Energy))
                         .Take(Worst))
            {
                Console.WriteLine($"  {name}: {Rms(tensorError, tensorEnergy):0.00000}");
            }
        }
        return 0;
    }

    /// <summary>The rows the graphs would quantize a tensor over, or null
    /// if it stays at float16 — the rule <c>ModelBuilder.Weight</c> applies,
    /// with the output extent read as the leading dimension, which is what
    /// every linear and convolution weight is laid out with. A matrix the
    /// graphs take through a plain constant is counted too, so this is the
    /// most int8 could cost, not less.</summary>
    private static int? Quantizes(HostTensor tensor) =>
        tensor.Shape.Length >= 2
        && tensor.ElementCount >= BlockQuantizer.MinimumElements
        && BlockQuantizer.Fits(tensor.ElementCount, tensor.Shape[0])
            ? tensor.Shape[0]
            : null;

    private static double OutOfRangeShare(NarrowingLoss loss) => Ratio(
        loss.OverflowedEnergy + loss.SubnormalEnergy + loss.FlushedEnergy, loss.Energy);

    private static double Ratio(double part, double whole) => whole > 0 ? part / whole : 0;

    /// <summary>An error energy against the energy as an RMS ratio.</summary>
    private static double Rms(double error, double energy) => Math.Sqrt(Ratio(error, energy));

    private static string Count(long count, double energy, NarrowingLoss total) =>
        $"{count:N0} ({Ratio(energy, total.Energy):P6} of the energy)";
}
