using NaiveDiffusion.Cli;
using NaiveDiffusion.Models.Anima;
using NaiveDiffusion.Tensors;
using NaiveDiffusion.Tests.Fixtures;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Tests;

/// <summary>What narrowing a stored tensor to float16 does to it: exact
/// inside the normal range from bfloat16, and each way out of the range
/// counted where it happens; and the inspect command that reports it for
/// a whole diffusion model, through the family's own loader.</summary>
public class NarrowingLossTests
{
    private static HostTensor BFloat16(params float[] values)
    {
        var bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
        {
            BitConverter.TryWriteBytes(bytes.AsSpan(i * 2),
                (ushort)(BitConverter.SingleToUInt32Bits(values[i]) >> 16));
        }
        return new HostTensor(HostDataType.BFloat16, new[] { values.Length }, bytes);
    }

    [Test]
    public void BFloat16InsideTheNormalRangeNarrowsExactly()
    {
        NarrowingLoss loss = NarrowingLoss.Measure(BFloat16(1f, -0.5f, 0f, 1024f, 1e-3f, -60000f),
            HostDataType.Float16);
        Assert.That(loss.Elements, Is.EqualTo(6));
        Assert.That(loss.OutOfRange, Is.False);
        Assert.That(loss.RelativeError, Is.Zero);
        Assert.That(loss.NonFinite, Is.Zero);
    }

    [Test]
    public void EachWayOutOfTheRangeIsCountedWithItsEnergy()
    {
        NarrowingLoss loss = NarrowingLoss.Measure(BFloat16(1e5f, 1e-5f, 1e-9f, 1f, float.NaN),
            HostDataType.Float16);
        Assert.That(loss.Overflowed, Is.EqualTo(1));
        Assert.That(loss.Subnormal, Is.EqualTo(1));
        Assert.That(loss.Flushed, Is.EqualTo(1));
        Assert.That(loss.NonFinite, Is.EqualTo(1));
        Assert.That(loss.OutOfRange);
        Assert.That(loss.OverflowedEnergy / loss.Energy, Is.GreaterThan(0.99));
        Assert.That(loss.FlushedEnergy, Is.GreaterThan(0).And.LessThan(1e-17));
    }

    [Test]
    public void Float32RoundsTheMantissaWithoutLeavingTheRange()
    {
        NarrowingLoss loss = NarrowingLoss.Measure(HostTensor.FromFloats(new[] { 1f / 3, 2f / 3, 1f }, 3),
            HostDataType.Float16);
        Assert.That(loss.OutOfRange, Is.False);
        Assert.That(loss.RelativeError, Is.GreaterThan(0).And.LessThan(1e-3));

        Assert.That(NarrowingLoss.Measure(HostTensor.FromFloats(new[] { 1e5f }, 1), HostDataType.Float32)
            .OutOfRange, Is.False, "nothing is lost widening");
        Assert.That(NarrowingLoss.Measure(HostTensor.FromHalves(new[] { (Half)0.1f }, 1), HostDataType.Float16)
            .RelativeError, Is.Zero, "nor staying at float16");
    }

    [Test]
    public void ALargeTensorIsMeasuredInPiecesThatAddUp()
    {
        // Past one piece of work, with a value out of range in the second.
        var values = new float[(1 << 20) + 5];
        Array.Fill(values, 0.25f);
        values[^1] = 1 << 20;
        NarrowingLoss loss = NarrowingLoss.Measure(BFloat16(values), HostDataType.Float16);
        Assert.That(loss.Elements, Is.EqualTo(values.Length));
        Assert.That(loss.Overflowed, Is.EqualTo(1));
        Assert.That(loss.Energy, Is.EqualTo((values.Length - 1) * 0.0625 + Math.Pow(2, 40)));
    }

    [Test]
    public void QuantizationErrorIsBoundedByHalfAStep()
    {
        var random = new Random(1);
        float[] values = Enumerable.Range(0, 4 * 64).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        (double error, double energy) = NarrowingLoss.Quantization(HostTensor.FromFloats(values, 4, 64), 4);
        Assert.That(Math.Sqrt(error / energy), Is.GreaterThan(0).And.LessThan(1.0 / 127));
    }

    /// <summary>The command reads what the family's loader hands the graphs
    /// — the adapter left out of Anima's — and names the tensor that leaves
    /// the range on stderr, with exit code 0: it is a report, not a refusal.</summary>
    [Test]
    public void InspectNamesTheTensorThatLeavesTheRange()
    {
        using var folder = new TempFolder();
        var files = new FakeSafetensors(folder);
        string anima = files.Write("anima", FakeSafetensors.AnimaDit().Append(
            new Tensor("net.blocks.0.self_attn.q_norm.weight", new[] { 4 }, "BF16",
                new[] { 1f, 2f, 1e5f, 3f })));

        var names = new List<string>();
        AnimaFamily.Instance.ReadDenoiserWeights(anima, weights =>
        {
            Assert.That(weights.DataType, Is.EqualTo(HostDataType.Float16));
            names.AddRange(weights.Keys);
            Assert.That(weights.Unnarrowed("blocks.0.self_attn.q_norm.weight").DataType,
                Is.EqualTo(HostDataType.BFloat16));
        });
        Assert.That(names, Has.None.StartsWith(AnimaCheckpoint.AdapterPrefix));

        Command inspect = Commands.Find("inspect")!;
        CommandLine line = CommandLine.Parse(new[] { "inspect", "--checkpoint", anima },
            Commands.ValueOptionsOf(inspect), inspect.FlagOptions);
        TextWriter error = Console.Error, output = Console.Out;
        var captured = new StringWriter();
        Console.SetError(captured);
        Console.SetOut(TextWriter.Null);
        try
        {
            Assert.That(inspect.Run(null, line), Is.Zero);
        }
        finally
        {
            Console.SetError(error);
            Console.SetOut(output);
        }
        Assert.That(captured.ToString(), Does.Contain("warning").And.Contain("blocks.0.self_attn.q_norm.weight"));
    }
}
