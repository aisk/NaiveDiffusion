using NaiveDiffusion.Models.Anima;
using NaiveDiffusion.Models.Sdxl;
using NaiveDiffusion.Pipeline;

namespace NaiveDiffusion.Tests;

/// <summary>The auto budget's arithmetic: what is left after the process's
/// own usage, the family's scratch, the streaming window and the margin —
/// and nothing, rather than a wrapped-around number, when that is more than
/// the card offers.</summary>
public class DenoiserBudgetTests
{
    private const ulong Mib = 1UL << 20;

    [Test]
    public void TheScratchIsTheFamilysOwnAndGrowsWithTheSize()
    {
        var square = new GenerationOptions { Width = 1024, Height = 1024 };
        Assert.That(SdxlFamily.Instance.DenoiserScratchBytes(square), Is.EqualTo(512 * Mib));
        Assert.That(SdxlFamily.Instance.DenoiserScratchBytes(square with { DenoiserInt8Weights = true }),
            Is.EqualTo(768 * Mib));
        Assert.That(AnimaFamily.Instance.DenoiserScratchBytes(square), Is.EqualTo(2048 * Mib));
        Assert.That(AnimaFamily.Instance.DenoiserScratchBytes(square with { Width = 2048 }),
            Is.EqualTo(4096 * Mib));
        Assert.That(SdxlFamily.Instance.DenoiserScratchBytes(square with { Width = 256, Height = 256 }),
            Is.EqualTo(256 * Mib), "never under a quarter of a gigabyte");
    }

    [Test]
    public void WhatIsLeftIsTheBudgetLessEverythingReserved()
    {
        // 16 GiB offered, 1 GiB held: less 512 scratch, 1024 window and a
        // fifth of the budget for margin.
        Assert.That(DenoiserBudget.FromReadings(1024 * Mib, 16384 * Mib, 512 * Mib),
            Is.EqualTo((16384 - 1024 - 512 - 1024 - 16384 / 5.0) * Mib).Within(Mib));
        // A small budget's margin is a gigabyte, not a fifth.
        Assert.That(DenoiserBudget.FromReadings(0, 4096 * Mib, 512 * Mib),
            Is.EqualTo((4096 - 512 - 1024 - 1024) * Mib));
        Assert.That(DenoiserBudget.FromReadings(0, 4096 * Mib, 2048 * Mib), Is.EqualTo(0UL),
            "stream everything rather than wrap around");
        Assert.That(DenoiserBudget.FromReadings(0, 0, 512 * Mib), Is.EqualTo(0UL),
            "an adapter that reports no budget");
    }
}
