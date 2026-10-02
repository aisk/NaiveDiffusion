using NaiveDiffusion.Dml;

namespace NaiveDiffusion.Pipeline;

/// <summary>A resident budget for a denoiser's weights from what the card
/// has free right now: the budget DXGI reports for this process, less what
/// it already holds, less the scratch a step at this size needs and a margin
/// for the activations, the driver's own allocations, and the budget itself
/// moving — it is not a fixed number but the OS's current opinion, and it
/// was watched falling by a tenth in the course of one run as the desktop
/// took memory back. On a card with room for everything this comes out
/// above the weights' size and nothing is streamed, so it is safe to apply
/// always; it only bites where the alternative is not fitting at all. Zero —
/// stream everything — on an adapter that will not say what its budget is.</summary>
public static class DenoiserBudget
{
    /// <summary>A family's measured scratch at 1024², scaled to the size
    /// being generated: it grows with the pixel count, and a quarter of a
    /// gigabyte is the least any size is given.</summary>
    public static ulong ScaleScratch(ulong bytesPerMegapixel, int height, int width)
    {
        double megapixels = (double)width * height / (1024.0 * 1024.0);
        return Math.Max(256UL << 20, (ulong)(bytesPerMegapixel * megapixels));
    }

    public static ulong FromFreeMemory(DmlDevice device, GenerationOptions options,
        IModelFamily family)
    {
        (ulong usage, ulong budget) = device.VideoMemory();
        return FromReadings(usage, budget,
            family.DenoiserScratchBytes(options));
    }

    /// <summary>The arithmetic, apart from the device it reads.</summary>
    public static ulong FromReadings(ulong usage, ulong budget, ulong scratch)
    {
        const ulong mib = 1UL << 20;
        if (budget == 0)
        {
            return 0;
        }
        // A streamed graph's weights pass through a window in video memory
        // the size of the largest of them, just under a gigabyte.
        ulong window = 1024 * mib;
        ulong margin = Math.Max(1024 * mib, budget / 5);
        ulong reserved = usage + scratch + window + margin;
        return budget > reserved ? budget - reserved : 0;
    }
}
