using NaiveDiffusion.Vae;

namespace NaiveDiffusion.Tests;

/// <summary>The tile grid the tiled VAE runs on and the blend between
/// neighbours, fixed here: a 1024² image's 128-cell latent is three 64-cell
/// tiles starting at 0, 32 and 64, and two ramps meeting must sum to one for
/// the seams to disappear.</summary>
public class VaeTilingTests
{
    [Test]
    public void AnImageOfOneTileOrLessIsOneTileAtTheOrigin()
    {
        Assert.That(VaeTiling.TileStarts(64, 64, 8), Is.EqualTo(new[] { 0 }));
        Assert.That(VaeTiling.TileStarts(32, 64, 8), Is.EqualTo(new[] { 0 }));
    }

    [Test]
    public void A1024ImageIsThreeTilesOverlappingByHalf()
    {
        Assert.That(VaeTiling.TileStarts(128, 64, 8), Is.EqualTo(new[] { 0, 32, 64 }));
        // The overlap only sets the count; the starts are spread evenly.
        Assert.That(VaeTiling.TileStarts(128, 64, 16), Is.EqualTo(new[] { 0, 32, 64 }));
        Assert.That(VaeTiling.TileStarts(96, 64, 8), Is.EqualTo(new[] { 0, 32 }));
        // Enough tiles to keep every overlap at least the asked-for width,
        // spread evenly: five over 256 cells, sharing 16.
        Assert.That(VaeTiling.TileStarts(256, 64, 8), Is.EqualTo(new[] { 0, 48, 96, 144, 192 }));
    }

    [Test]
    public void EveryTileEndsInsideTheImage()
    {
        foreach (int extent in new[] { 64, 72, 100, 128, 200, 256, 257 })
        {
            int[] starts = VaeTiling.TileStarts(extent, 64, 8);
            Assert.That(starts[0], Is.Zero);
            Assert.That(starts[^1] + 64, Is.LessThanOrEqualTo(extent), extent.ToString());
            Assert.That(starts, Is.Ordered);
        }
    }

    [Test]
    public void TwoRampsMeetingSumToOneAndASingleTileIsAllOnes()
    {
        Assert.That(VaeTiling.Crossfade(new[] { 0 }, 8), Is.All.EqualTo(1f));

        int[] starts = { 0, 4 };
        float[] ramp = VaeTiling.Crossfade(starts, 8);
        // Tiles [0, 8) and [4, 12) share [4, 8): the tail of the first and
        // the head of the second add up to one at every shared cell.
        for (int i = 0; i < 4; i++)
        {
            Assert.That(ramp[4 + i] + ramp[i], Is.EqualTo(1f).Within(1e-6f), i.ToString());
        }
        Assert.That(ramp[0], Is.LessThan(ramp[3]), "rising at the head");
        Assert.That(ramp[7], Is.LessThan(ramp[4]), "falling at the tail");
    }

    [Test]
    public void TheWindowIsTheProductOfTheTwoAxes()
    {
        float[] window = VaeTiling.Window(new[] { 0, 4 }, new[] { 0 }, 8);
        float[] vertical = VaeTiling.Crossfade(new[] { 0, 4 }, 8);
        for (int y = 0; y < 8; y++)
        {
            for (int x = 0; x < 8; x++)
            {
                Assert.That(window[y * 8 + x], Is.EqualTo(vertical[y]).Within(1e-6f));
            }
        }
    }
}
