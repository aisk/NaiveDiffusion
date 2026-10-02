using NaiveDiffusion.Images;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;

namespace NaiveDiffusion.Tests;

/// <summary>The PNG the library writes on its own, read back by another
/// decoder: the same pixels, whatever filter each row took, and the
/// parameters where a reader looks for them.</summary>
public class PngEncoderTests
{
    /// <summary>Smooth rows, noisy rows and flat rows, so that every filter
    /// wins somewhere.</summary>
    private static ImageResult Sample(int width, int height)
    {
        var random = new Random(7);
        var rgb = new byte[width * height * 3];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int at = (y * width + x) * 3;
                switch (y % 4)
                {
                    case 0:
                        rgb[at] = (byte)(x * 5);
                        rgb[at + 1] = (byte)(y * 9);
                        rgb[at + 2] = (byte)(x + y);
                        break;
                    case 1:
                        random.NextBytes(rgb.AsSpan(at, 3));
                        break;
                    case 2:
                        rgb[at] = rgb[at + 1] = rgb[at + 2] = 200;
                        break;
                    default:
                        rgb.AsSpan(at - width * 3, 3).CopyTo(rgb.AsSpan(at, 3));
                        break;
                }
            }
        }
        return new ImageResult(width, height, rgb);
    }

    [TestCase(1, 1)]
    [TestCase(37, 21)]
    [TestCase(256, 64)]
    public void AnotherDecoderReadsTheSamePixels(int width, int height)
    {
        ImageResult image = Sample(width, height);

        using Image<Rgb24> decoded = Image.Load<Rgb24>(image.ToPng());

        Assert.That(decoded.Width, Is.EqualTo(width));
        Assert.That(decoded.Height, Is.EqualTo(height));
        var pixels = new byte[width * height * 3];
        decoded.CopyPixelDataTo(pixels);
        Assert.That(pixels, Is.EqualTo(image.Rgb24));
    }

    [Test]
    public void TheParametersArriveAsText()
    {
        byte[] png = Sample(16, 16).ToPng("a prompt\nSteps: 20, Seed: 1");

        using Image<Rgb24> decoded = Image.Load<Rgb24>(png);

        var text = decoded.Metadata.GetPngMetadata().TextData.Single();
        Assert.That(text.Keyword, Is.EqualTo("parameters"));
        Assert.That(text.Value, Is.EqualTo("a prompt\nSteps: 20, Seed: 1"));
    }

    [Test]
    public void SavePngWritesTheFile()
    {
        string path = Path.Combine(Path.GetTempPath(), "NaiveDiffusion.PngEncoderTests",
            Guid.NewGuid().ToString("N") + ".png");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        try
        {
            ImageResult image = Sample(8, 8);
            image.SavePng(path);
            Assert.That(File.ReadAllBytes(path), Is.EqualTo(image.ToPng()));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Test]
    public void TheFourChannelLayoutsDifferOnlyInOrder()
    {
        var image = new ImageResult(1, 1, new byte[] { 10, 20, 30 });

        Assert.That(image.ToBgra32(), Is.EqualTo(new byte[] { 30, 20, 10, 255 }));
        Assert.That(image.ToRgba32(), Is.EqualTo(new byte[] { 10, 20, 30, 255 }));
    }

    [Test]
    public void ABufferOfTheWrongSizeIsRefused() =>
        Assert.That(() => Images.PngEncoder.Encode(4, 4, new byte[10]), Throws.ArgumentException);
}
