using NaiveDiffusion.Images;
using NaiveDiffusion.Pipeline;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace NaiveDiffusion.Cli;

/// <summary>Reading and writing image files, which the command line does
/// with ImageSharp. The models speak planar CHW floats in [-1, 1]; ImageSharp
/// speaks packed rows.</summary>
internal static class ImageFile
{
    /// <summary>An image file as [1, 3, size, size] in [-1, 1], centre-cropped
    /// to a square.</summary>
    public static float[] LoadSquare(string path, int size)
    {
        using Image<Rgb24> image = Image.Load<Rgb24>(path);
        image.Mutate(context => context.Resize(new ResizeOptions
        {
            Size = new Size(size, size),
            Mode = ResizeMode.Crop,
            Sampler = KnownResamplers.Bicubic,
        }));

        var chw = new float[3 * size * size];
        int plane = size * size;
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < size; y++)
            {
                Span<Rgb24> row = accessor.GetRowSpan(y);
                for (int x = 0; x < size; x++)
                {
                    int pixel = y * size + x;
                    chw[pixel] = row[x].R / 255.0f * 2.0f - 1.0f;
                    chw[plane + pixel] = row[x].G / 255.0f * 2.0f - 1.0f;
                    chw[2 * plane + pixel] = row[x].B / 255.0f * 2.0f - 1.0f;
                }
            }
        });
        return chw;
    }

    /// <summary>An image file as packed RGB24 rows at its own size, the way
    /// the pipeline takes a reference image; <see cref="Pixels.Resample"/>
    /// brings it to the size being generated. Turned the right way up by its
    /// EXIF tag before it is cropped.</summary>
    public static ImageResult LoadRgb24(string path)
    {
        using Image<Rgb24> image = Image.Load<Rgb24>(path);
        image.Mutate(context => context.AutoOrient());
        var rgb = new byte[image.Width * image.Height * 3];
        image.CopyPixelDataTo(rgb);
        return new ImageResult(image.Width, image.Height, rgb);
    }

    /// <summary>Write [1, 3, H, W] in [-1, 1] as a PNG.</summary>
    public static void SavePlanar(float[] chw, int height, int width, string path) =>
        SavePacked(Pixels.ToRgb24(chw, height, width), height, width, path);

    /// <summary>Write packed RGB24 rows as a PNG.</summary>
    public static void SavePacked(byte[] rgb, int height, int width, string path)
    {
        using var bitmap = Image.LoadPixelData<Rgb24>(rgb, width, height);
        bitmap.SaveAsPng(path);
    }
}
