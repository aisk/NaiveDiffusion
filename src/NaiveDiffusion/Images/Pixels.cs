namespace NaiveDiffusion.Images;

/// <summary>Converting between the layouts the models speak and the ones image
/// encoders and display surfaces want. The models produce planar CHW floats in
/// [-1, 1]; everything downstream wants packed 8-bit rows.</summary>
public static class Pixels
{
    /// <summary>One channel value, [-1, 1] to a byte, rounded and clamped.</summary>
    public static byte ToByte(float value) =>
        (byte)Math.Clamp((value * 0.5f + 0.5f) * 255.0f + 0.5f, 0.0f, 255.0f);

    /// <summary>[1, 3, H, W] in [-1, 1] to packed RGB24 rows.</summary>
    public static byte[] ToRgb24(float[] image, int height, int width)
    {
        int plane = height * width;
        var rgb = new byte[plane * 3];
        for (int pixel = 0; pixel < plane; pixel++)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                rgb[pixel * 3 + channel] = ToByte(image[channel * plane + pixel]);
            }
        }
        return rgb;
    }

    /// <summary>Packed RGB24 rows to [1, 3, H, W] in [-1, 1]: the inverse of
    /// <see cref="ToRgb24"/>, and what the VAE encoder eats.</summary>
    public static float[] FromRgb24(byte[] rgb, int height, int width)
    {
        int plane = height * width;
        var image = new float[plane * 3];
        for (int pixel = 0; pixel < plane; pixel++)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                image[channel * plane + pixel] = rgb[pixel * 3 + channel] / 255.0f * 2.0f - 1.0f;
            }
        }
        return image;
    }

    /// <summary>Packed RGB24 rows to packed 32-bit rows with an opaque alpha.
    /// <paramref name="swapRedBlue"/> makes it BGRA, which is what WinUI's
    /// WriteableBitmap wants; without it the order stays RGBA, which is what the
    /// PNG encoder wants.</summary>
    public static byte[] ToFourChannel(byte[] rgb, bool swapRedBlue)
    {
        int pixels = rgb.Length / 3;
        int red = swapRedBlue ? 2 : 0;
        int blue = swapRedBlue ? 0 : 2;
        var expanded = new byte[pixels * 4];
        for (int i = 0; i < pixels; i++)
        {
            expanded[i * 4 + red] = rgb[i * 3 + 0];
            expanded[i * 4 + 1] = rgb[i * 3 + 1];
            expanded[i * 4 + blue] = rgb[i * 3 + 2];
            expanded[i * 4 + 3] = 0xFF;
        }
        return expanded;
    }

    /// <summary>Packed RGB24 rows resized to cover <paramref name="width"/> by
    /// <paramref name="height"/>: scaled so the shorter side fits, then cropped
    /// to the centre. Reference images arrive at whatever size the camera or
    /// the last run gave them, and the pipeline wants exactly the size it is
    /// generating at.
    ///
    /// A separable triangle filter whose support widens with the shrink
    /// factor, so a large photo scaled to a megapixel averages over every
    /// source pixel it covers rather than sampling one in four; upscaling
    /// degenerates to plain bilinear. Same input, same output for every
    /// caller — one resampler, so a size picked in any gives the same
    /// crop.</summary>
    public static byte[] Resample(byte[] rgb, int sourceWidth, int sourceHeight,
        int width, int height)
    {
        if (sourceWidth == width && sourceHeight == height)
        {
            return rgb;
        }
        double scale = Math.Max((double)width / sourceWidth, (double)height / sourceHeight);
        double offsetX = (sourceWidth - width / scale) / 2.0;
        double offsetY = (sourceHeight - height / scale) / 2.0;

        // Horizontal pass over every source row, then vertical over the result;
        // the crop is folded into where the sample centres land.
        Tap[] columns = Taps(sourceWidth, width, scale, offsetX);
        Tap[] rows = Taps(sourceHeight, height, scale, offsetY);

        var across = new float[sourceHeight * width * 3];
        for (int y = 0; y < sourceHeight; y++)
        {
            int sourceRow = y * sourceWidth * 3;
            int row = y * width * 3;
            for (int x = 0; x < width; x++)
            {
                Tap tap = columns[x];
                float r = 0, g = 0, b = 0;
                for (int i = 0; i < tap.Weights.Length; i++)
                {
                    int at = sourceRow + (tap.Start + i) * 3;
                    float weight = tap.Weights[i];
                    r += rgb[at] * weight;
                    g += rgb[at + 1] * weight;
                    b += rgb[at + 2] * weight;
                }
                across[row + x * 3] = r;
                across[row + x * 3 + 1] = g;
                across[row + x * 3 + 2] = b;
            }
        }

        var result = new byte[height * width * 3];
        for (int y = 0; y < height; y++)
        {
            Tap tap = rows[y];
            int row = y * width * 3;
            for (int x = 0; x < width * 3; x++)
            {
                float value = 0;
                for (int i = 0; i < tap.Weights.Length; i++)
                {
                    value += across[(tap.Start + i) * width * 3 + x] * tap.Weights[i];
                }
                result[row + x] = (byte)Math.Clamp(value + 0.5f, 0.0f, 255.0f);
            }
        }
        return result;
    }

    /// <summary>Which source samples one destination sample along an axis is
    /// made of, and how much of each.</summary>
    private readonly record struct Tap(int Start, float[] Weights);

    private static Tap[] Taps(int sourceLength, int length, double scale, double offset)
    {
        // Support of one source pixel when enlarging, of 1/scale when shrinking:
        // the filter always spans the source pixels a destination pixel covers.
        double support = Math.Max(1.0, 1.0 / scale);
        var taps = new Tap[length];
        for (int i = 0; i < length; i++)
        {
            double centre = offset + (i + 0.5) / scale;
            int first = Math.Max(0, (int)Math.Floor(centre - support));
            int last = Math.Min(sourceLength - 1, (int)Math.Ceiling(centre + support));
            var weights = new float[last - first + 1];
            double total = 0;
            for (int j = first; j <= last; j++)
            {
                double weight = Math.Max(0.0, 1.0 - Math.Abs(j + 0.5 - centre) / support);
                weights[j - first] = (float)weight;
                total += weight;
            }
            for (int j = 0; j < weights.Length; j++)
            {
                weights[j] = (float)(weights[j] / total);
            }
            taps[i] = new Tap(first, weights);
        }
        return taps;
    }
}
