namespace NaiveDiffusion.Images;

/// <summary>An image as packed RGB24 rows: what the pipeline produces, what a
/// reference image arrives as, and what callers read and write.</summary>
public sealed record ImageResult(int Width, int Height, byte[] Rgb24)
{
    /// <summary>The image as PNG bytes. <paramref name="parameters"/>, when
    /// given, goes into the text chunk viewers and other generators read the
    /// settings from; <see cref="Pipeline.A1111Parameters"/> formats one.</summary>
    public byte[] ToPng(string? parameters = null)
    {
        byte[] png = PngEncoder.Encode(Width, Height, Rgb24);
        return parameters is null ? png : PngText.Insert(png, "parameters", parameters);
    }

    /// <summary>Write the image to a PNG file, replacing one already there.</summary>
    public void SavePng(string path, string? parameters = null) =>
        File.WriteAllBytes(path, ToPng(parameters));

    /// <summary>Packed 32-bit rows in blue, green, red, alpha order with an
    /// opaque alpha: the layout WriteableBitmap, WPF's Bgra32 and
    /// System.Drawing's Format32bppArgb share.</summary>
    public byte[] ToBgra32() => Pixels.ToFourChannel(Rgb24, swapRedBlue: true);

    /// <summary>Packed 32-bit rows in red, green, blue, alpha order with an
    /// opaque alpha.</summary>
    public byte[] ToRgba32() => Pixels.ToFourChannel(Rgb24, swapRedBlue: false);
}
