namespace NaiveDiffusion.Images;

/// <summary>An image as packed RGB24 rows: what the pipeline produces, what a
/// reference image arrives as, and what callers read and write.</summary>
public sealed record ImageResult(int Width, int Height, byte[] Rgb24);
