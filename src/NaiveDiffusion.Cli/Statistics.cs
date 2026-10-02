namespace NaiveDiffusion.Cli;

/// <summary>The numbers the verification commands print. Nothing here feeds a
/// model; it all exists to be compared against the Python reference by eye.</summary>
internal static class Statistics
{
    public static float Std(float[] values)
    {
        double mean = values.Average();
        double variance = values.Sum(value => (value - mean) * (value - mean)) / values.Length;
        return (float)Math.Sqrt(variance);
    }

    /// <summary>Peak signal-to-noise ratio between two [-1, 1] images, in dB.</summary>
    public static float Psnr(float[] a, float[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double left = Math.Clamp(a[i] * 0.5 + 0.5, 0, 1);
            double right = Math.Clamp(b[i] * 0.5 + 0.5, 0, 1);
            sum += (left - right) * (left - right);
        }
        return (float)(10.0 * Math.Log10(1.0 / Math.Max(sum / a.Length, 1e-12)));
    }
}
