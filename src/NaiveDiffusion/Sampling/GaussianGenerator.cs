namespace NaiveDiffusion.Sampling;

/// <summary>Standard normal variates from a seeded generator, via Box-Muller.
/// Not NumPy-compatible; seeds are only reproducible against this library.</summary>
public sealed class GaussianGenerator
{
    private readonly Random _random;
    private double? _spare;

    public GaussianGenerator(int seed) => _random = new Random(seed);

    public float Next()
    {
        if (_spare is double spare)
        {
            _spare = null;
            return (float)spare;
        }
        double u1 = 1.0 - _random.NextDouble();
        double u2 = _random.NextDouble();
        double radius = Math.Sqrt(-2.0 * Math.Log(u1));
        double angle = 2.0 * Math.PI * u2;
        _spare = radius * Math.Sin(angle);
        return (float)(radius * Math.Cos(angle));
    }

    public float[] Fill(int count)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = Next();
        }
        return values;
    }
}
