using NaiveDiffusion.Sampling;

namespace NaiveDiffusion.Tests;

/// <summary>The variation seed's blend: from one noise to the other along
/// the arc, keeping the length a sampler expects of its starting noise.</summary>
public class NoiseVariationTests
{
    private const int Length = 4 * 64 * 64;

    private static readonly float[] Seed = new GaussianGenerator(1).Fill(Length);
    private static readonly float[] Variation = new GaussianGenerator(2).Fill(Length);

    [Test]
    public void TheEndsAreTheTwoNoises()
    {
        Assert.That(NoiseVariation.Slerp(Seed, Variation, 0), Is.EqualTo(Seed).Within(1e-5f));
        Assert.That(NoiseVariation.Slerp(Seed, Variation, 1), Is.EqualTo(Variation).Within(1e-5f));
    }

    [Test]
    public void TheLengthHoldsAllTheWay()
    {
        // Two draws differ in length by a fraction of a percent, and the
        // arc goes from one to the other; a straight line would lose
        // three tenths of it halfway.
        double expected = Norm(Seed);
        foreach (float amount in new[] { 0.1f, 0.3f, 0.5f, 0.9f })
        {
            Assert.That(Norm(NoiseVariation.Slerp(Seed, Variation, amount)),
                Is.EqualTo(expected).Within(0.5).Percent, $"at {amount}");
        }
        float[] line = Seed.Zip(Variation, (a, b) => 0.5f * (a + b)).ToArray();
        Assert.That(Norm(line) / expected, Is.EqualTo(Math.Sqrt(0.5)).Within(0.01));
    }

    [Test]
    public void ASmallAmountStaysNearTheSeed()
    {
        float[] blended = NoiseVariation.Slerp(Seed, Variation, 0.1f);
        Assert.That(Cosine(blended, Seed), Is.GreaterThan(0.98));
        Assert.That(Cosine(blended, Variation), Is.LessThan(0.2));
    }

    [Test]
    public void TheSameNoiseBlendsToItself()
    {
        Assert.That(NoiseVariation.Slerp(Seed, Seed, 0.4f), Is.EqualTo(Seed).Within(1e-5f));
    }

    [Test]
    public void ANoiseOfNoLengthIsBlendedAlongTheLine()
    {
        float[] blended = NoiseVariation.Slerp(new float[Length], Variation, 0.5f);
        Assert.That(blended, Is.EqualTo(Variation.Select(value => 0.5f * value)).Within(1e-6f));
    }

    private static double Norm(float[] values) => Math.Sqrt(values.Sum(value => (double)value * value));

    private static double Cosine(float[] a, float[] b) =>
        a.Zip(b, (x, y) => (double)x * y).Sum() / (Norm(a) * Norm(b));
}
