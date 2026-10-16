namespace NaiveDiffusion.Sampling;

/// <summary>A1111's variation seed: a run's starting noise turned part of
/// the way toward another seed's, so a small amount keeps the image and
/// changes its details.</summary>
public static class NoiseVariation
{
    /// <summary>Spherical interpolation from <paramref name="from"/> toward
    /// <paramref name="to"/> by <paramref name="amount"/> in [0, 1]. Two
    /// draws of independent noise are all but orthogonal, and a straight
    /// line between them passes closer to the origin: halfway it has
    /// seven tenths of the length, and the sampler would start from noise
    /// weaker than the level it assumes. Along the arc the length holds.
    /// Over the whole latent at once, not per row as A1111 does it; the
    /// seeds do not match A1111's anyway.</summary>
    public static float[] Slerp(float[] from, float[] to, float amount)
    {
        if (from.Length != to.Length)
        {
            throw new ArgumentException("the two noises must be the same length");
        }
        double dot = 0, fromNorm = 0, toNorm = 0;
        for (int i = 0; i < from.Length; i++)
        {
            dot += (double)from[i] * to[i];
            fromNorm += (double)from[i] * from[i];
            toNorm += (double)to[i] * to[i];
        }
        double cosine = dot / Math.Sqrt(fromNorm * toNorm);

        double fromWeight, toWeight;
        if (cosine > 0.9995 || double.IsNaN(cosine))
        {
            // Nearly the same direction, where the arc is a line and its
            // formula divides by nearly zero; or a noise of no length,
            // which has no direction. Nearly opposite noises would divide
            // by nearly zero too, but two independent draws never are.
            fromWeight = 1 - amount;
            toWeight = amount;
        }
        else
        {
            double angle = Math.Acos(cosine);
            double sine = Math.Sin(angle);
            fromWeight = Math.Sin((1 - amount) * angle) / sine;
            toWeight = Math.Sin(amount * angle) / sine;
        }

        var blended = new float[from.Length];
        for (int i = 0; i < from.Length; i++)
        {
            blended[i] = (float)(fromWeight * from[i] + toWeight * to[i]);
        }
        return blended;
    }
}
