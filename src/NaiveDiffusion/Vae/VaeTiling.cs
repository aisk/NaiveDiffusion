namespace NaiveDiffusion.Vae;

/// <summary>The geometry the tiled VAE shares between its two halves: where
/// the tiles start, and how two neighbours are blended where they overlap.
/// Units are whatever the caller works in — pixels on the decoder's output,
/// latent cells on the encoder's.</summary>
internal static class VaeTiling
{
    /// <summary>Evenly spaced tile origins covering <paramref name="extent"/>,
    /// every tile the same size — one compiled graph then serves every tile,
    /// and the seams come out equally wide.</summary>
    public static int[] TileStarts(int extent, int tile, int overlap)
    {
        if (extent <= tile)
        {
            return new[] { 0 };
        }
        int count = ((extent - overlap) + (tile - overlap) - 1) / (tile - overlap);
        int span = extent - tile;
        var starts = new int[count];
        for (int i = 0; i < count; i++)
        {
            starts[i] = (int)Math.Round((double)i * span / (count - 1));
        }
        return starts;
    }

    /// <summary>Per-axis weights: a linear ramp across whatever the tiles
    /// actually share.</summary>
    public static float[] Crossfade(int[] starts, int tile)
    {
        var weight = new float[tile];
        Array.Fill(weight, 1.0f);
        if (starts.Length > 1)
        {
            int step = int.MaxValue;
            for (int i = 1; i < starts.Length; i++)
            {
                step = Math.Min(step, starts[i] - starts[i - 1]);
            }
            int overlap = tile - step;
            for (int i = 0; i < overlap; i++)
            {
                float ramp = (float)(i + 1) / (overlap + 1);
                weight[i] = ramp;
                weight[tile - 1 - i] = ramp;
            }
        }
        return weight;
    }

    /// <summary>The side by side blend one tile contributes with: the two
    /// axes' ramps multiplied. Two crossfades meeting sum to one; a ramp on
    /// the outer border has no neighbour to complete it, so whoever adds the
    /// tiles up also keeps the sum of the weights and divides it back out.</summary>
    public static float[] Window(int[] rows, int[] columns, int side)
    {
        float[] vertical = Crossfade(rows, side);
        float[] horizontal = Crossfade(columns, side);
        var window = new float[side * side];
        for (int y = 0; y < side; y++)
        {
            for (int x = 0; x < side; x++)
            {
                window[y * side + x] = vertical[y] * horizontal[x];
            }
        }
        return window;
    }
}
