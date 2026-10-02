using NaiveDiffusion.Tensors;
using NaiveDiffusion.Graph;

namespace NaiveDiffusion.Vae;

/// <summary>What the tiled decoder and the tiled encoder share: one
/// tile-sized graph, the grid of tile origins it is run at, and the
/// crossfade that blends neighbours where they overlap.
///
/// Running a VAE in tiles replaces the largest single allocation a run makes
/// — every intermediate of a whole-image pass is a full-size image, the
/// widest at hundreds of channels — with a fixed cost per tile; what grows
/// with the image is the number of dispatches, not any allocation.
///
/// It is an approximation, not an equivalent rewrite: GroupNorm normalizes
/// over the whole feature map, so a tile run alone sees different statistics
/// than the same region does inside a whole-image pass. Overlapping the tiles
/// and crossfading between them is what hides the difference, as diffusers
/// does. An image of one tile or less goes through exactly as the whole-image
/// graph would.</summary>
public abstract class TiledVae : IDisposable
{
    public const int TileLatent = 64;
    public const int TileOverlap = 8;

    private readonly ModelBuilder _model;

    protected LatentSpace Latent { get; }
    protected int Height { get; }
    protected int Width { get; }

    /// <summary>The tile's side in latent cells.</summary>
    protected int Tile { get; }

    /// <summary>Tile origins on the latent grid, top edges and left edges.</summary>
    protected int[] Rows { get; }
    protected int[] Columns { get; }

    /// <param name="build">Compiles the graph for one square tile of the
    /// side it is given, in pixels; the tiling is the same whatever the
    /// graph inside is.</param>
    protected TiledVae(LatentSpace latent, Func<int, ModelBuilder> build, int height, int width,
        int tile, int overlap)
    {
        Latent = latent;
        Height = height;
        Width = width;
        Tile = Math.Min(tile, Math.Min(height, width) / latent.ScaleFactor);
        overlap = Math.Min(overlap, Tile / 2);
        Rows = VaeTiling.TileStarts(height / latent.ScaleFactor, Tile, overlap);
        Columns = VaeTiling.TileStarts(width / latent.ScaleFactor, Tile, overlap);
        _model = build(TileSide);
    }

    /// <summary>The tile's side in pixels.</summary>
    public int TileSide => Tile * Latent.ScaleFactor;

    public ulong TemporarySize => _model.TemporarySize;

    protected ModelBuilder Model => _model;

    /// <summary>The blend one tile contributes with, on the grid the output
    /// lives on: <paramref name="scale"/> is 1 for the latent grid and the
    /// latent scale factor for pixels.</summary>
    protected float[] Window(int scale) => VaeTiling.Window(
        Rows.Select(row => row * scale).ToArray(),
        Columns.Select(column => column * scale).ToArray(), Tile * scale);

    /// <summary>Copy one square tile out of a [C, H, W] plane stack into a
    /// packed [C, side, side] piece.</summary>
    protected static void CopyTile(float[] source, int channels, int sourceWidth,
        int sourcePlane, int top, int left, int side, float[] piece)
    {
        for (int channel = 0; channel < channels; channel++)
        {
            for (int y = 0; y < side; y++)
            {
                Array.Copy(source, channel * sourcePlane + (top + y) * sourceWidth + left,
                    piece, (channel * side + y) * side, side);
            }
        }
    }

    /// <summary>Add one tile's result into the [C, H, W] output, weighted by
    /// the window, keeping the sum of the weights per cell alongside.</summary>
    protected static void BlendTile(float[] target, float[] weight, int channels,
        int targetWidth, int targetPlane, int top, int left, int side, float[] tile,
        float[] window)
    {
        for (int y = 0; y < side; y++)
        {
            int targetRow = (top + y) * targetWidth + left;
            int windowRow = y * side;
            for (int x = 0; x < side; x++)
            {
                float blend = window[windowRow + x];
                for (int channel = 0; channel < channels; channel++)
                {
                    target[channel * targetPlane + targetRow + x] +=
                        tile[(channel * side + y) * side + x] * blend;
                }
                weight[targetRow + x] += blend;
            }
        }
    }

    /// <summary>Two crossfades meeting sum to one, but a ramp on the outer
    /// border has no neighbour to complete it, so the weights are divided
    /// back out.</summary>
    protected static void DivideByWeights(float[] target, float[] weight, int channels, int plane)
    {
        for (int cell = 0; cell < plane; cell++)
        {
            float scale = 1.0f / weight[cell];
            for (int channel = 0; channel < channels; channel++)
            {
                target[channel * plane + cell] *= scale;
            }
        }
    }

    public void Dispose() => _model.Dispose();
}

/// <summary>A VAE decoder that runs one tile-sized graph over the whole
/// latent, the tiles' pixels crossfaded where they overlap.</summary>
public sealed class TiledVaeDecoder : TiledVae
{
    private readonly float[] _window;   // side x side crossfade weights, in pixels

    public TiledVaeDecoder(LatentSpace latent, Func<int, ModelBuilder> build, int height,
        int width, int tile = TileLatent, int overlap = TileOverlap)
        : base(latent, build, height, width, tile, overlap)
    {
        _window = Window(Latent.ScaleFactor);
    }

    /// <summary>Same contract as the whole-image decoder: one unscaled latent
    /// [1, C, H/f, W/f] in, one image [1, 3, H, W] out.</summary>
    public float[] Run(float[] latent)
    {
        int latentWidth = Width / Latent.ScaleFactor;
        int latentPlane = Height / Latent.ScaleFactor * latentWidth;
        int side = TileSide;
        int imagePlane = Height * Width;

        var image = new float[3 * imagePlane];
        var weight = new float[imagePlane];
        var piece = new float[Latent.Channels * Tile * Tile];
        foreach (int top in Rows)
        {
            foreach (int left in Columns)
            {
                CopyTile(latent, Latent.Channels, latentWidth, latentPlane, top, left, Tile, piece);
                float[] decoded = Model.Run(HostTensor.FromFloats(piece,
                    1, Latent.Channels, Tile, Tile))[0].ToFloats();
                BlendTile(image, weight, 3, Width, imagePlane,
                    top * Latent.ScaleFactor, left * Latent.ScaleFactor, side, decoded, _window);
            }
        }
        DivideByWeights(image, weight, 3, imagePlane);
        return image;
    }
}

/// <summary>The encoder half: one tile-sized graph run over the whole
/// reference image, the tiles' latents crossfaded where they overlap on the
/// latent grid. The encoder's intermediates are full-size images at up to
/// 128 channels before the first downsample, which makes a whole-image
/// encode the largest allocation an image-to-image run makes before the
/// diffusion model — and it runs while a cached one may still be resident.
/// Tiling caps it at one tile's worth however large the reference is.</summary>
public sealed class TiledVaeEncoder : TiledVae
{
    private readonly float[] _window;   // tile x tile crossfade weights, on the latent grid

    public TiledVaeEncoder(LatentSpace latent, Func<int, ModelBuilder> build, int height,
        int width, int tile = TileLatent, int overlap = TileOverlap)
        : base(latent, build, height, width, tile, overlap)
    {
        _window = Window(1);
    }

    /// <summary>Same contract as the whole-image encoder: one image [1, 3, H, W]
    /// in, the unscaled latent mean [1, C, H/f, W/f] out.</summary>
    public float[] Run(float[] image)
    {
        int side = TileSide;
        int imagePlane = Height * Width;
        int latentWidth = Width / Latent.ScaleFactor;
        int latentPlane = Height / Latent.ScaleFactor * latentWidth;

        var latent = new float[Latent.Channels * latentPlane];
        var weight = new float[latentPlane];
        var piece = new float[3 * side * side];
        foreach (int top in Rows)
        {
            foreach (int left in Columns)
            {
                CopyTile(image, 3, Width, imagePlane,
                    top * Latent.ScaleFactor, left * Latent.ScaleFactor, side, piece);
                float[] encoded = Model.Run(HostTensor.FromFloats(piece, 1, 3, side, side))[0]
                    .ToFloats();
                BlendTile(latent, weight, Latent.Channels, latentWidth, latentPlane,
                    top, left, Tile, encoded, _window);
            }
        }
        DivideByWeights(latent, weight, Latent.Channels, latentPlane);
        return latent;
    }
}
