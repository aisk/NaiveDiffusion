using System.IO.Compression;
using System.Text;

namespace NaiveDiffusion.Images;

/// <summary>Writing packed RGB24 rows as a PNG with nothing but the runtime's
/// zlib: 8-bit truecolour, no alpha, not interlaced. Each row takes whichever
/// of the five filters leaves the smallest residue, which is the heuristic
/// libpng uses and what makes a photograph compress at all.</summary>
public static class PngEncoder
{
    private static readonly byte[] Signature = { 0x89, (byte)'P', (byte)'N', (byte)'G', 13, 10, 26, 10 };

    private const int BytesPerPixel = 3;

    public static byte[] Encode(int width, int height, ReadOnlySpan<byte> rgb24)
    {
        if (width <= 0 || height <= 0 || rgb24.Length != width * height * BytesPerPixel)
        {
            throw new ArgumentException(
                $"{rgb24.Length} bytes is not a {width}x{height} RGB24 image", nameof(rgb24));
        }

        var header = new byte[13];
        PngText.WriteBigEndian(header, 0, (uint)width);
        PngText.WriteBigEndian(header, 4, (uint)height);
        header[8] = 8;  // bits per channel
        header[9] = 2;  // truecolour

        using var png = new MemoryStream();
        png.Write(Signature);
        WriteChunk(png, "IHDR", header);
        WriteChunk(png, "IDAT", Compress(width, height, rgb24));
        WriteChunk(png, "IEND", Array.Empty<byte>());
        return png.ToArray();
    }

    private static byte[] Compress(int width, int height, ReadOnlySpan<byte> rgb24)
    {
        int stride = width * BytesPerPixel;
        var candidates = new byte[5][];
        for (int filter = 0; filter < candidates.Length; filter++)
        {
            candidates[filter] = new byte[stride];
        }

        using var compressed = new MemoryStream();
        using (var zlib = new ZLibStream(compressed, CompressionLevel.Optimal, leaveOpen: true))
        {
            for (int y = 0; y < height; y++)
            {
                ReadOnlySpan<byte> row = rgb24.Slice(y * stride, stride);
                ReadOnlySpan<byte> above = y > 0 ? rgb24.Slice((y - 1) * stride, stride) : default;

                int best = 0;
                long bestCost = long.MaxValue;
                for (int filter = 0; filter < candidates.Length; filter++)
                {
                    long cost = Filter(filter, row, above, candidates[filter]);
                    if (cost < bestCost)
                    {
                        best = filter;
                        bestCost = cost;
                    }
                }
                zlib.WriteByte((byte)best);
                zlib.Write(candidates[best]);
            }
        }
        return compressed.ToArray();
    }

    /// <summary>One row through one of PNG's filters, and the sum of the
    /// residues read as signed bytes, which is what the rows are compared by.</summary>
    private static long Filter(int filter, ReadOnlySpan<byte> row, ReadOnlySpan<byte> above,
        Span<byte> target)
    {
        long cost = 0;
        for (int i = 0; i < row.Length; i++)
        {
            int left = i >= BytesPerPixel ? row[i - BytesPerPixel] : 0;
            int up = above.IsEmpty ? 0 : above[i];
            int upLeft = i >= BytesPerPixel && !above.IsEmpty ? above[i - BytesPerPixel] : 0;
            int predicted = filter switch
            {
                0 => 0,
                1 => left,
                2 => up,
                3 => (left + up) >> 1,
                _ => Paeth(left, up, upLeft),
            };
            byte residue = (byte)(row[i] - predicted);
            target[i] = residue;
            cost += residue < 128 ? residue : 256 - residue;
        }
        return cost;
    }

    private static int Paeth(int left, int up, int upLeft)
    {
        int estimate = left + up - upLeft;
        int fromLeft = Math.Abs(estimate - left);
        int fromUp = Math.Abs(estimate - up);
        int fromUpLeft = Math.Abs(estimate - upLeft);
        if (fromLeft <= fromUp && fromLeft <= fromUpLeft)
        {
            return left;
        }
        return fromUp <= fromUpLeft ? up : upLeft;
    }

    private static void WriteChunk(Stream png, string type, byte[] data)
    {
        var chunk = new byte[12 + data.Length];
        PngText.WriteBigEndian(chunk, 0, (uint)data.Length);
        Encoding.ASCII.GetBytes(type).CopyTo(chunk, 4);
        data.CopyTo(chunk, 8);
        // The checksum covers the type and the data, not the length.
        PngText.WriteBigEndian(chunk, 8 + data.Length, PngText.Crc32(chunk, 4, 4 + data.Length));
        png.Write(chunk);
    }
}
