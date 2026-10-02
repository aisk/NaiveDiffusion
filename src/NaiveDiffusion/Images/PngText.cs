using System.Text;

namespace NaiveDiffusion.Images;

/// <summary>Putting a text chunk into an encoded PNG. Every generator writes the
/// settings that made an image into one, under the keyword "parameters", which
/// is how a picture handed back to a viewer — or to A1111, or to ComfyUI — still
/// knows its own prompt and seed.</summary>
public static class PngText
{
    private const int SignatureLength = 8;

    /// <summary>A copy of <paramref name="png"/> carrying one more text chunk.
    /// It goes directly after the header, ahead of the pixels, where every
    /// reader passes it. Latin-1 text goes in a tEXt chunk and anything else — a
    /// Chinese prompt, an emoji — in iTXt, which is the UTF-8 one; that is the
    /// choice PIL makes, and so the one the ecosystem knows how to read. A
    /// buffer that is not a PNG comes back untouched: this runs while saving,
    /// where losing the metadata beats losing the image.</summary>
    public static byte[] Insert(byte[] png, string keyword, string text)
    {
        if (png.Length < SignatureLength + 12 ||
            png[1] != 'P' || png[2] != 'N' || png[3] != 'G')
        {
            return png;
        }

        // The header is the first chunk after the signature; its declared length
        // says where it ends and the rest of the file begins.
        int afterHeader = SignatureLength + 12 + BigEndian(png, SignatureLength);
        if (afterHeader > png.Length)
        {
            return png;
        }

        byte[] chunk = Chunk(keyword, text);
        var result = new byte[png.Length + chunk.Length];
        Array.Copy(png, 0, result, 0, afterHeader);
        Array.Copy(chunk, 0, result, afterHeader, chunk.Length);
        Array.Copy(png, afterHeader, result, afterHeader + chunk.Length, png.Length - afterHeader);
        return result;
    }

    private static byte[] Chunk(string keyword, string text)
    {
        bool latin1 = text.All(character => character <= 0xFF);

        var body = new List<byte>();
        body.AddRange(Encoding.Latin1.GetBytes(keyword));
        body.Add(0);
        if (!latin1)
        {
            // iTXt carries a compression flag and method, then an empty language
            // tag and an empty translated keyword, before the text itself.
            body.AddRange(new byte[] { 0, 0, 0, 0 });
        }
        body.AddRange(latin1 ? Encoding.Latin1.GetBytes(text) : Encoding.UTF8.GetBytes(text));

        var chunk = new byte[12 + body.Count];
        WriteBigEndian(chunk, 0, (uint)body.Count);
        Encoding.ASCII.GetBytes(latin1 ? "tEXt" : "iTXt").CopyTo(chunk, 4);
        body.CopyTo(chunk, 8);
        // The checksum covers the type and the data, not the length.
        WriteBigEndian(chunk, 8 + body.Count, Crc32(chunk, 4, 4 + body.Count));
        return chunk;
    }

    private static int BigEndian(byte[] data, int offset) =>
        (data[offset] << 24) | (data[offset + 1] << 16) |
        (data[offset + 2] << 8) | data[offset + 3];

    private static void WriteBigEndian(byte[] target, int offset, uint value)
    {
        target[offset] = (byte)(value >> 24);
        target[offset + 1] = (byte)(value >> 16);
        target[offset + 2] = (byte)(value >> 8);
        target[offset + 3] = (byte)value;
    }

    private static readonly uint[] CrcTable = BuildCrcTable();

    private static uint[] BuildCrcTable()
    {
        var table = new uint[256];
        for (uint i = 0; i < table.Length; i++)
        {
            uint value = i;
            for (int bit = 0; bit < 8; bit++)
            {
                value = (value & 1) != 0 ? 0xEDB88320u ^ (value >> 1) : value >> 1;
            }
            table[i] = value;
        }
        return table;
    }

    private static uint Crc32(byte[] data, int offset, int count)
    {
        uint crc = 0xFFFFFFFFu;
        for (int i = offset; i < offset + count; i++)
        {
            crc = CrcTable[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
        }
        return crc ^ 0xFFFFFFFFu;
    }
}
