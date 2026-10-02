using System.IO.Hashing;
using System.Text;
using NaiveDiffusion.Images;

namespace NaiveDiffusion.Tests;

/// <summary>The text chunk that carries an image's parameters: after the
/// header, with the checksum a reader will verify, as tEXt for Latin-1 and
/// iTXt for anything else — and a buffer that is not a PNG untouched, since
/// this runs on the save path.</summary>
public class PngTextTests
{
    private static readonly byte[] Signature = { 0x89, (byte)'P', (byte)'N', (byte)'G', 13, 10, 26, 10 };

    /// <summary>A PNG down to its header chunk and one more chunk after it.</summary>
    private static byte[] MinimalPng()
    {
        var png = new List<byte>(Signature);
        png.AddRange(Chunk("IHDR", new byte[13]));
        png.AddRange(Chunk("IEND", Array.Empty<byte>()));
        return png.ToArray();
    }

    private static byte[] Chunk(string type, byte[] data)
    {
        byte[] typeBytes = Encoding.ASCII.GetBytes(type);
        var chunk = new List<byte>();
        chunk.AddRange(BigEndian((uint)data.Length));
        chunk.AddRange(typeBytes);
        chunk.AddRange(data);
        chunk.AddRange(BigEndian(Crc32.HashToUInt32(typeBytes.Concat(data).ToArray())));
        return chunk.ToArray();
    }

    private static byte[] BigEndian(uint value) =>
        new[] { (byte)(value >> 24), (byte)(value >> 16), (byte)(value >> 8), (byte)value };

    private static uint ReadBigEndian(byte[] data, int offset) =>
        (uint)(data[offset] << 24 | data[offset + 1] << 16 | data[offset + 2] << 8 | data[offset + 3]);

    [Test]
    public void ALatin1TextGoesInATextChunkRightAfterTheHeader()
    {
        byte[] png = MinimalPng();
        byte[] result = PngText.Insert(png, "parameters", "1girl\nSteps: 20");

        int at = Signature.Length + 12 + 13;
        uint length = ReadBigEndian(result, at);
        Assert.That(Encoding.ASCII.GetString(result, at + 4, 4), Is.EqualTo("tEXt"));
        byte[] body = result[(at + 8)..(at + 8 + (int)length)];
        Assert.That(Encoding.Latin1.GetString(body), Is.EqualTo("parameters\0" + "1girl\nSteps: 20"));
        // The checksum covers the type and the data and is what a reader recomputes.
        uint crc = Crc32.HashToUInt32(result[(at + 4)..(at + 8 + (int)length)]);
        Assert.That(ReadBigEndian(result, at + 8 + (int)length), Is.EqualTo(crc));
        // Everything else is where it was.
        Assert.That(result[..at], Is.EqualTo(png[..at]));
        Assert.That(result[(at + 12 + (int)length)..], Is.EqualTo(png[at..]));
    }

    [Test]
    public void OtherTextGoesInAnInternationalChunkAsUtf8()
    {
        byte[] result = PngText.Insert(MinimalPng(), "parameters", "1girl, 初音ミク");
        int at = Signature.Length + 12 + 13;
        uint length = ReadBigEndian(result, at);
        Assert.That(Encoding.ASCII.GetString(result, at + 4, 4), Is.EqualTo("iTXt"));
        byte[] body = result[(at + 8)..(at + 8 + (int)length)];
        // keyword, null, compression flag, method, language, translated keyword, text.
        Assert.That(body[..11], Is.EqualTo(Encoding.ASCII.GetBytes("parameters\0")));
        Assert.That(body[11..15], Is.EqualTo(new byte[] { 0, 0, 0, 0 }));
        Assert.That(Encoding.UTF8.GetString(body[15..]), Is.EqualTo("1girl, 初音ミク"));
    }

    [Test]
    public void ABufferThatIsNotAPngComesBackUntouched()
    {
        byte[] jpeg = { 0xFF, 0xD8, 0xFF, 0xE0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        Assert.That(PngText.Insert(jpeg, "parameters", "x"), Is.SameAs(jpeg));
        byte[] tooShort = Signature;
        Assert.That(PngText.Insert(tooShort, "parameters", "x"), Is.SameAs(tooShort));
    }
}
