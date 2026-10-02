using System.Runtime.InteropServices;

namespace NaiveDiffusion.Cli;

/// <summary>Raw float32 files, the form every tensor the CLI dumps or reads
/// back takes: no header, the values in order, as numpy's tofile writes
/// them and fromfile reads them.</summary>
internal static class Dumps
{
    public static void Write(string path, ReadOnlySpan<float> values)
    {
        File.WriteAllBytes(path, MemoryMarshal.AsBytes(values).ToArray());
        Console.WriteLine($"Wrote {path}");
    }

    /// <summary>The floats in a file, refused with the option's name when
    /// the byte count is not a whole number of <paramref name="unit"/> values.</summary>
    public static float[] Read(string option, string path, int unit, string unitName)
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length == 0 || bytes.Length % (unit * 4) != 0)
        {
            CommandLine.Fail($"{option} {path}: {bytes.Length} bytes is not a whole number of {unitName}");
        }
        return MemoryMarshal.Cast<byte, float>(bytes).ToArray();
    }
}
