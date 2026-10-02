namespace NaiveDiffusion.Files;

/// <summary>A name the user typed, made into a file name: characters a file
/// name cannot hold become underscores, trailing dots and spaces go, an
/// overlong stem is cut, and a name that leaves nothing over is filed under
/// the fallback. The name itself is kept verbatim inside the document it
/// names, so nothing here has to be reversible.</summary>
public static class FileNames
{
    /// <summary>Two names are the same thing when they differ only in case:
    /// the file system underneath would give them one file anyway.</summary>
    public static readonly StringComparer NameComparer = StringComparer.OrdinalIgnoreCase;

    private const int StemLimit = 100;

    public static string ForName(string name, string fallback, string extension)
    {
        char[] invalid = Path.GetInvalidFileNameChars();
        var stem = new string(name.Trim().Select(c => invalid.Contains(c) ? '_' : c).ToArray())
            .TrimEnd('.', ' ');
        if (stem.Length > StemLimit)
        {
            stem = stem[..StemLimit].TrimEnd('.', ' ');
        }
        return (stem.Length == 0 ? fallback : stem) + extension;
    }
}
