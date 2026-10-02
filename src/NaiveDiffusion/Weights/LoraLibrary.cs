using NaiveDiffusion.Text;

namespace NaiveDiffusion.Weights;

/// <summary>Where a LoRA named in a prompt is looked for. A tag names a file
/// without its folder or extension — '&lt;lora:watercolor_style:0.8&gt;' —
/// because A1111 keeps every LoRA in one folder and finds it by name; this
/// does the same over whichever folders the caller has.</summary>
public static class LoraLibrary
{
    /// <summary>The file extension a LoRA is expected under.</summary>
    public const string Extension = ".safetensors";

    /// <summary>The folder names the two big front ends keep LoRAs under,
    /// next to their checkpoints folder: ComfyUI's models/loras and A1111's
    /// models/Lora. Windows does not care about the case, other systems
    /// do.</summary>
    private static readonly string[] SiblingNames = { "loras", "Lora", "lora" };

    /// <summary>The folders worth searching given where the checkpoint is: the
    /// checkpoint's own folder, and a LoRA folder beside it laid out the way
    /// ComfyUI or A1111 lays one out. Only folders that exist are
    /// returned.</summary>
    public static IReadOnlyList<string> DefaultFolders(string checkpointPath)
    {
        var folders = new List<string>();
        string? own = Path.GetDirectoryName(Path.GetFullPath(checkpointPath));
        if (own is null)
        {
            return folders;
        }
        if (Directory.Exists(own))
        {
            folders.Add(own);
        }
        if (Path.GetDirectoryName(own) is string parent)
        {
            foreach (string name in SiblingNames)
            {
                string sibling = Path.Combine(parent, name);
                if (Directory.Exists(sibling) && !folders.Contains(sibling, PathComparer))
                {
                    folders.Add(sibling);
                }
            }
        }
        return folders;
    }

    /// <summary>The file a tag names, or null when none of the folders has
    /// it. The name is tried as a path under each folder first — a tag may
    /// carry a subfolder, as A1111 writes for a LoRA filed under one — and
    /// then by file name anywhere in each folder's tree, the way A1111 finds
    /// one whose tag left the subfolder off. Case does not matter in either
    /// pass. The first folder to have it wins, so the caller orders them by
    /// preference.</summary>
    public static string? Find(string name, IEnumerable<string> folders)
    {
        string relative = name.Trim().Replace('/', Path.DirectorySeparatorChar);
        if (relative.Length == 0 || Path.IsPathRooted(relative) || relative.Contains(".."))
        {
            return null;
        }
        string fileName = Path.GetFileName(relative);
        foreach (string folder in folders.Distinct(PathComparer))
        {
            if (!Directory.Exists(folder))
            {
                continue;
            }
            string direct = Path.Combine(folder, relative + Extension);
            if (File.Exists(direct))
            {
                return direct;
            }
            if (File.Exists(Path.Combine(folder, relative)))
            {
                return Path.Combine(folder, relative);
            }
            try
            {
                string? found = Directory
                    .EnumerateFiles(folder, "*" + Extension, SearchOption.AllDirectories)
                    .FirstOrDefault(path => string.Equals(
                        Path.GetFileNameWithoutExtension(path), fileName,
                        StringComparison.OrdinalIgnoreCase));
                if (found is not null)
                {
                    return found;
                }
            }
            catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
            {
                // A folder that cannot be walked has nothing to offer; the
                // next one may.
            }
        }
        return null;
    }

    /// <summary>The tags in a prompt turned into files: the prompt without
    /// them, the LoRAs that were found, and the tags that were not. A tag
    /// whose weight is not a number is a miss too, reported by its text
    /// rather than applied at some weight it did not ask for.</summary>
    public static string Resolve(string prompt, IEnumerable<string> folders,
        out IReadOnlyList<LoraSpec> found, out IReadOnlyList<LoraTag> missing)
    {
        string stripped = LoraTags.Extract(prompt, out IReadOnlyList<LoraTag> tags);
        var loras = new List<LoraSpec>();
        var unresolved = new List<LoraTag>();
        string[] searched = folders.ToArray();
        foreach (LoraTag tag in tags)
        {
            string? path = float.IsFinite(tag.Weight) ? Find(tag.Name, searched) : null;
            if (path is null)
            {
                unresolved.Add(tag);
            }
            else
            {
                loras.Add(new LoraSpec(path, tag.Weight));
            }
        }
        found = loras;
        missing = unresolved;
        return stripped;
    }

    /// <summary>Paths compare the way the file system does here.</summary>
    private static StringComparer PathComparer =>
        OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;
}
