namespace NaiveDiffusion.Files;

/// <summary>Things kept as files by name — presets, snippets: one document
/// each in a folder, named after the thing, the document itself the source
/// of the name so a file renamed by hand still loads as what it says it is.
/// Two names differing only in case are one thing, as the file system
/// underneath would make them one file anyway; saving under a new spelling
/// replaces the old file rather than leaving it beside the new one.
///
/// The kinds differ only in how a document is read and written and where
/// its name sits, which is what the constructor takes. Files rather than a
/// settings store because a document can run to kilobytes and a collection
/// to dozens, and because a file can be copied to a friend or into a
/// backup.</summary>
public sealed class NamedFiles<T> where T : class
{
    public const string Extension = ".json";

    private readonly string _fallbackStem;
    private readonly Func<string, string, T?> _parse;
    private readonly Func<T, string> _serialize;
    private readonly Func<T, string> _nameOf;

    /// <param name="fallbackStem">The file name for a thing whose name
    /// leaves nothing a file name can hold.</param>
    /// <param name="parse">A document and the name to fall back on into the
    /// thing, or null when the document is not one.</param>
    public NamedFiles(string fallbackStem, Func<string, string, T?> parse,
        Func<T, string> serialize, Func<T, string> nameOf)
    {
        _fallbackStem = fallbackStem;
        _parse = parse;
        _serialize = serialize;
        _nameOf = nameOf;
    }

    /// <summary>The file a thing is written to, from its name — see
    /// <see cref="FileNames.ForName"/>.</summary>
    public string FileName(string name) => FileNames.ForName(name, _fallbackStem, Extension);

    /// <summary>Everything in the folder, by name. A file that is not one of
    /// these is passed over rather than stopping the rest from loading; a
    /// folder that is not there holds none; two files claiming one name keep
    /// the first read.</summary>
    public IReadOnlyList<T> LoadAll(string folder)
    {
        if (!Directory.Exists(folder))
        {
            return Array.Empty<T>();
        }
        var items = new List<T>();
        foreach (string path in Directory.EnumerateFiles(folder, "*" + Extension))
        {
            if (Read(path) is T item
                && !items.Any(kept => FileNames.NameComparer.Equals(_nameOf(kept), _nameOf(item))))
            {
                items.Add(item);
            }
        }
        return items.OrderBy(_nameOf, FileNames.NameComparer).ToList();
    }

    /// <summary>Write the thing, replacing any already under that name —
    /// including one filed under a different spelling of the same name,
    /// whose file would otherwise linger beside the new one. The path it
    /// went to. Throws as the file system does; the caller says so.
    ///
    /// Two names can map to one file name — "a:b" and "a_b" both to a_b —
    /// and the second saved must not take the first's file, so a file that
    /// says it holds another name is left alone and this one goes beside it
    /// under a number. The document is written whole before it replaces
    /// anything: a disk that fills halfway leaves the old one, not half of
    /// the new.</summary>
    public string Save(string folder, T item)
    {
        Directory.CreateDirectory(folder);
        string name = _nameOf(item);
        List<string> held = FilesNamed(folder, name);
        string path = Path.Combine(folder, FileName(name));
        if (HoldsAnother(path, name))
        {
            path = held.Count > 0 ? held[0] : FreeBeside(path, name);
        }

        string partial = path + ".tmp";
        File.WriteAllText(partial, _serialize(item));
        File.Move(partial, path, overwrite: true);
        foreach (string stale in held)
        {
            if (!string.Equals(stale, path, StringComparison.OrdinalIgnoreCase))
            {
                File.Delete(stale);
            }
        }
        return path;
    }

    /// <summary>True when the file is there and its document names something
    /// other than <paramref name="name"/>. One that cannot be read as a
    /// document holds nothing, and is written over.</summary>
    private bool HoldsAnother(string path, string name) =>
        File.Exists(path) && Read(path) is T other
        && !FileNames.NameComparer.Equals(_nameOf(other), name);

    private string FreeBeside(string path, string name)
    {
        string stem = Path.Combine(Path.GetDirectoryName(path)!, Path.GetFileNameWithoutExtension(path));
        for (int number = 2; ; number++)
        {
            string candidate = $"{stem}-{number}{Extension}";
            if (!HoldsAnother(candidate, name))
            {
                return candidate;
            }
        }
    }

    /// <summary>Remove the thing of that name. True when a file went.</summary>
    public bool Delete(string folder, string name)
    {
        bool deleted = false;
        foreach (string path in FilesNamed(folder, name))
        {
            File.Delete(path);
            deleted = true;
        }
        return deleted;
    }

    /// <summary>The thing in a file, or null when the file cannot be read
    /// right now or is not one. Unreadable is not an error here: the rest
    /// of a folder still loads.</summary>
    private T? Read(string path)
    {
        try
        {
            return _parse(File.ReadAllText(path), Path.GetFileNameWithoutExtension(path));
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <summary>The files in the folder holding a thing of that name, which
    /// is what their documents say and not what they are called: the file
    /// the name maps to may be another's, whose name maps there too.</summary>
    private List<string> FilesNamed(string folder, string name)
    {
        var files = new List<string>();
        if (!Directory.Exists(folder))
        {
            return files;
        }
        foreach (string path in Directory.EnumerateFiles(folder, "*" + Extension))
        {
            if (Read(path) is T item && FileNames.NameComparer.Equals(_nameOf(item), name))
            {
                files.Add(path);
            }
        }
        return files;
    }
}
