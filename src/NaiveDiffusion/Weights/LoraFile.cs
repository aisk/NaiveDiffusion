using System.Globalization;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using NaiveDiffusion.Tensors;

namespace NaiveDiffusion.Weights;

/// <summary>One LoRA to apply and how strongly: the file, and the multiplier
/// on its delta — 1 is what it was trained at, 0 is not having it.</summary>
public sealed record LoraSpec(string Path, float Weight)
{
    /// <summary>The LoRAs as one string that changes when any of them does,
    /// for a cache key: path, size and write time of each, with its weight, in
    /// order. A path can be overwritten in place with a different file, as a
    /// checkpoint can.</summary>
    public static string Stamp(IReadOnlyList<LoraSpec> loras)
    {
        if (loras.Count == 0)
        {
            return "";
        }
        // A LoRA at weight 0 changes nothing, so it is not part of what a
        // built model depends on either.
        return string.Join("\n", loras.Where(lora => lora.Weight != 0).Select(lora =>
        {
            var file = new FileInfo(lora.Path);
            return $"{System.IO.Path.GetFullPath(lora.Path)}|{file.Length}|" +
                   $"{file.LastWriteTimeUtc.Ticks}|{lora.Weight.ToString("R", CultureInfo.InvariantCulture)}";
        }));
    }

    /// <summary>One list out of several — the list a caller keeps and the
    /// tags a prompt carried, say — in which a later entry for a file already
    /// on the list replaces the earlier one instead of being folded in on top
    /// of it: the same LoRA twice would double its delta, and the tag a prompt
    /// was pasted with is the fresher word on the weight.</summary>
    public static IReadOnlyList<LoraSpec> Combine(params IEnumerable<LoraSpec>[] sources)
    {
        var combined = new List<LoraSpec>();
        foreach (LoraSpec lora in sources.SelectMany(source => source))
        {
            int index = combined.FindIndex(existing => SameFile(existing.Path, lora.Path));
            if (index >= 0)
            {
                combined[index] = lora;
            }
            else
            {
                combined.Add(lora);
            }
        }
        return combined;
    }

    private static bool SameFile(string first, string second) => string.Equals(
        System.IO.Path.GetFullPath(first), System.IO.Path.GetFullPath(second),
        OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
}

/// <summary>One model a LoRA can change and how the files name it: kohya's
/// prefix (<c>lora_unet_…</c>) and PEFT's (<c>unet.…</c>), and the name
/// the merge asks for it by.</summary>
/// <param name="Name">The tower's name as the merge is asked for it.</param>
/// <param name="Description">How an error names it: "UNet", "CLIP-L".</param>
/// <param name="OtherPeftPrefixes">The other prefixes a dotted name may sit
/// under, for a model different trainers address differently; an empty one
/// for a model whose layers are also named with no prefix at all.</param>
public sealed record LoraTower(string Name, string Description, string KohyaPrefix,
    string PeftPrefix, params string[] OtherPeftPrefixes)
{
    internal IEnumerable<string> PeftPrefixes => OtherPeftPrefixes.Prepend(PeftPrefix);
}

/// <summary>A name a LoRA has for part of a weight: <paramref name="Count"/>
/// of its rows from <paramref name="First"/> on, for a model that stores as
/// one matrix what the LoRA addresses as several.</summary>
public readonly record struct LoraRows(string Name, int First, int Count);

/// <summary>How one family's LoRAs are laid out: which towers they may
/// change, and which prefixes mark a file made for another architecture —
/// refused on sight, with the reason, because a file that half fits would
/// otherwise fold its fitting half in without complaint.</summary>
public sealed class LoraLayout
{
    private readonly Dictionary<string, LoraTower> _byKohya;
    private readonly Dictionary<string, LoraTower> _byPeft;
    private readonly Dictionary<string, string> _byName;

    /// <param name="architecture">The family's name for the error that
    /// says a file was trained for another one.</param>
    /// <param name="refusedKohyaPrefixes">Kohya prefixes that mark a file
    /// for another architecture, each with the sentence that says so.</param>
    public LoraLayout(string architecture, IReadOnlyList<LoraTower> towers,
        IReadOnlyDictionary<string, string> refusedKohyaPrefixes)
    {
        Architecture = architecture;
        Towers = towers;
        RefusedKohyaPrefixes = refusedKohyaPrefixes;
        _byKohya = towers.ToDictionary(tower => tower.KohyaPrefix, StringComparer.Ordinal);
        _byPeft = towers.SelectMany(tower => tower.PeftPrefixes.Select(prefix => (prefix, tower)))
            .ToDictionary(pair => pair.prefix, pair => pair.tower, StringComparer.Ordinal);
        _byName = towers.ToDictionary(tower => tower.Name, tower => tower.Description,
            StringComparer.Ordinal);
        string kohya = string.Join("|", towers.Select(tower => Regex.Escape(tower.KohyaPrefix))
            .Concat(refusedKohyaPrefixes.Keys.Select(Regex.Escape)));
        string peft = string.Join("|", _byPeft.Keys.Where(prefix => prefix.Length > 0).Select(Regex.Escape));
        // A tower named with no prefix takes whatever dotted name sits under
        // none of the others.
        string prefix = _byPeft.ContainsKey("") ? $@"(?:({peft})\.)?" : $@"({peft})\.";
        KohyaPattern = new Regex(
            $@"^lora_({kohya})_(.+)\.(lora_down\.weight|lora_up\.weight|alpha)$", RegexOptions.Compiled);
        // The dotted names end the way PEFT ends them, the way kohya does —
        // which is how ComfyUI saves its own — or with PEFT's adapter name
        // left in, as DiffSynth leaves it.
        PeftPattern = new Regex(
            $@"^{prefix}(.+)\.(lora_A\.weight|lora_B\.weight|lora_A\.default\.weight|" +
            @"lora_B\.default\.weight|lora_down\.weight|lora_up\.weight|alpha)$",
            RegexOptions.Compiled);
    }

    public string Architecture { get; }

    public IReadOnlyList<LoraTower> Towers { get; }

    public IReadOnlyDictionary<string, string> RefusedKohyaPrefixes { get; }

    internal Regex KohyaPattern { get; }

    internal Regex PeftPattern { get; }

    internal LoraTower? ByKohya(string prefix) => _byKohya.GetValueOrDefault(prefix);

    internal LoraTower? ByPeft(string prefix) => _byPeft.GetValueOrDefault(prefix);

    /// <summary>How an error names a tower.</summary>
    public string Describe(string tower) => _byName.GetValueOrDefault(tower, tower);
}

/// <summary>A LoRA's tensors, grouped into the low-rank pairs they came as.
///
/// A LoRA stores, for each linear or convolution layer it touches, a pair of
/// small matrices whose product is the change to that layer's weight:
/// W' = W + weight × (alpha / rank) × up · down. This reads the pairs under
/// either naming the ecosystem uses — kohya's, which is what Civitai and
/// ComfyUI trade in, and PEFT's, which diffusers writes — and files them by the
/// model they belong to and the layer within it, with the layer's name written
/// the way kohya writes it: the module path with dots replaced by
/// underscores, under the towers a <see cref="LoraLayout"/> names.
/// <see cref="LoraMerge"/> matches those against a model's tensors and folds
/// the products in on the CPU, before the weights are uploaded or quantized,
/// so the graphs never know a LoRA was involved.
///
/// The tensors are windows onto the mapped file, as a checkpoint's are, so
/// this is disposed only once the merge has copied out what it needs.</summary>
public sealed class LoraFile : IDisposable
{
    /// <summary>One layer's pair. <paramref name="Alpha"/> is the scale the
    /// trainer chose, applied as alpha / rank; a file that carries none was
    /// trained with alpha equal to the rank, which makes the factor 1.</summary>
    public sealed record Module(HostTensor Down, HostTensor Up, float? Alpha);

    private readonly SafetensorsFile _file;
    private readonly LoraLayout _layout;
    private readonly Dictionary<string, Dictionary<string, Module>> _towers = new();

    public string Path { get; }

    /// <summary>The file's name without its extension, which is what the
    /// A1111 convention calls a LoRA in the prompt.</summary>
    public string Name => System.IO.Path.GetFileNameWithoutExtension(Path);

    public LoraFile(string path, LoraLayout layout)
    {
        Path = path;
        _layout = layout;
        _file = new SafetensorsFile(path);
        try
        {
            Index();
        }
        catch
        {
            _file.Dispose();
            throw;
        }
    }

    /// <summary>The layers this file changes in one model, by kohya-style
    /// layer name. Empty for a model the file leaves alone.</summary>
    public IReadOnlyDictionary<string, Module> Modules(string tower) =>
        _towers.TryGetValue(tower, out Dictionary<string, Module>? modules)
            ? modules
            : new Dictionary<string, Module>();

    private void Index()
    {
        var downs = new Dictionary<(string Tower, string Module), HostTensor>();
        var ups = new Dictionary<(string Tower, string Module), HostTensor>();
        var alphas = new Dictionary<(string Tower, string Module), float>();
        int unreadable = 0;

        foreach (string key in _file.Keys)
        {
            if (key.EndsWith(".dora_scale", StringComparison.Ordinal))
            {
                throw new InvalidDataException(
                    $"{Name} is a DoRA, which changes the weights' directions rather " +
                    "than adding to them; this library applies only LoRA");
            }

            (string Tower, string Module) id;
            string part;
            Match match = _layout.KohyaPattern.Match(key);
            if (match.Success)
            {
                string prefix = match.Groups[1].Value;
                if (_layout.RefusedKohyaPrefixes.TryGetValue(prefix, out string? reason))
                {
                    throw new InvalidDataException($"{Name} {reason}");
                }
                id = (_layout.ByKohya(prefix)!.Name, match.Groups[2].Value);
                part = match.Groups[3].Value switch
                {
                    "lora_down.weight" => "down",
                    "lora_up.weight" => "up",
                    _ => "alpha",
                };
            }
            else if ((match = _layout.PeftPattern.Match(key)).Success)
            {
                id = (_layout.ByPeft(match.Groups[1].Value)!.Name,
                    match.Groups[2].Value.Replace('.', '_'));
                part = match.Groups[3].Value switch
                {
                    "lora_A.weight" or "lora_A.default.weight" or "lora_down.weight" => "down",
                    "lora_B.weight" or "lora_B.default.weight" or "lora_up.weight" => "up",
                    _ => "alpha",
                };
            }
            else
            {
                unreadable++;
                continue;
            }

            switch (part)
            {
                case "down":
                    downs[id] = _file.Read(key);
                    break;
                case "up":
                    ups[id] = _file.Read(key);
                    break;
                default:
                    alphas[id] = _file.Read(key).ToFloats()[0];
                    break;
            }
        }

        foreach ((string tower, string module) in ups.Keys)
        {
            if (!downs.ContainsKey((tower, module)))
            {
                throw new InvalidDataException(
                    $"{Name}: {module} has an up matrix but no down matrix");
            }
        }
        foreach (((string tower, string module), HostTensor down) in downs)
        {
            if (!ups.TryGetValue((tower, module), out HostTensor? up))
            {
                throw new InvalidDataException(
                    $"{Name}: {module} has a down matrix but no up matrix");
            }
            if (!_towers.TryGetValue(tower, out Dictionary<string, Module>? modules))
            {
                _towers[tower] = modules = new Dictionary<string, Module>();
            }
            modules[module] = new Module(down, up,
                alphas.TryGetValue((tower, module), out float alpha) ? alpha : null);
        }

        if (_towers.Count == 0)
        {
            // A LoRA for another architecture names its layers in a way this
            // layout does not read either; the file may say which one it is.
            string trainedFor = _file.Metadata.TryGetValue("modelspec.architecture",
                out string? architecture) && architecture.Length > 0
                ? $"; the file says it is a {architecture}, not for {_layout.Architecture}"
                : "";
            throw new InvalidDataException(unreadable > 0
                ? $"{Name} holds no LoRA pairs this library reads for {_layout.Architecture}; " +
                  "LoHa, LoKr and full-weight diffs are not supported" + trainedFor
                : $"{Name} holds no tensors");
        }
    }

    public void Dispose() => _file.Dispose();
}

/// <summary>Folds LoRAs into a model's weights on the CPU.</summary>
public static class LoraMerge
{
    /// <summary>Replace every weight in <paramref name="tensors"/> that any of
    /// the <paramref name="loras"/> touches with the weight plus the LoRA's
    /// scaled delta. A layer is found by its kohya name — the key without its
    /// ".weight", dots to underscores — and by whatever other names
    /// <paramref name="aliases"/> gives for the key, for a model whose tensors
    /// are keyed one way while the files name them another.
    ///
    /// Only the weights change: a LoRA carries no biases. The merged tensor is
    /// a fresh copy at the weight's own width, or half precision for a weight
    /// stored as bfloat16, which nothing downstream reads; the rest of the
    /// dictionary stays as it was, windows onto the checkpoint included.
    ///
    /// A file none of whose layers for this model are found is refused: it was
    /// trained against a different architecture, and applying nothing while
    /// claiming to have applied it is the one outcome worse than an error.
    /// A file some of whose layers are found is applied as far as it goes,
    /// as the reference implementations do. <paramref name="parts"/> gives
    /// the names a LoRA has for some of a weight's rows.</summary>
    public static void Apply(Dictionary<string, HostTensor> tensors, LoraLayout layout,
        string tower, IReadOnlyList<LoraSpec> loras, Func<string, IEnumerable<string>>? aliases = null,
        Func<string, IEnumerable<LoraRows>>? parts = null)
    {
        using LoraPatchSet patches = LoraPatchSet.Prepare(tensors, layout, tower, loras, aliases, parts);
        foreach (string key in patches.Keys.ToArray())
        {
            tensors[key] = patches.Apply(key, tensors[key]);
        }
    }

    internal static string ModuleName(string key) =>
        key[..^".weight".Length].Replace('.', '_');

    /// <summary>Whether a pair is the shape of the rows it is to change. A
    /// linear weight is [out, in]; a convolution's is [out, in, kh, kw], and
    /// its down matrix carries the same trailing axes. Either way a row is
    /// everything past the first axis, and the up matrix is [out, rank] with,
    /// for a convolution, two trailing axes of one.</summary>
    internal static void Check(HostTensor weight, LoraFile.Module module, int rowCount, string key)
    {
        int rank = module.Down.Shape[0];
        long columns = weight.ElementCount / weight.Shape[0];
        if (module.Down.ElementCount != rank * columns
            || module.Up.Shape[0] != rowCount || module.Up.ElementCount != (long)rowCount * rank)
        {
            throw new InvalidDataException(
                $"the LoRA's {key} is [{string.Join(", ", module.Up.Shape)}] x " +
                $"[{string.Join(", ", module.Down.Shape)}] but the model's is " +
                $"[{string.Join(", ", weight.Shape)}]; the LoRA was trained against a " +
                "different model");
        }
    }

    /// <summary>The weight plus every delta, row by row: each output row is
    /// widened, takes rank scaled rows of each down matrix, and is narrowed
    /// back. Nothing the size of the weight exists in float32 at any point,
    /// and the inner loop is one vector multiply-add per (row, rank). A delta
    /// for part of the weight starts at its first row.</summary>
    internal static HostTensor Merge(HostTensor weight,
        List<(LoraFile.Module Module, float Scale, int FirstRow)> patches)
    {
        int rows = weight.Shape[0];
        int columns = checked((int)(weight.ElementCount / rows));

        var factors = new (float[] Down, float[] Up, int Rank, float Scale, int First, int Count)[patches.Count];
        for (int i = 0; i < patches.Count; i++)
        {
            (LoraFile.Module module, float scale, int first) = patches[i];
            factors[i] = (module.Down.ToFloats(), module.Up.ToFloats(), module.Down.Shape[0], scale,
                first, module.Up.Shape[0]);
        }

        HostDataType type = weight.DataType == HostDataType.Float32
            ? HostDataType.Float32
            : HostDataType.Float16;
        int itemSize = HostTensor.BytesPerElement(type);
        var data = new byte[checked(rows * columns * itemSize)];

        Parallel.For(0, rows, () => new float[columns], (row, _, scratch) =>
        {
            weight.WidenRow(row, columns, scratch);
            foreach ((float[] down, float[] up, int rank, float scale, int first, int count) in factors)
            {
                if (row < first || row >= first + count)
                {
                    continue;
                }
                for (int r = 0; r < rank; r++)
                {
                    float factor = scale * up[(row - first) * rank + r];
                    if (factor == 0)
                    {
                        continue;
                    }
                    TensorPrimitives.MultiplyAdd(down.AsSpan(r * columns, columns), factor,
                        scratch, scratch);
                }
            }
            Span<byte> target = data.AsSpan(row * columns * itemSize, columns * itemSize);
            if (type == HostDataType.Float32)
            {
                MemoryMarshal.AsBytes<float>(scratch).CopyTo(target);
            }
            else
            {
                TensorPrimitives.ConvertToHalf(scratch, MemoryMarshal.Cast<byte, Half>(target));
            }
            return scratch;
        }, _ => { });

        return new HostTensor(type, weight.Shape, data);
    }
}

/// <summary>The deltas a set of LoRAs has for one model's weights, matched by
/// name and held ready, with the files open behind them: <see cref="Apply"/>
/// folds them into one weight at a time, which is what lets the UNet be
/// merged graph by graph instead of all at once.</summary>
public sealed class LoraPatchSet : IDisposable
{
    private readonly Dictionary<string, List<(LoraFile.Module Module, float Scale, int FirstRow)>> _patches;
    private readonly List<LoraFile> _files;

    private LoraPatchSet(Dictionary<string, List<(LoraFile.Module, float, int)>> patches, List<LoraFile> files)
    {
        _patches = patches;
        _files = files;
    }

    /// <summary>Open the files and match their layers against
    /// <paramref name="tensors"/>, as <see cref="LoraMerge.Apply"/> describes:
    /// by kohya name, by whatever other name <paramref name="aliases"/>
    /// gives a key, and by the names <paramref name="parts"/> gives some of
    /// its rows. A file none of whose layers are found is refused here,
    /// before anything has been merged, and so is a pair that is not the
    /// shape of the weight it names.</summary>
    public static LoraPatchSet Prepare(IReadOnlyDictionary<string, HostTensor> tensors,
        LoraLayout layout, string tower,
        IReadOnlyList<LoraSpec> loras, Func<string, IEnumerable<string>>? aliases = null,
        Func<string, IEnumerable<LoraRows>>? parts = null)
    {
        // Every name a weight answers to, so a file is matched with one lookup
        // per layer rather than a scan of the model per layer.
        var byModule = new Dictionary<string, (string Key, int First, int Count)>(StringComparer.Ordinal);
        foreach ((string key, HostTensor tensor) in tensors)
        {
            if (!key.EndsWith(".weight", StringComparison.Ordinal) || tensor.Shape.Length == 0)
            {
                continue;
            }
            int rows = tensor.Shape[0];
            byModule[LoraMerge.ModuleName(key)] = (key, 0, rows);
            foreach (string alias in aliases?.Invoke(key) ?? Array.Empty<string>())
            {
                byModule[LoraMerge.ModuleName(alias)] = (key, 0, rows);
            }
            foreach (LoraRows part in parts?.Invoke(key) ?? Array.Empty<LoraRows>())
            {
                byModule[LoraMerge.ModuleName(part.Name)] = (key, part.First, part.Count);
            }
        }

        // Gathered per weight before anything is computed: a weight two files
        // both change is widened once, takes both deltas, and is narrowed once.
        var patches = new Dictionary<string, List<(LoraFile.Module, float, int)>>();
        var files = new List<LoraFile>(loras.Count);
        try
        {
            foreach (LoraSpec lora in loras)
            {
                if (lora.Weight == 0)
                {
                    // Nothing to add, so nothing to read or to copy.
                    continue;
                }
                var file = new LoraFile(lora.Path, layout);
                files.Add(file);
                IReadOnlyDictionary<string, LoraFile.Module> modules = file.Modules(tower);
                int found = 0;
                foreach ((string module, LoraFile.Module pair) in modules)
                {
                    if (!byModule.TryGetValue(module, out (string Key, int First, int Count) target))
                    {
                        continue;
                    }
                    found++;
                    LoraMerge.Check(tensors[target.Key], pair, target.Count, target.Key);
                    int rank = pair.Down.Shape[0];
                    float scale = lora.Weight * (pair.Alpha ?? rank) / rank;
                    if (!patches.TryGetValue(target.Key, out List<(LoraFile.Module, float, int)>? list))
                    {
                        patches[target.Key] = list = new List<(LoraFile.Module, float, int)>();
                    }
                    list.Add((pair, scale, target.First));
                }
                if (modules.Count > 0 && found == 0)
                {
                    throw new InvalidDataException(
                        $"{file.Name}: none of its {modules.Count} {layout.Describe(tower)} layers " +
                        "exist in this model; it was trained for a different architecture " +
                        $"than {layout.Architecture}");
                }
            }
        }
        catch
        {
            foreach (LoraFile file in files)
            {
                file.Dispose();
            }
            throw;
        }
        return new LoraPatchSet(patches, files);
    }

    /// <summary>The keys that have a delta.</summary>
    public IEnumerable<string> Keys => _patches.Keys;

    public bool Touches(string key) => _patches.ContainsKey(key);

    /// <summary>A fresh copy of <paramref name="weight"/> with the deltas for
    /// <paramref name="key"/> added, or the weight itself when there are none.</summary>
    public HostTensor Apply(string key, HostTensor weight) =>
        _patches.TryGetValue(key, out List<(LoraFile.Module Module, float Scale, int FirstRow)>? list)
            ? LoraMerge.Merge(weight, list)
            : weight;

    public void Dispose()
    {
        foreach (LoraFile file in _files)
        {
            file.Dispose();
        }
        _files.Clear();
        _patches.Clear();
    }
}
