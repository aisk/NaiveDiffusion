using System.Text.Json;
using System.Text.Json.Serialization;
using NaiveDiffusion.Files;
using NaiveDiffusion.Sampling;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Pipeline;

/// <summary>A recipe kept under a name: what an image is asked for, without
/// the machine it is asked of. The prompts, the model with its extra files
/// and the LoRAs, the size, the sampler settings, the clip skip, the seed and
/// the reference strength travel; the
/// device, the memory switches and the batch count stay where they are, since
/// swapping recipes should not swap graphics cards.
///
/// The model and the LoRAs are kept as paths, which is what this library can
/// reopen; a preset moved to another machine is matched up by file name
/// instead, which is the caller's job. A preset that lacks a field takes the
/// default, so a file written by an older build still loads.
///
/// Settable rather than init-only: the generated deserializer builds an
/// init-only type the way it builds one with a constructor, handing every
/// property a value whether the document had one or not — so a missing string
/// would land as null instead of its initializer. <see cref="PresetFile.Parse"/>
/// still guards the strings, against a document that says null outright.</summary>
public sealed record Preset
{
    public string Name { get; set; } = "";
    public string Prompt { get; set; } = "";
    public string Negative { get; set; } = "";

    /// <summary>The checkpoint's path; empty leaves the model alone.</summary>
    public string Checkpoint { get; set; } = "";

    /// <summary>The files given for the model's parts, by part id — a VAE
    /// under "vae". A part not here is the checkpoint's own.</summary>
    public IReadOnlyDictionary<string, string> Components { get; set; } =
        new Dictionary<string, string>();

    public IReadOnlyList<LoraSpec> Loras { get; set; } = Array.Empty<LoraSpec>();
    public int Width { get; set; } = 1024;
    public int Height { get; set; } = 1024;
    public int Steps { get; set; } = 20;
    public float Guidance { get; set; } = 5.0f;
    public SamplerKind Sampler { get; set; } = SamplerKind.Euler;
    public ScheduleKind Schedule { get; set; } = ScheduleKind.Leading;
    public int Seed { get; set; }
    public float Strength { get; set; } = 0.65f;
    public int ClipSkip { get; set; } = GenerationOptions.DefaultClipSkip;

    /// <summary>The sequence: one run of the prompt per entry, with the
    /// entry put where '{step}' is — see <see cref="PromptTemplate.WithStep"/>.
    /// Empty is no sequence. The snippets the prompt refers to are not in
    /// here: they are the library's, and a preset names them as the prompt
    /// does.</summary>
    public IReadOnlyList<string> Sequence { get; set; } = Array.Empty<string>();

    public SequenceSeeds SequenceSeeds { get; set; } = SequenceSeeds.Same;
}

/// <summary>Which seeds each step of a sequence samples with.</summary>
public enum SequenceSeeds
{
    /// <summary>Every step the same seeds, so what changes between two
    /// steps is what the step text changed — as near as a diffusion model
    /// keeps a composition when the prompt moves, which is partly.</summary>
    Same,

    /// <summary>Counting on from the last image of the step before, so no
    /// two images of the sequence share a seed.</summary>
    Continue,
}

/// <summary>Presets as files, one JSON document each — see
/// <see cref="NamedFiles{T}"/> for the folder's rules.</summary>
public static class PresetFile
{
    public const string Extension = NamedFiles<Preset>.Extension;

    /// <summary>Two names are the same preset when they differ only in case.</summary>
    public static StringComparer NameComparer => FileNames.NameComparer;

    private static readonly NamedFiles<Preset> Files = new("preset", Parse, Serialize, preset => preset.Name);

    public static string FileName(string name) => Files.FileName(name);

    public static string Serialize(Preset preset) =>
        JsonSerializer.Serialize(preset, PresetJsonContext.Default.Preset);

    /// <summary>The preset in a document, or null when it is not one. An
    /// unnamed document takes the name it is given, which the loader makes
    /// the file's own.</summary>
    public static Preset? Parse(string json, string fallbackName = "")
    {
        try
        {
            Preset? preset = JsonSerializer.Deserialize(json, PresetJsonContext.Default.Preset);
            if (preset is null)
            {
                return null;
            }
            string name = (preset.Name ?? "").Trim();
            return preset with
            {
                Name = name.Length == 0 ? fallbackName : name,
                Prompt = preset.Prompt ?? "",
                Negative = preset.Negative ?? "",
                Checkpoint = preset.Checkpoint ?? "",
                // A part without a file is the checkpoint's own.
                Components = (preset.Components ?? new Dictionary<string, string>())
                    .Where(pair => !string.IsNullOrWhiteSpace(pair.Key)
                                   && !string.IsNullOrWhiteSpace(pair.Value))
                    .ToDictionary(pair => pair.Key, pair => pair.Value),
                // An entry without a file is not a LoRA to look for.
                Loras = (preset.Loras ?? Array.Empty<LoraSpec>())
                    .Where(lora => !string.IsNullOrWhiteSpace(lora?.Path))
                    .ToArray(),
                // A blank step would run the prompt as it stands, which is
                // never what a sequence was saved for.
                Sequence = (preset.Sequence ?? Array.Empty<string>())
                    .Where(step => !string.IsNullOrWhiteSpace(step))
                    .ToArray(),
            };
        }
        catch (JsonException)
        {
            return null;
        }
    }

    public static IReadOnlyList<Preset> LoadAll(string folder) => Files.LoadAll(folder);

    public static string Save(string folder, Preset preset) => Files.Save(folder, preset);

    public static bool Delete(string folder, string name) => Files.Delete(folder, name);
}

/// <summary>Generated serializer: a trimmed publish leaves the
/// reflection-based one nothing to reflect on. Enums go by name, so a
/// document reads as what it means and survives the enum being reordered.</summary>
[JsonSourceGenerationOptions(WriteIndented = true, UseStringEnumConverter = true)]
[JsonSerializable(typeof(Preset))]
internal sealed partial class PresetJsonContext : JsonSerializerContext
{
}
