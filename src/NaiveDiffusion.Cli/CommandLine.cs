using System.Globalization;
using NaiveDiffusion.Models;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Sampling;
using NaiveDiffusion.Weights;
using static NaiveDiffusion.Models.CheckpointInspector;

namespace NaiveDiffusion.Cli;

/// <summary>Something on the command line that cannot be run as asked: the
/// message is the whole story, and the process ends with exit code 2 once
/// it has been printed — no stack, no default in its place.</summary>
internal sealed class UsageException : Exception
{
    public UsageException(string message) : base(message)
    {
    }
}

/// <summary>One command line, parsed and checked against what the command
/// actually accepts. A misspelt option is worth an exit code, not a run of the
/// default: the whole point of passing it was to get something other than the
/// default. That holds for the option's name as much as for its value, so an
/// unknown name, a missing value and a value that will not parse all stop the
/// run here rather than quietly turning into a default.</summary>
internal sealed class CommandLine
{
    private readonly Dictionary<string, string> _values = new(StringComparer.Ordinal);
    private readonly List<(string Name, string Value)> _all = new();
    private readonly HashSet<string> _flags = new(StringComparer.Ordinal);
    private IModelFamily? _family;

    /// <summary>The options that may be given any number of times, read with
    /// <see cref="Values"/>. Any other given twice is refused: the last one
    /// winning is a default in place of the first.</summary>
    private static readonly string[] Repeatable =
    {
        "--lora", "--lora-dir", "--set", "--snippet-dir", "--step",
    };

    /// <summary>The command's single positional argument — a prompt, an image
    /// path — or null when it was left out.</summary>
    public string? Argument { get; private set; }

    private CommandLine()
    {
    }

    /// <summary>Parse everything after the command name.
    /// <paramref name="valueOptions"/> each take a value;
    /// <paramref name="flagOptions"/> stand alone.</summary>
    public static CommandLine Parse(string[] args, string[] valueOptions, string[] flagOptions)
    {
        var line = new CommandLine();
        for (int i = 1; i < args.Length; i++)
        {
            string argument = args[i];
            if (!argument.StartsWith("--", StringComparison.Ordinal))
            {
                if (line.Argument is not null)
                {
                    Fail($"unexpected argument: {argument}");
                }
                line.Argument = argument;
                continue;
            }

            if (flagOptions.Contains(argument))
            {
                line._flags.Add(argument);
                continue;
            }
            if (!valueOptions.Contains(argument))
            {
                Fail($"unknown option: {argument}");
            }
            if (i + 1 >= args.Length || args[i + 1].StartsWith("--", StringComparison.Ordinal))
            {
                Fail($"{argument} needs a value");
            }
            if (line._values.ContainsKey(argument) && !Repeatable.Contains(argument))
            {
                Fail($"{argument} was given twice");
            }
            line._values[argument] = args[++i];
            line._all.Add((argument, args[i]));
        }
        return line;
    }

    public bool Flag(string name) => _flags.Contains(name);

    // Numbers are read the way the usage line writes them, whatever the
    // console's locale: "0.7" is seven tenths on a comma-decimal machine too.

    public string? Value(string name) => _values.GetValueOrDefault(name);

    /// <summary>Every value an option was given, in order — for the options
    /// that may be repeated. <see cref="Value"/> sees the last of them.</summary>
    public IReadOnlyList<string> Values(string name) =>
        _all.Where(pair => pair.Name == name).Select(pair => pair.Value).ToList();

    public int Int(string name, int fallback) =>
        _values.TryGetValue(name, out string? text)
            ? int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value)
                ? value
                : Fail<int>($"{name} {text}: expected a whole number")
            : fallback;

    public float Float(string name, float fallback) =>
        _values.TryGetValue(name, out string? text)
            ? float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float value)
                ? value
                : Fail<float>($"{name} {text}: expected a number")
            : fallback;

    /// <summary>--size sets both sides; --width and --height override it one
    /// at a time.</summary>
    public (int Width, int Height) Size(int fallback)
    {
        int square = Int("--size", fallback);
        return (Int("--width", square), Int("--height", square));
    }

    /// <summary>The family the --checkpoint belongs to, from its header: the
    /// file says which model it holds for the cost of a page fault, so a
    /// wrong file stops here with a sentence and an exit code rather than as
    /// an unhandled exception several gigabytes into the load. An SDXL
    /// checkpoint is checked against what the command actually reads out of
    /// it — <paramref name="required"/> — so roundtrip has no business
    /// rejecting a file over its text encoders; Anima's checkpoint is the
    /// transformer alone and has nothing to be missing.</summary>
    public IModelFamily Family(CheckpointParts required = CheckpointParts.All)
    {
        if (_family is not null)
        {
            return _family;
        }
        string? path = Value("--checkpoint");
        if (path is null || !File.Exists(path))
        {
            return Fail<IModelFamily>("--checkpoint <file.safetensors> is required: a single-file " +
                                      "SDXL checkpoint in the format ComfyUI and A1111 use, an " +
                                      "Anima transformer or a Qwen-Image 2.1 transformer");
        }
        if (!SafetensorsInspector.TryOpen(path, out SafetensorsFile? file))
        {
            return Fail<IModelFamily>($"{path}: not a readable safetensors file");
        }
        using (file)
        {
            IModelFamily? family = ModelFamilies.Identify(file, out CheckpointReport report, required);
            if (family is null)
            {
                return Fail<IModelFamily>(report.Kind == CheckpointKind.SdxlBase
                    && report.UnsupportedDataType is null
                    ? $"{path}: this checkpoint is missing " +
                      CheckpointInspector.NameParts(required & report.Missing)
                    : $"{path}: {CheckpointInspector.Explain(report)}");
            }
            return _family = family;
        }
    }

    /// <summary>The checkpoint every model-loading command needs, once
    /// <see cref="Family"/> has accepted it.</summary>
    public string Checkpoint(CheckpointParts required = CheckpointParts.All)
    {
        Family(required);
        return Value("--checkpoint")!;
    }

    /// <summary>The option each of a family's extra files is given by: its
    /// id with the dashes in front, so --vae is the VAE. The list comes from
    /// the families, not from here, so a family with more parts has more
    /// options without a line of parsing. Every family's parts are accepted
    /// by the parser — the family is only known once --checkpoint is read —
    /// and <see cref="Components"/> then turns away the ones the checkpoint's
    /// family does not have.</summary>
    public static string ComponentOption(ModelComponent component) => "--" + component.Id;

    public static string[] ComponentOptions() =>
        ModelFamilies.AllComponents.Select(ComponentOption).ToArray();

    /// <summary>The component options as the usage line writes them.</summary>
    public static string ComponentUsage() =>
        string.Join(" ", ModelFamilies.AllComponents.Select(part => $"[{ComponentOption(part)} file]"));

    /// <summary>The extra files given for the family's parts, each checked
    /// against the part: a file that is not there, or not that part, stops
    /// here with the family's own sentence. A required part left out stops
    /// here too, unless the command only runs some <paramref name="parts"/>
    /// of the family and the file belongs to another.</summary>
    public Dictionary<string, string> Components(IModelFamily family,
        PipelineParts parts = PipelineParts.All)
    {
        var components = new Dictionary<string, string>();
        foreach (ModelComponent component in ModelFamilies.AllComponents)
        {
            string option = ComponentOption(component);
            string? path = Value(option);
            ModelComponent? own = family.Components.FirstOrDefault(part => part.Id == component.Id);
            if (own is null)
            {
                if (path is not null)
                {
                    Fail($"{option}: {family.Name} takes no {component.Name} file");
                }
                continue;
            }
            if (path is null)
            {
                if (own.Required && (own.Part & parts) != 0)
                {
                    Fail($"{option} <file.safetensors> is required: the {own.Name}");
                }
                continue;
            }
            if (!File.Exists(path))
            {
                Fail($"{option} {path}: no such file");
            }
            ComponentVerdict verdict = family.InspectComponent(own, path);
            if (verdict != ComponentVerdict.Accepted)
            {
                Fail($"{option} {path}: {verdict.Explain(own)}");
            }
            components[own.Id] = path;
        }
        return components;
    }

    /// <summary>--clip-skip, A1111's count: 1 is the last layer, 2 the one
    /// before it and SDXL's own. A family without the choice refuses the
    /// option rather than reading past it.</summary>
    public int ClipSkip(IModelFamily family)
    {
        if (family.MaxClipSkip == 0)
        {
            return Value("--clip-skip") is null
                ? GenerationOptions.DefaultClipSkip
                : Fail<int>($"--clip-skip: {family.Name} reads its text encoder's last layer only");
        }
        int clipSkip = Int("--clip-skip", GenerationOptions.DefaultClipSkip);
        if (clipSkip < 1 || clipSkip > family.MaxClipSkip)
        {
            Fail($"--clip-skip {clipSkip}: between 1 and {family.MaxClipSkip}");
        }
        return clipSkip;
    }

    /// <summary>--unet-vram: how much video memory the diffusion model's
    /// weights may keep between steps, in MiB, or "auto" to take what the
    /// card has free when the model is built. Left out, everything stays
    /// resident. --int8 stores the large matrices block-quantized;
    /// --fp16-compute runs a transformer's blocks at half precision, for
    /// the timing (see
    /// <see cref="GenerationOptions.DenoiserHalfCompute"/>).</summary>
    public GenerationOptions DenoiserMemory(GenerationOptions options, bool allowAuto = true)
    {
        options = options with
        {
            DenoiserInt8Weights = Flag("--int8"),
            DenoiserHalfCompute = Flag("--fp16-compute"),
        };
        string? text = Value("--unet-vram");
        if (text is null)
        {
            return options;
        }
        if (text == "auto" && allowAuto)
        {
            return options with { DenoiserAutoBudget = true };
        }
        return ulong.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out ulong mib)
            ? options with { DenoiserResidentBytes = mib << 20 }
            : Fail<GenerationOptions>($"--unet-vram {text}: MiB" + (allowAuto ? " or auto" : ""));
    }

    /// <summary>--lora file[:weight], any number of times: LoRAs to fold in,
    /// in the order given, each at its weight, 1 when none is given. The
    /// weight follows the last colon, which on Windows cannot be the drive's:
    /// that one is second in the path.</summary>
    public GenerationOptions Loras(GenerationOptions options)
    {
        var loras = new List<LoraSpec>();
        foreach (string text in Values("--lora"))
        {
            string path = text;
            float weight = 1.0f;
            int colon = text.LastIndexOf(':');
            if (colon > 1)
            {
                if (!float.TryParse(text[(colon + 1)..], NumberStyles.Float,
                        CultureInfo.InvariantCulture, out weight))
                {
                    return Fail<GenerationOptions>(
                        $"--lora {text}: the part after the colon must be a weight");
                }
                path = text[..colon];
            }
            if (!File.Exists(path))
            {
                return Fail<GenerationOptions>($"--lora {path}: no such file");
            }
            if (NotLora(path) is string problem)
            {
                return Fail<GenerationOptions>($"--lora {problem}");
            }
            Console.WriteLine($"LoRA {path}: weight {weight:0.##}");
            loras.Add(new LoraSpec(path, weight));
        }
        return loras.Count == 0 ? options : options with { Loras = loras };
    }

    /// <summary>Why a file cannot be applied as a LoRA, or null when it can.
    /// The header says what a file is for the cost of a page fault, so a
    /// checkpoint given where a LoRA goes is turned away here rather than by
    /// the merge a minute in.</summary>
    public static string? NotLora(string path)
    {
        if (!TryInspect(path, out CheckpointReport report))
        {
            return $"{path}: not a readable safetensors file";
        }
        if (report.Kind == CheckpointKind.Lora)
        {
            return null;
        }
        return $"{path}: not a LoRA; " + (report.CanRun
            ? "this is a checkpoint, which goes after --checkpoint"
            : Explain(report));
    }

    /// <summary>--seed, which the generator takes the magnitude of: a
    /// negative one would sample what its positive twin does.</summary>
    public int Seed()
    {
        int seed = Int("--seed", 0);
        return seed >= 0 ? seed : Fail<int>($"--seed {seed}: 0 or above");
    }

    /// <summary>The pipeline's own check of a run, before anything is
    /// loaded. What it refuses — a size the family cannot do, a schedule or
    /// a LoRA it does not take, a file that is gone — was asked for on the
    /// command line, so it ends as any other bad option does: the sentence
    /// and exit code 2, not the runtime's report of an exception.</summary>
    public static ModelSampling Prepare(GenerationPipeline pipeline, GenerationOptions options,
        PipelineParts parts = PipelineParts.All)
    {
        try
        {
            return pipeline.Prepare(options, parts);
        }
        catch (Exception exception) when (exception is ArgumentException
            or FileNotFoundException or CheckpointNotSupportedException)
        {
            return Fail<ModelSampling>(exception is FileNotFoundException { FileName: string file }
                ? $"{exception.Message}: {file}"
                : exception.Message);
        }
    }

    public static void Fail(string message) => throw new UsageException(message);

    public static T Fail<T>(string message) => throw new UsageException(message);
}
