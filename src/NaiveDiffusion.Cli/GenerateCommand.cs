using System.Globalization;
using NaiveDiffusion.Dml;
using NaiveDiffusion.Images;
using NaiveDiffusion.Models;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Sampling;
using NaiveDiffusion.Text;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Cli;

/// <summary>The whole pipeline: a prompt in, one PNG per seed out.</summary>
internal static class GenerateCommand
{
    public static readonly string[] ValueOptions = new[]
    {
        "--checkpoint", "--size", "--width", "--height", "--negative",
        "--steps", "--seed", "--count", "--guidance", "--sampler", "--schedule", "--out",
        "--denoiser-vram", "--weights", "--compute", "--image", "--strength", "--lora",
        "--lora-dir", "--clip-skip", "--step", "--step-seeds",
    }.Concat(SnippetOptions.ValueOptions)
     .Concat(CommandLine.ComponentOptions()).ToArray();

    public static readonly string[] FlagOptions = { "--whole-vae" };

    public static int Run(DmlDevice device, CommandLine line)
    {
        (IModelFamily family, int count,
            IReadOnlyList<(StepRun Step, GenerationOptions Options)> runs) = Read(line);
        string output = line.Value("--out") ?? "generated.png";

        WarnAboutSize(family, runs[0].Options);

        // Across steps the denoiser is the same model, so it stays where it
        // is between them — the seconds it takes to build are the whole cost
        // of a step otherwise. Kept only for the run.
        using DenoiserCache? cache = runs.Count > 1 ? new DenoiserCache { Enabled = true } : null;
        // And the negative prompt is the same text, encoded once.
        PromptCache? prompts = runs.Count > 1 ? new PromptCache() : null;
        GenerationPipeline pipeline = ModelFamilies.PipelineFor(family);
        foreach ((StepRun step, GenerationOptions run) in runs)
        {
            string label = step.InSequence ? $"step {step.Index + 1}/{step.Count} " : "";
            ImageResult[] images = pipeline.Generate(device, run, step.Seeds,
                new SynchronousProgress<Snapshot>(snapshot => Trace(label, snapshot)), cache: cache,
                prompts: prompts);

            for (int i = 0; i < images.Length; i++)
            {
                string name = !step.InSequence && count == 1
                    ? output
                    : Numbered(output, step.FilePrefix, step.Seeds[i]);
                ImageFile.SavePacked(images[i].Rgb24, images[i].Height, images[i].Width, name);
                Console.WriteLine($"Wrote {name}");
            }
        }
        return 0;
    }

    /// <summary>The command line as the runs it asks for, one per step, each
    /// put to the pipeline's own check here — before a weight is loaded, and
    /// every step before the first one runs, so a sequence does not find its
    /// fifth step refused an hour in. Opens no device.</summary>
    internal static (IModelFamily Family, int Count,
        IReadOnlyList<(StepRun Step, GenerationOptions Options)> Runs) Read(CommandLine line)
    {
        IModelFamily family = line.Family();
        (int width, int height) = line.Size(1024);
        var options = new GenerationOptions
        {
            Prompt = line.Argument ?? "an astronaut riding a horse on mars, highly detailed",
            Negative = line.Value("--negative") ?? "",
            Width = width,
            Height = height,
            Steps = line.Int("--steps", 20),
            Seed = line.Seed(),
            Guidance = line.Float("--guidance", family.DefaultGuidance),
            Sampler = line.Value("--sampler") switch
            {
                "euler" or null => SamplerKind.Euler,
                "euler-a" => SamplerKind.EulerAncestral,
                "dpmpp-2m" => SamplerKind.DpmPlusPlus2M,
                "dpmpp-2m-sde" => SamplerKind.DpmPlusPlus2MSde,
                string other => CommandLine.Fail<SamplerKind>(
                    $"--sampler {other}: one of euler, euler-a, dpmpp-2m, dpmpp-2m-sde"),
            },
            Schedule = line.Value("--schedule") switch
            {
                null => family.DefaultSchedule,
                "leading" => ScheduleKind.Leading,
                "linspace" => ScheduleKind.Linspace,
                "karras" => ScheduleKind.Karras,
                "exponential" => ScheduleKind.Exponential,
                "ays" => ScheduleKind.AlignYourSteps,
                string other => CommandLine.Fail<ScheduleKind>(
                    $"--schedule {other}: one of leading, linspace, karras, exponential, ays"),
            },
            CheckpointPath = line.Checkpoint(),
            Components = line.Components(family),
            ClipSkip = line.ClipSkip(family),
            TileVae = !line.Flag("--whole-vae"),
        };
        options = line.DenoiserMemory(options, family);
        options = Reference(line, options);
        options = line.Loras(options);

        int count = line.Int("--count", 1);
        if (count < 1)
        {
            CommandLine.Fail($"--count {count}: at least 1");
        }
        (string[] steps, bool continueSeeds) = Steps(line);
        RunPlan plan = Plan(line, options, steps, count, continueSeeds);

        GenerationPipeline pipeline = ModelFamilies.PipelineFor(family);
        var runs = new List<(StepRun, GenerationOptions)>();
        foreach (StepRun step in plan.Steps)
        {
            GenerationOptions run = options with
            {
                Prompt = step.Prompt, Negative = plan.Negative, Loras = step.Loras,
            };
            CommandLine.Prepare(pipeline, run);
            runs.Add((step, run));
        }
        return (family, count, runs);
    }

    /// <summary>The run as the pipeline will get it: snippets expanded —
    /// from --set and --snippet-dir — the sequence's text in place, and the
    /// '&lt;lora:name&gt;' tags the prompts carry resolved to files under
    /// --lora-dir (any number of them), or, given none, under the
    /// checkpoint's own folder and a loras/Lora folder beside it. A name
    /// nothing has is an error rather than a warning: the image would come
    /// out without the LoRA and nothing else would say so. A tag naming a
    /// file also given with --lora sets that one's weight.</summary>
    private static RunPlan Plan(CommandLine line, GenerationOptions options, string[] steps,
        int count, bool continueSeeds)
    {
        IReadOnlyList<string> folders = line.Values("--lora-dir");
        foreach (string folder in folders)
        {
            if (!Directory.Exists(folder))
            {
                CommandLine.Fail($"--lora-dir {folder}: no such folder");
            }
        }
        if (folders.Count == 0)
        {
            folders = LoraLibrary.DefaultFolders(options.CheckpointPath);
        }

        RunPlan? plan = RunPlanner.Plan(options.Prompt, options.Negative, steps,
            SnippetOptions.Resolver(line), options.Loras, folders, options.Seed, count,
            continueSeeds, out PlanProblem? problem);
        if (plan is null)
        {
            return CommandLine.Fail<RunPlan>(problem!.Kind switch
            {
                PlanProblemKind.StepWithoutSteps =>
                    $"the {problem.Where} has {PromptTemplate.Format(problem.Name)} but no --step was given",
                PlanProblemKind.MissingSnippet =>
                    $"the {problem.Where} refers to {PromptTemplate.Format(problem.Name)}, which nothing " +
                    $"defines; give it with --set {problem.Name}=text or a folder with --snippet-dir",
                PlanProblemKind.CyclicSnippet =>
                    $"{PromptTemplate.Format(problem.Name)} refers to itself, through however many others",
                PlanProblemKind.MissingLora =>
                    $"<lora:{problem.Name}>: no {problem.Name}{LoraLibrary.Extension} under " +
                    $"{string.Join(", ", folders)}; give its folder with --lora-dir",
                _ => $"<lora:{problem.Name}>: the weight is not a number",
            });
        }
        foreach (StepRun step in plan.Steps)
        {
            if (step.InSequence)
            {
                Console.WriteLine($"step {step.Index + 1}/{step.Count}: {step.Prompt}");
            }
            foreach (LoraSpec lora in step.PromptLoras)
            {
                if (CommandLine.NotLora(lora.Path) is string wrong)
                {
                    return CommandLine.Fail<RunPlan>($"from the prompt, {wrong}");
                }
                Console.WriteLine($"LoRA {lora.Path}: weight {lora.Weight:0.##} (from the prompt)");
            }
        }
        return plan;
    }

    /// <summary>--step text, any number of times: the sequence, one run of
    /// the prompt per step with the step's text where '{step}' is, or on
    /// the end when the prompt has no place for it. A blank step is
    /// refused: it would run the prompt as it stands, which is not a step
    /// of anything. --step-seeds says whether every step samples with the
    /// same seeds (the default) or counts on from the step before.</summary>
    private static (string[] Steps, bool ContinueSeeds) Steps(CommandLine line)
    {
        IReadOnlyList<string> steps = line.Values("--step");
        foreach (string step in steps)
        {
            if (string.IsNullOrWhiteSpace(step))
            {
                CommandLine.Fail("--step: a step needs some text");
            }
        }
        bool continueSeeds = line.Value("--step-seeds") switch
        {
            "same" or null => false,
            "continue" => true,
            string other => CommandLine.Fail<bool>($"--step-seeds {other}: same or continue"),
        };
        if (steps.Count == 0 && line.Value("--step-seeds") is not null)
        {
            CommandLine.Fail("--step-seeds needs --step");
        }
        return (steps.ToArray(), continueSeeds);
    }

    /// <summary>--image: a picture to start from instead of noise, scaled and
    /// centre-cropped to the size being generated; --strength says how much of
    /// it to redo, through the library's resampler.</summary>
    private static GenerationOptions Reference(CommandLine line, GenerationOptions options)
    {
        string? path = line.Value("--image");
        float strength = line.Float("--strength", options.Strength);
        if (path is null)
        {
            return line.Value("--strength") is null
                ? options
                : CommandLine.Fail<GenerationOptions>("--strength needs --image");
        }
        if (!File.Exists(path))
        {
            return CommandLine.Fail<GenerationOptions>($"--image {path}: no such file");
        }
        if (!(strength > 0.0f) || strength > 1.0f)
        {
            return CommandLine.Fail<GenerationOptions>(
                $"--strength {strength}: above 0 and at most 1");
        }
        ImageResult image = ImageFile.LoadRgb24(path);
        Console.WriteLine($"Reference {path}: {image.Width}x{image.Height}" +
                          (image.Width == options.Width && image.Height == options.Height
                              ? "" : $", resampled to {options.Width}x{options.Height}") +
                          $", strength {strength:0.##}");
        return options with
        {
            ReferenceImage = new ImageResult(options.Width, options.Height,
                Pixels.Resample(image.Rgb24, image.Width, image.Height,
                    options.Width, options.Height)),
            Strength = strength,
        };
    }

    /// <summary>Not an error — the size runs. It just spends 3.3 GiB more than
    /// it has to, which on a 16 GiB card can be the difference between running
    /// and hanging the device. Warned before the loading, not after.</summary>
    private static void WarnAboutSize(IModelFamily family, GenerationOptions options)
    {
        if (!family.SizeWastesMemory(options.Height, options.Width))
        {
            return;
        }
        string instead = family.NearbyEfficientHeight(options.Height, options.Width,
            step: 8, reach: 256, minimum: 1, maximum: int.MaxValue) is int alternative
            ? $" {options.Width}x{alternative} does not."
            : "";
        Console.WriteLine($"warning: at {options.Width}x{options.Height} DirectML keeps a " +
                          $"second copy of the UNet's widest weights, at least 3.3 GiB more " +
                          $"than the size needs.{instead}");
    }

    private static string Numbered(string output, string prefix, int seed) => Path.Combine(
        Path.GetDirectoryName(output) ?? "",
        $"{Path.GetFileNameWithoutExtension(output)}-{prefix}{seed}{Path.GetExtension(output)}");

    private static void Trace(string label, Snapshot snapshot)
    {
        string position = snapshot.TotalSteps > 0
            ? $" {snapshot.Step}/{snapshot.TotalSteps}" : "";
        string which = snapshot.TotalImages > 1
            ? $" image {snapshot.Image + 1}/{snapshot.TotalImages}" : "";
        Console.Write($"\r[{snapshot.Elapsed.TotalSeconds,5:0} s] " +
                      $"{label}{snapshot.Stage.Label()}{which}{position}    ");
        if (snapshot.TotalSteps == 0)
        {
            Console.WriteLine();
        }
    }
}

/// <summary>IProgress that reports on the calling thread, so console lines come
/// out in order.</summary>
internal sealed class SynchronousProgress<T> : IProgress<T>
{
    private readonly Action<T> _handler;

    public SynchronousProgress(Action<T> handler) => _handler = handler;

    public void Report(T value) => _handler(value);
}
