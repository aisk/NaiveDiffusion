using System.Text.RegularExpressions;
using NaiveDiffusion.Cli;
using NaiveDiffusion.Models;
using NaiveDiffusion.Models.Anima;
using NaiveDiffusion.Models.QwenImage;
using NaiveDiffusion.Models.Sdxl;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Sampling;
using NaiveDiffusion.Tests.Fixtures;

namespace NaiveDiffusion.Tests;

/// <summary>The command line's one rule: anything that cannot be run as
/// asked stops with a sentence, never with a default in its place. And
/// the usage text says what each command parses, so the two cannot
/// drift.</summary>
public class CommandLineTests
{
    private static readonly string[] Values = { "--steps", "--sampler", "--checkpoint" };
    private static readonly string[] Flags = { "--whole-vae" };

    private static CommandLine Parse(params string[] args) =>
        CommandLine.Parse(new[] { "generate" }.Concat(args).ToArray(), Values, Flags);

    [Test]
    public void AnUnknownOptionAMissingValueOrABadNumberIsRefused()
    {
        Assert.That(() => Parse("--sampelr", "euler"),
            Throws.TypeOf<UsageException>().With.Message.Contains("--sampelr"));
        Assert.That(() => Parse("--steps"),
            Throws.TypeOf<UsageException>().With.Message.Contains("needs a value"));
        Assert.That(() => Parse("--steps", "--whole-vae"),
            Throws.TypeOf<UsageException>().With.Message.Contains("needs a value"));
        Assert.That(() => Parse("--steps", "abc").Int("--steps", 20),
            Throws.TypeOf<UsageException>().With.Message.Contains("whole number"));
        Assert.That(() => Parse("a prompt", "another"),
            Throws.TypeOf<UsageException>().With.Message.Contains("unexpected argument"));
    }

    [Test]
    public void WhatWasGivenIsReadAsGiven()
    {
        CommandLine line = Parse("a prompt", "--steps", "8", "--whole-vae", "--sampler", "euler-a");
        Assert.That(line.Argument, Is.EqualTo("a prompt"));
        Assert.That(line.Int("--steps", 20), Is.EqualTo(8));
        Assert.That(line.Flag("--whole-vae"));
        Assert.That(line.Value("--sampler"), Is.EqualTo("euler-a"));
        Assert.That(line.Int("--seed", 3), Is.EqualTo(3), "an option left out is its fallback");
    }

    [Test]
    [SetCulture("de-DE")]
    public void NumbersAreInvariantWhateverTheConsole()
    {
        CommandLine line = CommandLine.Parse(new[] { "x", "--strength", "0.7", "--size", "512" },
            new[] { "--strength", "--size" }, Array.Empty<string>());
        Assert.That(line.Float("--strength", 1f), Is.EqualTo(0.7f));
        Assert.That(line.Size(1024), Is.EqualTo((512, 512)));
    }

    [Test]
    public void TheDenoiserMemoryOptionsTakeMebibytesOrAuto()
    {
        var blank = new GenerationOptions();
        IModelFamily anima = AnimaFamily.Instance;
        Assert.That(Parse("--denoiser-vram", "2000", "--weights", "int8").DenoiserMemory(blank, anima),
            Is.EqualTo(blank with { DenoiserResidentBytes = 2000UL << 20, DenoiserWeights = WeightStorage.Int8 }));
        Assert.That(Parse("--denoiser-vram", "auto").DenoiserMemory(blank, anima),
            Is.EqualTo(blank with { DenoiserAutoBudget = true }));
        Assert.That(() => Parse("--denoiser-vram", "auto").DenoiserMemory(blank, anima, allowAuto: false),
            Throws.TypeOf<UsageException>());
        Assert.That(() => Parse("--denoiser-vram", "lots").DenoiserMemory(blank, anima),
            Throws.TypeOf<UsageException>().With.Message.Contains("MiB"));

        static CommandLine Parse(params string[] args) => CommandLine.Parse(
            new[] { "x" }.Concat(args).ToArray(),
            new[] { "--denoiser-vram", "--weights", "--compute" }, Array.Empty<string>());
    }

    /// <summary>The weight storage and the compute precision are named, and
    /// a name that is not one is refused; left out, the precision is the
    /// family's own. A precision the family does not have is the
    /// pipeline's to refuse, as a usage error.</summary>
    [Test]
    public void TheWeightStorageAndTheComputePrecisionAreNamed()
    {
        var blank = new GenerationOptions();
        foreach (IModelFamily family in ModelFamilies.All)
        {
            GenerationOptions unset = Parse().DenoiserMemory(blank, family);
            Assert.That(unset.DenoiserCompute, Is.EqualTo(family.DefaultCompute), family.Name);
            Assert.That(unset.DenoiserWeights, Is.EqualTo(WeightStorage.Float16), family.Name);
        }
        IModelFamily anima = AnimaFamily.Instance;
        Assert.That(Parse("--compute", "fp32").DenoiserMemory(blank, anima).DenoiserCompute,
            Is.EqualTo(ComputePrecision.Float32));
        Assert.That(Parse("--compute", "fp16").DenoiserMemory(blank, anima).DenoiserCompute,
            Is.EqualTo(ComputePrecision.Float16));
        Assert.That(Parse("--weights", "fp16").DenoiserMemory(blank, anima).DenoiserWeights,
            Is.EqualTo(WeightStorage.Float16));
        Assert.That(() => Parse("--compute", "fp8").DenoiserMemory(blank, anima),
            Throws.TypeOf<UsageException>().With.Message.Contains("fp32 or fp16"));
        Assert.That(() => Parse("--weights", "int4").DenoiserMemory(blank, anima),
            Throws.TypeOf<UsageException>().With.Message.Contains("fp16 or int8"));

        using var folder = new TempFolder();
        string sdxl = new FakeSafetensors(folder).Write("sdxl", FakeSafetensors.SdxlCheckpoint());
        Command generate = Commands.Find("generate")!;
        Assert.That(() => Read("--compute", "fp32"),
            Throws.TypeOf<UsageException>().With.Message.Contains("fp32"));
        Assert.That(Read("--compute", "fp16", "--weights", "int8").Runs[0].Options.DenoiserWeights,
            Is.EqualTo(WeightStorage.Int8));

        (IModelFamily Family, int Count, IReadOnlyList<(StepRun Step, GenerationOptions Options)> Runs) Read(
            params string[] more) => GenerateCommand.Read(CommandLine.Parse(
                new[] { "generate", "1girl", "--checkpoint", sdxl }.Concat(more).ToArray(),
                Commands.ValueOptionsOf(generate), generate.FlagOptions));

        static CommandLine Parse(params string[] args) => CommandLine.Parse(
            new[] { "x" }.Concat(args).ToArray(),
            new[] { "--weights", "--compute" }, Array.Empty<string>());
    }

    [Test]
    public void TheFamilyComesFromTheCheckpointsHeader()
    {
        using var folder = new TempFolder();
        var files = new FakeSafetensors(folder);
        string sdxl = files.Write("sdxl", FakeSafetensors.SdxlCheckpoint());
        string anima = files.Write("anima", FakeSafetensors.AnimaDit());
        string vae = files.Write("wanvae", FakeSafetensors.WanVae());
        string[] options = new[] { "--checkpoint", "--clip-skip" }.Concat(CommandLine.ComponentOptions()).ToArray();

        Assert.That(Parse(sdxl, options).Family(), Is.SameAs(SdxlFamily.Instance));
        Assert.That(Parse(anima, options).Family(), Is.SameAs(AnimaFamily.Instance));
        Assert.That(() => Parse(vae, options).Family(),
            Throws.TypeOf<UsageException>().With.Message.Contains("only a VAE"));
        Assert.That(() => CommandLine.Parse(new[] { "x" }, options, Array.Empty<string>()).Family(),
            Throws.TypeOf<UsageException>().With.Message.Contains("--checkpoint"));

        // A checkpoint pruned of its towers still roundtrips.
        string noTowers = files.Write("novae", FakeSafetensors.SdxlCheckpoint()
            .Where(t => !t.Name.StartsWith("conditioner.", StringComparison.Ordinal)));
        Assert.That(() => Parse(noTowers, options).Family(),
            Throws.TypeOf<UsageException>().With.Message.Contains("missing"));
        Assert.That(Parse(noTowers, options).Family(CheckpointInspector.CheckpointParts.Vae),
            Is.SameAs(SdxlFamily.Instance));

        // The options a family does not have are refused, not ignored.
        CommandLine line = Parse(sdxl, options, "--text_encoder", vae);
        Assert.That(() => line.Components(line.Family()),
            Throws.TypeOf<UsageException>().With.Message.Contains("--text_encoder"));
        CommandLine skip = Parse(anima, options, "--clip-skip", "1");
        Assert.That(() => skip.ClipSkip(skip.Family()),
            Throws.TypeOf<UsageException>().With.Message.Contains("--clip-skip"));
        CommandLine missing = Parse(anima, options);
        Assert.That(() => missing.Components(missing.Family()),
            Throws.TypeOf<UsageException>().With.Message.Contains("is required"));

        static CommandLine Parse(string checkpoint, string[] options, params string[] more) =>
            CommandLine.Parse(new[] { "x", "--checkpoint", checkpoint }.Concat(more).ToArray(),
                options, Array.Empty<string>());
    }

    [Test]
    public void AnOptionGivenTwiceIsRefusedUnlessItRepeats()
    {
        Assert.That(() => Parse("--steps", "10", "--steps", "30"),
            Throws.TypeOf<UsageException>().With.Message.Contains("--steps"));
        CommandLine line = CommandLine.Parse(new[] { "x", "--step", "sitting", "--step", "standing" },
            new[] { "--step" }, Array.Empty<string>());
        Assert.That(line.Values("--step"), Is.EqualTo(new[] { "sitting", "standing" }));
    }

    /// <summary>What the pipeline's Prepare refuses is refused as a bad
    /// option is — a UsageException, so exit code 2 and no stack — and
    /// before a device is opened: Read takes none.</summary>
    [Test]
    public void WhatThePipelineRefusesIsAUsageErrorBeforeAnythingLoads()
    {
        using var folder = new TempFolder();
        var files = new FakeSafetensors(folder);
        string sdxl = files.Write("sdxl", FakeSafetensors.SdxlCheckpoint());
        string anima = files.Write("anima", FakeSafetensors.AnimaDit());
        string qwen = files.Write("qwen", FakeSafetensors.Qwen3TextEncoder());
        string vae = files.Write("wanvae", FakeSafetensors.WanVae());
        string[] animaParts = { "--text_encoder", qwen, "--vae", vae };

        Assert.That(Read(sdxl).Runs, Has.Count.EqualTo(1));
        Assert.That(Read(sdxl, "--step", "sitting", "--step", "standing").Runs, Has.Count.EqualTo(2));
        Assert.That(Read(anima, animaParts).Family, Is.SameAs(AnimaFamily.Instance));

        Assert.That(() => Read(sdxl, "--steps", "0"),
            Throws.TypeOf<UsageException>().With.Message.Contains("steps"));
        Assert.That(() => Read(sdxl, "--steps", "2000"),
            Throws.TypeOf<UsageException>().With.Message.Contains("steps"));
        Assert.That(() => Read(sdxl, "--size", "1001"),
            Throws.TypeOf<UsageException>().With.Message.Contains("multiple"));
        Assert.That(() => Read(sdxl, "--guidance", "nan"),
            Throws.TypeOf<UsageException>().With.Message.Contains("guidance"));
        Assert.That(() => Read(anima, animaParts.Concat(new[] { "--schedule", "ays" }).ToArray()),
            Throws.TypeOf<UsageException>());
        Assert.That(() => Read(anima, animaParts.Concat(new[] { "--size", "1000" }).ToArray()),
            Throws.TypeOf<UsageException>().With.Message.Contains("16"));

        // Nothing is run with a default in place of what was asked for.
        Assert.That(() => Read(sdxl, "--count", "0"),
            Throws.TypeOf<UsageException>().With.Message.Contains("--count"));
        Assert.That(() => Read(sdxl, "--seed", "-5"),
            Throws.TypeOf<UsageException>().With.Message.Contains("--seed"));

        // Qwen-Image is a family of its own on the command line, with its
        // own two files; Anima's are refused for it with the part's name.
        string qwenImage = files.Write("qwen21", FakeSafetensors.QwenImageDit());
        string qwen3Vl = files.Write("qwen3vl", FakeSafetensors.Qwen3VlTextEncoder());
        string qwenVae = files.Write("qwen21vae", FakeSafetensors.Wan22Vae());
        string[] qwenParts = { "--text_encoder", qwen3Vl, "--vae", qwenVae };
        Assert.That(Read(qwenImage, qwenParts).Family, Is.SameAs(QwenImageFamily.Instance));

        // Left out, the guidance is the family's own; given, it is as given.
        Assert.That(Read(sdxl).Runs[0].Options.Guidance, Is.EqualTo(5f));
        Assert.That(Read(anima, animaParts).Runs[0].Options.Guidance, Is.EqualTo(4f));
        Assert.That(Read(qwenImage, qwenParts).Runs[0].Options.Guidance, Is.EqualTo(1f));
        Assert.That(Read(qwenImage, qwenParts.Concat(new[] { "--guidance", "3" }).ToArray())
            .Runs[0].Options.Guidance, Is.EqualTo(3f));
        Assert.That(() => Read(qwenImage, "--text_encoder", qwen, "--vae", qwenVae),
            Throws.TypeOf<UsageException>().With.Message.Contains("--text_encoder"));
        Assert.That(() => Read(qwenImage, "--text_encoder", qwen3Vl),
            Throws.TypeOf<UsageException>().With.Message.Contains("--vae"));

        static (IModelFamily Family, int Count,
            IReadOnlyList<(StepRun Step, GenerationOptions Options)> Runs) Read(
                string checkpoint, params string[] more)
        {
            Command generate = Commands.Find("generate")!;
            return GenerateCommand.Read(CommandLine.Parse(
                new[] { "generate", "1girl", "--checkpoint", checkpoint }.Concat(more).ToArray(),
                Commands.ValueOptionsOf(generate), generate.FlagOptions));
        }
    }

    /// <summary>One forward pass reads only the parts it runs: without
    /// --context the text encoder and no VAE, with it neither — the rows
    /// come from the file, at the width the checkpoint's header says, and
    /// a family whose conditioning is not rows refuses the option. All
    /// before a device is opened.</summary>
    [Test]
    public void DenoiseAsksForTheFilesOfThePartsItRuns()
    {
        using var folder = new TempFolder();
        var files = new FakeSafetensors(folder);
        string sdxl = files.Write("sdxl", FakeSafetensors.SdxlCheckpoint());
        string qwenImage = files.Write("qwen21", FakeSafetensors.QwenImageDit());
        string qwen3Vl = files.Write("qwen3vl", FakeSafetensors.Qwen3VlTextEncoder());
        string rows = Path.Combine(folder.Path, "context.bin");
        File.WriteAllBytes(rows, new byte[2 * 4096 * 4]);
        string ragged = Path.Combine(folder.Path, "ragged.bin");
        File.WriteAllBytes(ragged, new byte[4096 * 4 + 4]);

        Assert.That(Read(sdxl).Context, Is.Null);
        Assert.That(() => Read(qwenImage),
            Throws.TypeOf<UsageException>().With.Message.Contains("--text_encoder"), "the text encoder runs");
        Assert.That(Read(qwenImage, "--text_encoder", qwen3Vl).Options.Components.Keys,
            Is.EqualTo(new[] { "text_encoder" }), "and the VAE never does");

        (IModelFamily family, GenerationOptions options, _, _, Conditioning? context) =
            Read(qwenImage, "--context", rows, "--compute", "fp32");
        Assert.That(family, Is.SameAs(QwenImageFamily.Instance));
        Assert.That(options.Components, Is.Empty);
        Assert.That(context!.Branches, Is.EqualTo(1));
        Assert.That(context.Specialization, Is.EqualTo("2"), "two rows of 4096");
        Assert.That(Read(qwenImage, "--context", rows).Context!.Specialization, Is.EqualTo("60"),
            "at the family's half precision the sequence is padded to a multiple of 64");

        Assert.That(() => Read(sdxl, "--context", rows),
            Throws.TypeOf<UsageException>().With.Message.Contains("--context"));
        Assert.That(() => Read(qwenImage, "--context", ragged),
            Throws.TypeOf<UsageException>().With.Message.Contains("4096"));
        Assert.That(() => Read(qwenImage, "--context", Path.Combine(folder.Path, "none.bin")),
            Throws.TypeOf<UsageException>().With.Message.Contains("no such file"));

        static (IModelFamily Family, GenerationOptions Options, ModelSampling Sampling, float Sigma,
            Conditioning? Context) Read(string checkpoint, params string[] more)
        {
            Command denoise = Commands.Find("denoise")!;
            return DenoiseCommand.Read(CommandLine.Parse(
                new[] { "denoise", "1girl", "--checkpoint", checkpoint, "--size", "32" }.Concat(more).ToArray(),
                Commands.ValueOptionsOf(denoise), denoise.FlagOptions));
        }
    }

    /// <summary>Every option a command parses is in its usage line, and
    /// every option its usage line shows is one it parses — so a new
    /// option cannot be added to one without the other.</summary>
    [Test]
    public void TheUsageTextMatchesWhatEachCommandParses()
    {
        foreach (Command command in Commands.All)
        {
            var shown = Regex.Matches(command.Usage, @"--[a-z0-9_-]+")
                .Select(match => match.Value).ToHashSet();
            var parsed = command.ValueOptions.Concat(command.FlagOptions).ToHashSet();
            Assert.That(shown, Is.EquivalentTo(parsed), command.Name);
        }
        Assert.That(Commands.Usage, Does.Contain("--adapter"));
        Assert.That(Commands.Find("generate")!.OpensDevice);
        Assert.That(Commands.Find("encode")!.OpensDevice, Is.False);
        Assert.That(Commands.Find("nothing"), Is.Null);
    }
}
