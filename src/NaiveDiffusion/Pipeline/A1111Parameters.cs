using System.Globalization;
using System.Text;
using NaiveDiffusion.Images;
using NaiveDiffusion.Sampling;
using NaiveDiffusion.Text;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Pipeline;

/// <summary>The settings behind an image, laid out the way A1111 writes them and
/// every metadata reader in the ecosystem parses them. <see cref="PngText"/>
/// puts the result in the file.
///
/// English throughout, whatever the display language: what reads these back is
/// another program, and A1111 is what it will have been written against. The
/// numbers are invariant for the same reason — a comma decimal would split a
/// value in two at the reader's comma.</summary>
public static class A1111Parameters
{
    /// <summary>Everything a run needs to be repeated. The family says which
    /// extra files have a key in the convention — the VAE does — and those
    /// are written by name, as the model is; and whether clip skip means
    /// anything for it. The version names the program that made the image.</summary>
    public static string Format(GenerationOptions options, int seed, IModelFamily family,
        string version = "NaiveDiffusion")
    {
        var text = new StringBuilder(options.Prompt);
        // A1111 applies a LoRA from a tag in the prompt, and that is where
        // every reader looks for which ones an image used — so the prompt
        // is written as it would have been typed over there.
        foreach (LoraSpec lora in options.Loras)
        {
            text.Append(text.Length > 0 ? " " : "")
                .Append(LoraTags.Format(Path.GetFileNameWithoutExtension(lora.Path), lora.Weight));
        }
        if (options.Negative.Length > 0)
        {
            text.Append("\nNegative prompt: ").Append(options.Negative);
        }
        return text
            .Append($"\nSteps: {options.Steps}")
            .Append($", Sampler: {Name(options.Sampler)}")
            .Append($", Schedule type: {Name(options.Schedule)}")
            .Append(Invariant($", CFG scale: {options.Guidance:0.##}"))
            .Append($", Seed: {seed}")
            .Append($", Size: {options.Width}x{options.Height}")
            .Append($", Model: {Path.GetFileNameWithoutExtension(options.CheckpointPath)}")
            .Append(NamedComponents(options, family.Components))
            // Written even at 2: it is the default here and not in A1111,
            // and a reader that sees no key assumes 1. Not written for a
            // family whose text side has no such setting.
            .Append(family.MaxClipSkip > 0 ? $", Clip skip: {options.ClipSkip}" : "")
            // A1111's key for the image-to-image strength. The reference image
            // itself is not recorded, as it is not there either.
            .Append(options.ReferenceImage is null
                ? ""
                : Invariant($", Denoising strength: {options.Strength:0.##}"))
            .Append($", Version: {version}")
            .ToString();
    }

    private static string NamedComponents(GenerationOptions options,
        IReadOnlyList<ModelComponent> components)
    {
        var text = new StringBuilder();
        foreach (ModelComponent component in components)
        {
            if (component.ParametersKey is string key
                && options.ComponentPath(component.Id) is string path)
            {
                text.Append($", {key}: {Path.GetFileNameWithoutExtension(path)}");
            }
        }
        return text.ToString();
    }

    private static string Invariant(FormattableString text) =>
        text.ToString(CultureInfo.InvariantCulture);

    private static string Name(SamplerKind sampler) => sampler switch
    {
        SamplerKind.Euler => "Euler",
        SamplerKind.EulerAncestral => "Euler a",
        SamplerKind.DpmPlusPlus2M => "DPM++ 2M",
        SamplerKind.DpmPlusPlus2MSde => "DPM++ 2M SDE",
        _ => sampler.ToString(),
    };

    /// <summary>"Automatic" is the convention's name for the schedule a sampler
    /// comes with, which for us is Leading.</summary>
    private static string Name(ScheduleKind schedule) => schedule switch
    {
        ScheduleKind.Leading => "Automatic",
        ScheduleKind.Linspace => "Normal",
        ScheduleKind.Karras => "Karras",
        ScheduleKind.Exponential => "Exponential",
        ScheduleKind.AlignYourSteps => "Align Your Steps",
        _ => schedule.ToString(),
    };
}
