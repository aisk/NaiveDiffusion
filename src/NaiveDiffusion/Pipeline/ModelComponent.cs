namespace NaiveDiffusion.Pipeline;

/// <summary>One file a family runs from besides the checkpoint, named by
/// the part it plays: a VAE to use in place of the checkpoint's own, a text
/// encoder the checkpoint does not carry. The family declares which parts
/// it has and whether each must be given; the callers build an option per
/// part from the list and never name a part themselves, so a family with
/// three files and one with a single file share the same front end and the
/// same command line.</summary>
/// <param name="Id">The part's stable name, lower-case: the key in
/// <see cref="GenerationOptions.Components"/>, the CLI's option
/// (<c>--vae</c>), and a stem a front end can key its strings on.</param>
/// <param name="Name">The part's English name, for the CLI's wording.</param>
/// <param name="Required">Whether a run needs the file. An optional part
/// left out is taken from the checkpoint.</param>
/// <param name="Part">Which part of the pipeline reads the file, so a
/// caller that runs only some parts asks only for their files.</param>
/// <param name="ParametersKey">The key A1111's parameters block records
/// the file under, or null for a part it has no key for.</param>
public sealed record ModelComponent(string Id, string Name, bool Required, PipelineParts Part,
    string? ParametersKey = null);

/// <summary>The parts of a family a caller runs, as
/// <see cref="IModelFamily"/> hands them out: a full generation runs all
/// three, a VAE roundtrip the codec alone, one forward pass of the
/// diffusion model on conditioning read from a file the denoiser alone.
/// <see cref="GenerationPipeline.Prepare"/> asks for the files of the parts
/// that will run and no others.</summary>
[Flags]
public enum PipelineParts
{
    Conditioner = 1,
    Denoiser = 2,
    Codec = 4,
    All = Conditioner | Denoiser | Codec,
}

/// <summary>What a family makes of a file offered for one of its
/// components, from the header alone.</summary>
public enum ComponentVerdict
{
    Accepted,

    /// <summary>Not a readable safetensors file at all.</summary>
    Unreadable,

    /// <summary>A file of another kind: a checkpoint offered as a VAE.</summary>
    NotThisComponent,

    /// <summary>The right kind of file, made for another family: a VAE with
    /// a different latent width.</summary>
    WrongArchitecture,

    /// <summary>Stored at a width this library cannot read.</summary>
    UnsupportedDataType,
}

public static class ComponentVerdicts
{
    /// <summary>The verdict on a file offered for a component, from its
    /// header: unreadable and quantized are the same answer for every
    /// family, and <paramref name="verdict"/> gives the family's own for a
    /// file that is neither.</summary>
    public static ComponentVerdict Inspect(string path,
        Func<Weights.SafetensorsFile, ComponentVerdict> verdict)
    {
        if (!Weights.SafetensorsInspector.TryOpen(path, out Weights.SafetensorsFile? file))
        {
            return ComponentVerdict.Unreadable;
        }
        using (file)
        {
            if (Weights.SafetensorsInspector.UnreadableDataType(file) is not null)
            {
                return ComponentVerdict.UnsupportedDataType;
            }
            return verdict(file);
        }
    }

    /// <summary>The verdict as a sentence, in English, for the CLI and the
    /// exceptions; a front end has its own wording.</summary>
    public static string Explain(this ComponentVerdict verdict, ModelComponent component) =>
        verdict switch
        {
            ComponentVerdict.Accepted => "accepted",
            ComponentVerdict.Unreadable => "not a readable safetensors file",
            ComponentVerdict.NotThisComponent => $"this file is not a {component.Name}",
            ComponentVerdict.WrongArchitecture =>
                $"this {component.Name} was made for another model family",
            ComponentVerdict.UnsupportedDataType =>
                "the weights are stored at a width this library cannot read; " +
                "pick a file that is not quantized",
            _ => verdict.ToString(),
        };
}
