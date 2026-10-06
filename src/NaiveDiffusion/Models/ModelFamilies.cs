using NaiveDiffusion.Dml;
using NaiveDiffusion.Images;
using NaiveDiffusion.Models.Anima;
using NaiveDiffusion.Models.QwenImage;
using NaiveDiffusion.Models.Sdxl;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Weights;
using static NaiveDiffusion.Models.CheckpointInspector;

namespace NaiveDiffusion.Models;

/// <summary>Every family this library runs, and which one a checkpoint belongs
/// to. Callers come in here: they hand over a path, learn the family from
/// its header, build their options from that family's parts, and generate
/// through the pipeline the family gets. This is the one place that names
/// more than one family.</summary>
public static class ModelFamilies
{
    public static readonly IReadOnlyList<IModelFamily> All =
        new IModelFamily[] { SdxlFamily.Instance, AnimaFamily.Instance, QwenImageFamily.Instance };

    private static readonly Dictionary<IModelFamily, GenerationPipeline> Pipelines =
        All.ToDictionary(family => family, family => new GenerationPipeline(family));

    /// <summary>Every part any family takes, once per id, for a command line
    /// that has to accept them all before it knows which family it is
    /// serving.</summary>
    public static IReadOnlyList<ModelComponent> AllComponents { get; } =
        All.SelectMany(family => family.Components).DistinctBy(part => part.Id).ToList();

    /// <summary>The family that runs the checkpoint at <paramref name="path"/>,
    /// or null with <paramref name="report"/> saying what the file is instead
    /// — the inspector's verdict, which knows every kind of file this library
    /// has been handed and how to say so. Throws
    /// <see cref="InvalidDataException"/> for a file that is not safetensors
    /// at all.</summary>
    public static IModelFamily? Identify(string path, out CheckpointReport report)
    {
        using var file = new SafetensorsFile(path);
        return Identify(file, out report);
    }

    /// <param name="required">Which of an SDXL checkpoint's parts the caller
    /// is going to read: a tool that only decodes has no business turning a
    /// checkpoint away over its text encoders. A family whose checkpoint has
    /// no parts to be missing is unaffected.</param>
    public static IModelFamily? Identify(SafetensorsFile file, out CheckpointReport report,
        CheckpointParts required = CheckpointParts.All)
    {
        report = CheckpointInspector.Inspect(file);
        CheckpointReport relaxed = report with { Present = report.Present | (CheckpointParts.All & ~required) };
        return All.FirstOrDefault(family => family.Runs(relaxed));
    }

    /// <summary>The family for a checkpoint, or a
    /// <see cref="CheckpointNotSupportedException"/> that carries the verdict.</summary>
    public static IModelFamily Require(string path)
    {
        return Identify(path, out CheckpointReport report)
            ?? throw new CheckpointNotSupportedException(report,
                $"{path}: {CheckpointInspector.Explain(report)}");
    }

    public static GenerationPipeline PipelineFor(IModelFamily family) => Pipelines[family];

    /// <inheritdoc cref="GenerationPipeline.Generate(DmlDevice, GenerationOptions, IReadOnlyList{int}, IProgress{Snapshot}?, CancellationToken, Action{int, ImageResult}?, DenoiserCache?, PromptCache?)"/>
    /// <remarks>Through the family the checkpoint's header names.</remarks>
    public static ImageResult[] Generate(DmlDevice device, GenerationOptions options,
        IReadOnlyList<int> seeds, IProgress<Snapshot>? progress = null,
        CancellationToken cancellation = default, Action<int, ImageResult>? onImage = null,
        DenoiserCache? cache = null, PromptCache? prompts = null)
    {
        if (!File.Exists(options.CheckpointPath))
        {
            throw new FileNotFoundException(
                GenerationPipeline.NoCheckpointMessage, options.CheckpointPath);
        }
        return PipelineFor(Require(options.CheckpointPath))
            .Generate(device, options, seeds, progress, cancellation, onImage, cache, prompts);
    }

    /// <inheritdoc cref="Generate(DmlDevice, GenerationOptions, IReadOnlyList{int}, IProgress{Snapshot}?, CancellationToken, Action{int, ImageResult}?, DenoiserCache?, PromptCache?)"/>
    public static ImageResult Generate(DmlDevice device, GenerationOptions options,
        IProgress<Snapshot>? progress = null, CancellationToken cancellation = default,
        DenoiserCache? cache = null, PromptCache? prompts = null)
        => Generate(device, options, new[] { options.Seed }, progress, cancellation, cache: cache,
            prompts: prompts)[0];
}
