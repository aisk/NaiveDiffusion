using NaiveDiffusion.Dml;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Sampling;
using NaiveDiffusion.Text;
using NaiveDiffusion.Vae;
using NaiveDiffusion.Weights;
using static NaiveDiffusion.Models.CheckpointInspector;

namespace NaiveDiffusion.Models.QwenImage;

/// <summary>Qwen-Image 2.1 as the pipeline runs it: Qwen3-VL-8B's language
/// model over a chat template on the CPU, the single-stream transformer as
/// a chain of graphs, classifier-free guidance that is off at a scale of
/// 1 — the model is released to run without it — the Wan 2.2-layout VAE
/// on one frame, and rectified flow with a shift set from the image's
/// size. Three files, the text encoder and the VAE declared as required
/// parts. Text to image only, for now: the reference images an edit
/// splices into the sequence, the alpha the VAE also makes and the text's
/// cache across steps are not done yet, and what is not done is said
/// through the capabilities so the pipeline refuses it.</summary>
public sealed class QwenImageFamily : IModelFamily
{
    /// <summary>Qwen3-VL-8B, as Comfy-Org repacks it
    /// (qwen3vl_8b_bf16.safetensors): the transformer was trained on this
    /// model's hidden states and no other reads the prompt for it. The
    /// vision tower in the file is not read.</summary>
    public static readonly ModelComponent TextEncoderComponent =
        new("text_encoder", "text encoder", Required: true, PipelineParts.Conditioner);

    /// <summary>The Qwen-Image 2.1 VAE (qwen_image_2.1_vae_bf16.safetensors):
    /// sixteen pixels a cell, sixty-four channels, RGBA.</summary>
    public static readonly ModelComponent VaeComponent =
        new("vae", "VAE", Required: true, PipelineParts.Codec, ParametersKey: "VAE");

    public static readonly QwenImageFamily Instance = new();

    /// <summary>The scheduler's dynamic shift, from the release's scheduler
    /// config: the log-shift runs linearly from 0.5 at 256 image tokens to
    /// 0.9 at 8192, so 1024² (4096 tokens) gets 0.69, which is also the
    /// constant ComfyUI settled on.</summary>
    private const double BaseTokens = 256, MaxTokens = 8192, BaseShift = 0.5, MaxShift = 0.9;

    private QwenImageFamily()
    {
    }

    public string Name => "qwen-image";

    public IConditioner Conditioner { get; } = new QwenImageConditioner();

    public ILatentCodec Codec { get; } = new QwenImageLatentCodec();

    public GuidanceStrategy Guidance => OptionalClassifierFreeGuidance.Instance;

    /// <summary>The release's: no guidance, one branch.</summary>
    public float DefaultGuidance => 1f;

    public IReadOnlyList<ModelComponent> Components { get; } =
        new[] { TextEncoderComponent, VaeComponent };

    /// <summary>The transformer, stored at a width this library reads; the
    /// checkpoint has no other parts to be missing.</summary>
    public bool Runs(CheckpointReport report) =>
        report.Kind == CheckpointKind.QwenImage && report.UnsupportedDataType is null;

    /// <summary>Qwen3-VL is read at its last layer only, as the transformer
    /// was trained.</summary>
    public int MaxClipSkip => 0;

    /// <summary>Folded into the transformer and the text encoder under the
    /// names ComfyUI reads them by (<see cref="QwenImageLoras"/>).</summary>
    public bool SupportsLoras => true;

    public bool SupportsWeights(WeightStorage weights) => true;

    public bool SupportsCompute(ComputePrecision compute) => true;

    public ComputePrecision DefaultCompute => ComputePrecision.Float16;

    /// <summary>Sentences: the prompt goes into Qwen3-VL's template as
    /// written, with no splitting and no weights, and the model is prompted
    /// in prose, Chinese included.</summary>
    public PromptStyle PromptStyle => PromptStyle.Sentences;

    /// <summary>One latent cell is one token and sixteen pixels, so a side
    /// has to divide by sixteen.</summary>
    public int SizeAlignment => QwenImageDit.Patch * Wan22Vae.Latent.ScaleFactor;

    /// <summary>No Align Your Steps table has been solved for Qwen-Image.</summary>
    public bool SupportsSchedule(ScheduleKind schedule) =>
        schedule != ScheduleKind.AlignYourSteps;

    /// <summary>A flow model starts from pure noise: the first sigma has to
    /// be 1, which evenly spaced from the last training timestep gives.</summary>
    public ScheduleKind DefaultSchedule => ScheduleKind.Linspace;

    /// <summary>The transformer's gemms have the cell count for rows, a
    /// multiple of 64 at the usual sizes; nothing to advise.</summary>
    public bool SizeWastesMemory(int height, int width) => false;

    /// <summary>1834 MiB measured at 1024² with each block cut into six
    /// links (one block as one graph took 9 GiB: a compiled graph's scratch
    /// is the sum of its intermediates); 2048 leaves a margin.</summary>
    public ulong DenoiserScratchBytes(GenerationOptions options) => DenoiserBudget.ScaleScratch(
        2048UL << 20, options.Height, options.Width);

    public int? NearbyEfficientHeight(int height, int width, int step, int reach, int minimum,
        int maximum) => null;

    public ComponentVerdict InspectComponent(ModelComponent component, string path)
    {
        if (component == TextEncoderComponent)
        {
            return ComponentVerdicts.Inspect(path, file =>
            {
                if (!SafetensorsInspector.IsQwen3TextEncoder(file))
                {
                    return ComponentVerdict.NotThisComponent;
                }
                // A Qwen3 without the vision tower — Anima's 0.6B, a plain
                // 8B — or a smaller Qwen3-VL is a text encoder, but not the
                // one this was trained on.
                return QwenImageCheckpoint.IsTextEncoder(file)
                    ? ComponentVerdict.Accepted
                    : ComponentVerdict.WrongArchitecture;
            });
        }
        if (component == VaeComponent)
        {
            return ComponentVerdicts.Inspect(path, file =>
            {
                if (QwenImageCheckpoint.IsVae(file))
                {
                    return ComponentVerdict.Accepted;
                }
                // Wan 2.1's sixteen channels, or Stable Diffusion's four:
                // a VAE, but not one this transformer was trained against.
                return SafetensorsInspector.IsWanVae(file)
                       || SafetensorsInspector.IsBareVae(file)
                       || SafetensorsInspector.HasPrefix(file, "first_stage_model.")
                    ? ComponentVerdict.WrongArchitecture
                    : ComponentVerdict.NotThisComponent;
            });
        }
        throw new ArgumentException($"Qwen-Image has no component named {component.Id}");
    }

    /// <summary>The shift for this run's image: exp of the log-shift the
    /// token count reads off the line, baked into the training table.</summary>
    public ModelSampling Sampling(CheckpointReport report, GenerationOptions options)
    {
        int scale = Wan22Vae.Latent.ScaleFactor * QwenImageDit.Patch;
        double tokens = (double)(options.Height / scale) * (options.Width / scale);
        return new ModelSampling(new FlowShiftSchedule(Math.Exp(Mu(tokens))), FlowPrediction.Instance);
    }

    /// <summary>diffusers' calculate_shift over the release's constants.</summary>
    public static double Mu(double imageTokens)
    {
        double m = (MaxShift - BaseShift) / (MaxTokens - BaseTokens);
        double b = BaseShift - m * BaseTokens;
        return imageTokens * m + b;
    }

    /// <summary>Nothing beyond what the pipeline checks for every family.</summary>
    public void Validate(GenerationOptions options)
    {
    }

    public IDenoiser BuildDenoiser(DmlDevice device, GenerationOptions options,
        Conditioning conditioning, ulong? residentBudget, CancellationToken cancellation)
    {
        using var weights = new QwenImageWeights(options.CheckpointPath);
        using LazyWeights parameters = weights.LoadDit(options.Loras);
        cancellation.ThrowIfCancellationRequested();
        var dit = new QwenImageDit(device, parameters, options.Height, options.Width,
            ((QwenImageConditioning)conditioning).Rows, residentBudget,
            options.DenoiserWeights == WeightStorage.Int8,
            options.DenoiserCompute == ComputePrecision.Float16);
        parameters.Dispose();
        return dit;
    }

    public void ReadDenoiserWeights(string checkpointPath, Action<LazyWeights> read)
    {
        using var weights = new QwenImageWeights(checkpointPath);
        using LazyWeights parameters = weights.LoadDit();
        read(parameters);
    }
}
