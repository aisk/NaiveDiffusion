using NaiveDiffusion.Dml;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Sampling;
using NaiveDiffusion.Text;
using NaiveDiffusion.Vae;
using NaiveDiffusion.Weights;
using static NaiveDiffusion.Models.CheckpointInspector;

namespace NaiveDiffusion.Models.Anima;

/// <summary>Anima as the pipeline runs it: a Qwen3-0.6B text encoder read
/// through the checkpoint's own adapter on the CPU, the Cosmos-shaped
/// transformer as a chain of graphs, classifier-free guidance that is off
/// at a scale of 1 — where the turbo model runs — the Wan 2.1
/// VAE on one frame, and rectified flow with a shift of 3 — three files where
/// SDXL has one, the text encoder and the VAE declared as required parts.
/// What it does not do yet — reading the text encoder short of its last
/// layer — it says so through the capabilities, and the pipeline refuses
/// the option.</summary>
public sealed class AnimaFamily : IModelFamily
{
    /// <summary>The Qwen3-0.6B base model, as ComfyUI ships it
    /// (qwen_3_06b_base.safetensors): the checkpoint's adapter was trained
    /// against this exact model, so no other reads the prompt for it.</summary>
    public static readonly ModelComponent TextEncoderComponent =
        new("text_encoder", "text encoder", Required: true, PipelineParts.Conditioner);

    /// <summary>The Wan 2.1 video VAE, or the Qwen-Image VAE that shares its
    /// weights (qwen_image_vae.safetensors), on a single frame.</summary>
    public static readonly ModelComponent VaeComponent =
        new("vae", "VAE", Required: true, PipelineParts.Codec, ParametersKey: "VAE");

    public static readonly AnimaFamily Instance = new();

    /// <summary>ComfyUI's sampling settings for the family: shift 3 on the
    /// straight line, every release the same.</summary>
    private static readonly ModelSampling Flow =
        new(new FlowShiftSchedule(3.0), FlowPrediction.Instance);

    private AnimaFamily()
    {
    }

    public string Name => "anima";

    public IConditioner Conditioner { get; } = new AnimaConditioner();

    public ILatentCodec Codec { get; } = new AnimaLatentCodec();

    /// <summary>Off at 1: the turbo model is released to run without
    /// guidance, and running the negative prompt at that scale only doubled
    /// every step for a prediction the arithmetic then threw away.</summary>
    public GuidanceStrategy Guidance => OptionalClassifierFreeGuidance.Instance;

    /// <summary>ComfyUI's for the base model; the turbo model wants 1,
    /// which its header does not tell apart.</summary>
    public float DefaultGuidance => 4f;

    public IReadOnlyList<ModelComponent> Components { get; } =
        new[] { TextEncoderComponent, VaeComponent };

    /// <summary>The transformer with its adapter, stored at a width this library
    /// reads; the checkpoint has no other parts to be missing.</summary>
    public bool Runs(CheckpointReport report) =>
        report.Kind == CheckpointKind.Anima && report.UnsupportedDataType is null;

    /// <summary>Qwen3 is read at its last layer only: the adapter was trained
    /// on that and nothing else.</summary>
    public int MaxClipSkip => 0;

    /// <summary>Folded into the transformer, its adapter and the text
    /// encoder under the names ComfyUI reads them by
    /// (<see cref="AnimaLoras"/>).</summary>
    public bool SupportsLoras => true;

    public bool SupportsWeights(WeightStorage weights) => true;

    public bool SupportsCompute(ComputePrecision compute) => true;

    public ComputePrecision DefaultCompute => ComputePrecision.Float16;

    /// <summary>Tags: the model is trained on Danbooru tags, and its encoder
    /// takes the prompt as weighted segments (<see cref="AnimaPrompt"/>).</summary>
    public PromptStyle PromptStyle => PromptStyle.Tags;

    /// <summary>The transformer's patches are two latent cells, eight pixels
    /// each, so a side has to divide by sixteen; Cosmos pads in the graph,
    /// this does not.</summary>
    public int SizeAlignment => AnimaDit.Patch * WanVae.Latent.ScaleFactor;

    /// <summary>No Align Your Steps table has been solved for Anima.</summary>
    public bool SupportsSchedule(ScheduleKind schedule) =>
        schedule != ScheduleKind.AlignYourSteps;

    /// <summary>ComfyUI's: evenly spaced from the last training timestep,
    /// so the first sigma is 1 and the noise the run starts from is all the
    /// model is told is there. Leading starts at 0.98 on twenty steps and
    /// 0.95 on the turbo model's eight — a start the model reads as holding
    /// that much of an image, which the noise does not. It is also the
    /// spacing every comparison against ComfyUI was made on.</summary>
    public ScheduleKind DefaultSchedule => ScheduleKind.Linspace;

    /// <summary>The transformer's gemms have the patch count for rows, which
    /// is a multiple of 64 at the usual sizes; nothing to advise.</summary>
    public bool SizeWastesMemory(int height, int width) => false;

    /// <summary>2048 MiB measured for the transformer at 1024²: its blocks
    /// compute at single precision (see <see cref="AnimaDit"/>), four times
    /// what the UNet takes at half.</summary>
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
                // Another Qwen3 — a wider one, or Qwen-Image's with its
                // vision tower — is a text encoder, but not the one the
                // adapter was trained on.
                return AnimaCheckpoint.IsTextEncoder(file)
                    ? ComponentVerdict.Accepted
                    : ComponentVerdict.WrongArchitecture;
            });
        }
        if (component == VaeComponent)
        {
            return ComponentVerdicts.Inspect(path, file =>
            {
                if (SafetensorsInspector.IsWanVae(file))
                {
                    // Qwen-Image 2.1's sixty-four channels are a Wan VAE too.
                    return AnimaCheckpoint.IsVae(file)
                        ? ComponentVerdict.Accepted
                        : ComponentVerdict.WrongArchitecture;
                }
                // Stable Diffusion's autoencoder, bare or inside a checkpoint:
                // a VAE, but not one this transformer was trained against.
                return SafetensorsInspector.IsBareVae(file)
                       || SafetensorsInspector.HasPrefix(file, "first_stage_model.")
                    ? ComponentVerdict.WrongArchitecture
                    : ComponentVerdict.NotThisComponent;
            });
        }
        throw new ArgumentException($"Anima has no component named {component.Id}");
    }

    public ModelSampling Sampling(CheckpointReport report, GenerationOptions options) => Flow;

    /// <summary>Nothing beyond what the pipeline checks for every family:
    /// the files' kinds are the components' verdicts, and every option that
    /// does not apply is refused through the capabilities above.</summary>
    public void Validate(GenerationOptions options)
    {
    }

    public IDenoiser BuildDenoiser(DmlDevice device, GenerationOptions options,
        Conditioning conditioning, ulong? residentBudget, CancellationToken cancellation)
    {
        using var weights = new AnimaWeights(options.CheckpointPath);
        using LazyWeights parameters = weights.LoadDit(options.Loras);
        cancellation.ThrowIfCancellationRequested();
        var dit = new AnimaDit(device, parameters, options.Height, options.Width,
            ((AnimaConditioning)conditioning).Rows, residentBudget,
            options.DenoiserWeights == WeightStorage.Int8,
            options.DenoiserCompute == ComputePrecision.Float16);
        parameters.Dispose();
        return dit;
    }

    public void ReadDenoiserWeights(string checkpointPath, Action<LazyWeights> read)
    {
        using var weights = new AnimaWeights(checkpointPath);
        using LazyWeights parameters = weights.LoadDit();
        read(parameters);
    }
}
