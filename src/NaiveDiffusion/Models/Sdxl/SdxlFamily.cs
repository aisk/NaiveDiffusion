using NaiveDiffusion.Dml;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Sampling;
using NaiveDiffusion.Text;
using NaiveDiffusion.Text.Clip;
using NaiveDiffusion.Weights;
using static NaiveDiffusion.Models.CheckpointInspector;

namespace NaiveDiffusion.Models.Sdxl;

/// <summary>Stable Diffusion XL as the pipeline runs it: two CLIP towers on
/// the CPU, the UNet as a chain of graphs, classifier-free guidance,
/// diffusers' AutoencoderKL, and the scaled-linear schedule — epsilon or
/// v-prediction, with or without the zero-terminal-SNR rescale, as the
/// checkpoint's markers say.</summary>
public sealed class SdxlFamily : IModelFamily
{
    /// <summary>A VAE to decode and encode with instead of the checkpoint's:
    /// the fp16-fix that does not overflow, or the one a finetune was trained
    /// against. The only part SDXL takes from outside its single file — the
    /// towers and the UNet are what the checkpoint is.</summary>
    /// Declared before <see cref="Instance"/>: the instance lists it, and static
    /// fields initialize in the order written.
    public static readonly ModelComponent VaeComponent =
        new("vae", "VAE", Required: false, PipelineParts.Codec, ParametersKey: "VAE");

    public static readonly SdxlFamily Instance = new();

    private static readonly NoiseSchedule ZeroTerminalSnrLevels =
        VpScaledLinearSchedule.Sdxl.WithZeroTerminalSnr();

    private SdxlFamily()
    {
    }

    public string Name => "sdxl";

    public IConditioner Conditioner { get; } = new SdxlConditioner();

    public ILatentCodec Codec { get; } = new SdxlLatentCodec();

    public GuidanceStrategy Guidance => ClassifierFreeGuidance.Instance;

    /// <summary>What a run starts at when no scale is given.</summary>
    public float DefaultGuidance => 5f;

    public IReadOnlyList<ModelComponent> Components { get; } = new[] { VaeComponent };

    /// <summary>A base checkpoint with all four models inside, stored at a
    /// width this library reads. A refiner, an earlier Stable Diffusion, or a
    /// base checkpoint pruned of its VAE or a tower is not one.</summary>
    public bool Runs(CheckpointReport report) =>
        report.Kind == CheckpointKind.SdxlBase
        && report.Present == CheckpointParts.All
        && report.UnsupportedDataType is null;

    public int MaxClipSkip => ClipTextEncoder.MaxClipSkip;

    public bool SupportsLoras => true;

    public bool SupportsHalfCompute => false;

    /// <summary>Tags with weights, the A1111 way: the two CLIP towers see
    /// the prompt as runs of tokens, each with the weight of its group.</summary>
    public PromptStyle PromptStyle => PromptStyle.Tags;

    public int SizeAlignment => SdxlVae.Latent.ScaleFactor;

    /// <summary>Every spacing has a table for SDXL, Align Your Steps
    /// included — the v-prediction finetunes reuse the base model's.</summary>
    public bool SupportsSchedule(ScheduleKind schedule) => true;

    /// <summary>The published config's: the training timesteps taken evenly,
    /// leading.</summary>
    public ScheduleKind DefaultSchedule => ScheduleKind.Leading;

    public bool SizeWastesMemory(int height, int width) =>
        UNetModel.WeightsAreDuplicated(height, width);

    /// <summary>512 MiB measured for the UNet at 1024², and 768 with int8
    /// weights: each dispatch writes the dequantized copy of its matrices
    /// into the scratch.</summary>
    public ulong DenoiserScratchBytes(GenerationOptions options) => DenoiserBudget.ScaleScratch(
        (options.DenoiserInt8Weights ? 768UL : 512UL) << 20, options.Height, options.Width);

    public int? NearbyEfficientHeight(int height, int width, int step, int reach, int minimum,
        int maximum) => UNetModel.NearbyAligned(height, width, step, reach, minimum, maximum);

    public ComponentVerdict InspectComponent(ModelComponent component, string path)
    {
        if (component != VaeComponent)
        {
            throw new ArgumentException($"SDXL has no component named {component.Id}");
        }
        if (!CheckpointInspector.TryInspect(path, out CheckpointReport report))
        {
            return ComponentVerdict.Unreadable;
        }
        // A standalone VAE, or a checkpoint whose VAE is the one wanted — a
        // finetune that ships its VAE baked in is the usual source of one.
        if (report.Kind != CheckpointKind.VaeOnly && !report.Carries(CheckpointParts.Vae))
        {
            return ComponentVerdict.NotThisComponent;
        }
        if (report.VaeLatentChannels != SdxlVae.Latent.Channels)
        {
            return ComponentVerdict.WrongArchitecture;
        }
        if (report.UnsupportedDataType is not null)
        {
            return ComponentVerdict.UnsupportedDataType;
        }
        return ComponentVerdict.Accepted;
    }

    /// <summary>From the two marker tensors the v-pred finetunes carry:
    /// <c>v_pred</c> for what the UNet puts out, <c>ztsnr</c> for the table it
    /// was trained against. The base model has neither and is epsilon on
    /// the published betas.</summary>
    public ModelSampling Sampling(CheckpointReport report, GenerationOptions options) => new(
        report.ZeroTerminalSnr ? ZeroTerminalSnrLevels : VpScaledLinearSchedule.Sdxl,
        report.VelocityPrediction ? VelocityPrediction.Instance : EpsilonPrediction.Instance);

    /// <summary>Nothing beyond what the pipeline checks for every family.</summary>
    public void Validate(GenerationOptions options)
    {
    }

    public IDenoiser BuildDenoiser(DmlDevice device, GenerationOptions options,
        Conditioning conditioning, ulong? residentBudget, CancellationToken cancellation)
    {
        // The tensors are windows onto the mapped checkpoint, so it stays open
        // until the graphs have been compiled and the weights uploaded.
        using var checkpoint = new SdxlWeights(options.CheckpointPath);
        using LazyWeights parameters = checkpoint.LoadUnet(options.Loras);
        cancellation.ThrowIfCancellationRequested();
        var unet = new UNetModel(device, parameters, options.Height, options.Width,
            ((SdxlConditioning)conditioning).Tokens, residentBudget: residentBudget,
            int8Weights: options.DenoiserInt8Weights);
        parameters.Dispose();
        return unet;
    }

    public void ReadDenoiserWeights(string checkpointPath, Action<LazyWeights> read)
    {
        using var checkpoint = new SdxlWeights(checkpointPath);
        using LazyWeights parameters = checkpoint.LoadUnet();
        read(parameters);
    }
}
