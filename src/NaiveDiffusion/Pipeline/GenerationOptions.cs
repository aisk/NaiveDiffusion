using NaiveDiffusion.Images;
using NaiveDiffusion.Sampling;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Pipeline;

/// <summary>Everything one generation is asked for. Nothing here belongs to
/// one model family: the prompt, the size, the sampler and the memory
/// choices mean the same thing whichever model is behind the checkpoint.</summary>
public sealed record GenerationOptions
{
    public string Prompt { get; init; } = "";
    public string Negative { get; init; } = "";
    public int Width { get; init; } = 1024;
    public int Height { get; init; } = 1024;
    public int Steps { get; init; } = 20;
    public float Guidance { get; init; } = 5.0f;
    public int Seed { get; init; }
    public SamplerKind Sampler { get; init; } = SamplerKind.Euler;
    public ScheduleKind Schedule { get; init; } = ScheduleKind.Leading;

    /// <summary>The checkpoint: a single-file LDM checkpoint holding all
    /// four of SDXL's models, or the transformer of a family that keeps its
    /// other parts in <see cref="Components"/>. The header says which family
    /// it is. Required.</summary>
    public string CheckpointPath { get; init; } = "";

    /// <summary>Files standing in for, or adding to, what the checkpoint
    /// carries, by the id of the part each plays — see
    /// <see cref="IModelFamily.Components"/> for the parts a family has.
    /// A part not named here comes from the checkpoint; a part the family
    /// does not have is refused, not ignored.</summary>
    public IReadOnlyDictionary<string, string> Components { get; init; } =
        new Dictionary<string, string>();

    /// <summary>The file given for a part, or null for the checkpoint's own.</summary>
    public string? ComponentPath(string id) =>
        Components.TryGetValue(id, out string? path) && path.Length > 0 ? path : null;

    /// <summary>Which of the text encoder's layers the prompt is read from,
    /// counted from the end the way A1111 counts: 1 is the last layer, 2
    /// the one before it. 2 is what SDXL was trained on and what every
    /// SDXL finetune means by its recommended value; 1 sharpens the
    /// reading of the words at the cost of what the model expects. A family
    /// whose text side has no such choice accepts only the default.</summary>
    public int ClipSkip { get; init; } = DefaultClipSkip;

    public const int DefaultClipSkip = 2;

    /// <summary>LoRAs to fold into the checkpoint's diffusion model and
    /// text towers for this run, in order, each at its own weight. Folded on
    /// the CPU as the weights are read, so they cost a few seconds of
    /// loading and nothing per step — and a cached denoiser is only reused
    /// by a run with the same set at the same weights. A family that cannot
    /// fold them (<see cref="IModelFamily.SupportsLoras"/>) refuses a run
    /// that names any.</summary>
    public IReadOnlyList<LoraSpec> Loras { get; init; } = Array.Empty<LoraSpec>();

    /// <summary>Run the VAE in overlapping tiles — the decoder on every
    /// image, the encoder on a reference image — which cuts the largest
    /// allocations a run makes down to a fixed cost, at the price of an
    /// approximation across the seams. An image of one tile or less goes
    /// through identically either way.</summary>
    public bool TileVae { get; init; } = true;

    /// <summary>Video memory the diffusion model's weights may take. Null
    /// puts them all there, which is fastest and needs about 5.5 GiB for
    /// SDXL's UNet and 3.7 for Anima's transformer. A smaller figure puts
    /// that much there and leaves the rest in system memory for the GPU to
    /// read across the bus at every step — slower by the time that takes,
    /// and the only way onto a card that cannot hold the weights and the
    /// scratch together. Zero keeps none in video memory.
    /// <see cref="DenoiserAutoBudget"/> derives one from what the card has
    /// free; a figure given here wins over it.</summary>
    public ulong? DenoiserResidentBytes { get; init; }

    /// <summary>Set the resident budget from what the card has free at the
    /// moment the diffusion model is built — after the previous run's model
    /// and the pool's buffers have been let go, which is why the pipeline
    /// takes the reading rather than the caller: taken any earlier it
    /// counts memory the run is about to release, and streams weights
    /// the card had room for. On a card with room for everything the
    /// figure comes out above the weights' size and nothing changes.</summary>
    public bool DenoiserAutoBudget { get; init; }

    /// <summary>What the diffusion model's large weight matrices are kept
    /// as in video memory. <see cref="WeightStorage.Int8"/> quantizes them
    /// in blocks as they are read: about half the weight memory, with the
    /// arithmetic still at <see cref="DenoiserCompute"/>. The images change
    /// slightly; the quantization error is a few times fp16's own rounding.
    /// A family that cannot store them so
    /// (<see cref="IModelFamily.SupportsWeights"/>) refuses the run.</summary>
    public WeightStorage DenoiserWeights { get; init; } = WeightStorage.Float16;

    /// <summary>What the diffusion model's blocks compute at. For a
    /// transformer <see cref="ComputePrecision.Float16"/> narrows the
    /// products and the attention and leaves the residual stream, the norms
    /// and the rotary positions wide: half the arithmetic and half the
    /// activation traffic, with fine detail coming out slightly differently
    /// from <see cref="ComputePrecision.Float32"/>, which is what the
    /// reference implementations compute at. SDXL's UNet computes at half
    /// precision throughout and has no other. A family refuses a precision
    /// it does not have (<see cref="IModelFamily.SupportsCompute"/>).</summary>
    public ComputePrecision DenoiserCompute { get; init; } = ComputePrecision.Float16;

    /// <summary>An image to start from instead of noise, at exactly
    /// <see cref="Width"/> by <see cref="Height"/> — the caller crops and
    /// scales, so that what is shown as the reference is what runs. It is
    /// encoded once through the VAE and every seed samples from it; the
    /// prompts still steer as they do from noise. Null starts from
    /// noise.</summary>
    public ImageResult? ReferenceImage { get; init; }

    /// <summary>How much of the reference to redo, in (0, 1]: the sampler
    /// runs the last <c>Steps × Strength</c> of its schedule, starting
    /// from the reference noised to that level. 1 lets nothing of the
    /// reference survive but its composition; around 0.3 keeps it and
    /// changes the rendering. Ignored without a reference.</summary>
    public float Strength { get; init; } = 0.65f;
}

/// <summary>What a diffusion model's large weight matrices are stored as
/// once loaded, whatever width the file has them at.</summary>
public enum WeightStorage
{
    Float16,
    /// <summary>Block-quantized 8-bit integers with a scale per block,
    /// dequantized in the graph.</summary>
    Int8,
}

/// <summary>The floating-point width a diffusion model's blocks compute at.</summary>
public enum ComputePrecision
{
    Float16,
    Float32,
}

public static class Precisions
{
    /// <summary>The short name a command line takes and a message prints.</summary>
    public static string Label(this WeightStorage weights) => weights switch
    {
        WeightStorage.Float16 => "fp16",
        WeightStorage.Int8 => "int8",
        _ => weights.ToString(),
    };

    public static string Label(this ComputePrecision compute) => compute switch
    {
        ComputePrecision.Float16 => "fp16",
        ComputePrecision.Float32 => "fp32",
        _ => compute.ToString(),
    };
}

/// <summary>The stages of a run, in the order they happen; what a
/// <see cref="Snapshot"/> says it is in.</summary>
public enum Stage
{
    /// <summary>Reading the text encoder's weights off the files: seconds
    /// for a large one, and nothing has been encoded yet.</summary>
    LoadingTextEncoder,
    EncodingPrompt,
    /// <summary>The reference image through the VAE encoder; only with one.</summary>
    EncodingImage,
    /// <summary>Reading, converting, uploading and compiling the diffusion
    /// model — skipped when a cached one fits. The weights are read as the
    /// graphs are built, so loading and building are one wait.</summary>
    BuildingDenoiser,
    Sampling,
    /// <summary>Reading the VAE's weights and compiling the decoder, after
    /// the last sampling step and before the first image.</summary>
    LoadingDecoder,
    Decoding,
    Done,
}

public static class Stages
{
    /// <summary>The stage's English name, in lower case: what the memory
    /// log labels its readings with and what the CLI prints. A front end
    /// has its own wording.</summary>
    public static string Label(this Stage stage) => stage switch
    {
        Stage.LoadingTextEncoder => "loading text encoder",
        Stage.EncodingPrompt => "encoding prompt",
        Stage.EncodingImage => "encoding image",
        Stage.BuildingDenoiser => "building denoiser",
        Stage.Sampling => "sampling",
        Stage.LoadingDecoder => "loading decoder",
        Stage.Decoding => "decoding",
        Stage.Done => "done",
        _ => stage.ToString(),
    };
}

/// <summary>Where a run is. <paramref name="Preview"/> is the latent as it
/// stood at the end of the step, projected straight to colour: a
/// reduced-scale, approximate look at the image taking shape. Sampling steps
/// carry one; the other stages have nothing to show.</summary>
public sealed record Snapshot(Stage Stage, int Step, int TotalSteps, TimeSpan Elapsed,
    int Image = 0, int TotalImages = 1, ImageResult? Preview = null);
