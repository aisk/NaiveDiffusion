using NaiveDiffusion.Dml;
using NaiveDiffusion.Models;
using NaiveDiffusion.Sampling;
using NaiveDiffusion.Text;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Pipeline;

/// <summary>One model family — SDXL, Anima, Qwen-Image, whatever comes next
/// — as the parts <see cref="GenerationPipeline"/> runs in order: the text
/// side, the diffusion model, the guidance that combines its runs, the VAE at
/// both ends, and what the sampler has to be set up with to drive that model.
/// One object hands out all of them so that they can share types of their
/// own — what the conditioner produces is what the denoiser reads, and the
/// pipeline never looks inside.
///
/// The family also answers what it can do — which options apply to it, what
/// its sizes must divide by — so that the pipeline can refuse a run the
/// family cannot make in one place, and a front end can hide or grey out what
/// does not apply, without either naming a family.</summary>
public interface IModelFamily
{
    /// <summary>Short and stable: part of a cached denoiser's key.</summary>
    string Name { get; }

    IConditioner Conditioner { get; }

    ILatentCodec Codec { get; }

    GuidanceStrategy Guidance { get; }

    /// <summary>The guidance scale a front end fills in when a model of this
    /// family is picked and a command line left without one gets: the one
    /// the model's own reference runs with. At 1 a family whose
    /// <see cref="Guidance"/> can turn off runs one branch and never
    /// encodes the negative prompt, so a default borrowed from another
    /// family costs twice the time for another image.</summary>
    float DefaultGuidance { get; }

    /// <summary>The files this family can take besides the checkpoint, in
    /// the order a front end should list them. Empty for a family that runs
    /// from the checkpoint alone.</summary>
    IReadOnlyList<ModelComponent> Components { get; }

    /// <summary>Whether a checkpoint the inspector has read is one this
    /// family runs, with everything it needs inside.</summary>
    bool Runs(CheckpointInspector.CheckpointReport report);

    // --- What the family can do ------------------------------------------

    /// <summary>How far back from the end the text encoder can be read —
    /// A1111's clip skip count, 12 for CLIP's twelve layers — or 0 for a
    /// family whose text side has no such choice, which then accepts only
    /// <see cref="GenerationOptions.DefaultClipSkip"/>.</summary>
    int MaxClipSkip { get; }

    /// <summary>Whether LoRAs can be folded into this family's weights.</summary>
    bool SupportsLoras { get; }

    /// <summary>Whether this family's diffusion model can keep its weights
    /// as <paramref name="weights"/>
    /// (<see cref="GenerationOptions.DenoiserWeights"/>).</summary>
    bool SupportsWeights(WeightStorage weights);

    /// <summary>Whether this family's diffusion model can compute at
    /// <paramref name="compute"/>
    /// (<see cref="GenerationOptions.DenoiserCompute"/>): the transformers
    /// have both, SDXL's UNet half precision only. A front end offers the
    /// choice where more than one is supported.</summary>
    bool SupportsCompute(ComputePrecision compute);

    /// <summary>The precision a front end starts at and a command line left
    /// without one gets.</summary>
    ComputePrecision DefaultCompute { get; }

    /// <summary>How this family's encoder reads the prompt — as tags with
    /// weights, or as sentences taken as written — so a prompt box can show
    /// it the fitting way and stop pointing at things that are not wrong.</summary>
    PromptStyle PromptStyle { get; }

    /// <summary>What each side of an image must be a multiple of: the
    /// latent scale for a UNet, the patch size on top of it for a
    /// transformer.</summary>
    int SizeAlignment { get; }

    /// <summary>Whether a noise-level spacing can be made for this family's
    /// models — Align Your Steps needs a table solved for the model, and
    /// not every model has one.</summary>
    bool SupportsSchedule(ScheduleKind schedule);

    /// <summary>The spacing a front end's "default" stands for and a command
    /// line left without one gets: the one the model's own reference runs with.
    /// Not the same for every family — a flow model has to start from pure
    /// noise, sigma 1, which SDXL's leading spacing stops short of.</summary>
    ScheduleKind DefaultSchedule { get; }

    /// <summary>Whether this image size makes the denoiser cost more video
    /// memory than the size itself needs — for SDXL, a token count off the
    /// gemm alignment duplicates the widest weights — so a caller can say so.</summary>
    bool SizeWastesMemory(int height, int width);

    /// <summary>The closest height to <paramref name="height"/> that does not,
    /// searched in <paramref name="step"/>-pixel moves out to
    /// <paramref name="reach"/> and kept inside [<paramref name="minimum"/>,
    /// <paramref name="maximum"/>]; null when nothing in range qualifies, or
    /// when the family has no such concern.</summary>
    int? NearbyEfficientHeight(int height, int width, int step, int reach, int minimum, int maximum);

    /// <summary>The scratch one step of the denoiser takes at this size, as
    /// measured — what <see cref="DenoiserBudget"/> leaves free before it
    /// says how much of the weights may stay in video memory. The families
    /// differ fourfold here, so there is no figure to share.</summary>
    ulong DenoiserScratchBytes(GenerationOptions options);

    /// <summary>Whether a file will do for one of <see cref="Components"/>,
    /// from its header — cheap enough to run the moment it is picked.
    /// Never throws for a file that is not safetensors; that is a
    /// verdict.</summary>
    ComponentVerdict InspectComponent(ModelComponent component, string path);

    // --- One run -----------------------------------------------------------

    /// <summary>The noise levels this run's diffusion model was trained at
    /// and what its output means — per checkpoint, because one architecture
    /// ships in several trainings, and the header says which; and per run,
    /// because a flow model may set its shift from the image's size. Called
    /// after the options have passed every other check.</summary>
    ModelSampling Sampling(CheckpointInspector.CheckpointReport report, GenerationOptions options);

    /// <summary>Refuse a run the family's files cannot finish, before
    /// anything is loaded. The pipeline has already checked everything the
    /// families share — the sizes, the step count, that the files exist and
    /// are the parts they were given as, that the options apply to this
    /// family — so what is left is the family's own: a LoRA that is not one,
    /// say.</summary>
    void Validate(GenerationOptions options);

    /// <summary>Build the diffusion model for this run: its size, its
    /// conditioning's shape, and the options' memory choices, with
    /// <paramref name="residentBudget"/> already resolved by the pipeline
    /// (null keeps every weight in video memory).</summary>
    IDenoiser BuildDenoiser(DmlDevice device, GenerationOptions options,
        Conditioning conditioning, ulong? residentBudget, CancellationToken cancellation);

    /// <summary>Hand <paramref name="read"/> the diffusion model's weights
    /// as <see cref="BuildDenoiser"/> reads them — the same tensors, at the
    /// same width, from the same file — without building anything, so what
    /// narrowing them costs can be measured. The tensors are windows onto
    /// the mapped checkpoint and live only for the call.</summary>
    void ReadDenoiserWeights(string checkpointPath, Action<LazyWeights> read);
}
