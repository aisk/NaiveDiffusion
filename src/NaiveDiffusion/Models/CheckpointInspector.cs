using NaiveDiffusion.Models.Anima;
using NaiveDiffusion.Models.QwenImage;
using NaiveDiffusion.Models.Sdxl;
using NaiveDiffusion.Weights;
using static NaiveDiffusion.Weights.SafetensorsInspector;

namespace NaiveDiffusion.Models;

/// <summary>What a .safetensors file turns out to be, decided from its header
/// alone. The header is the JSON table of names, dtypes and offsets that sits in
/// front of the data; reading it costs a page fault or two, so this can run the
/// moment a file is picked instead of after a minute of loading.
///
/// The point is to fail where a user can still do something about it. Without
/// this, an SD 1.5 checkpoint is accepted, spends the prompt encoding, and
/// dies on a key name from the diffusers side that appears nowhere in the file;
/// an SDXL checkpoint with the VAE stripped out samples to completion first and
/// throws away every image on the way to the decoder.
///
/// It reports rather than decides: which family, if any, runs a file is
/// the families' answer to the report (<see cref="Pipeline.IModelFamily.Runs"/>),
/// and the wording belongs to the caller — a front end may have it in
/// several languages, the CLI has one, and neither wants an English sentence
/// baked into a library exception.</summary>
public static class CheckpointInspector
{
    /// <summary>What the file is. <see cref="SdxlBase"/>, <see cref="Anima"/>
    /// and <see cref="QwenImage"/> are the kinds a family here runs.</summary>
    public enum CheckpointKind
    {
        /// <summary>An SDXL 1.0 base checkpoint, run by the SDXL family.</summary>
        SdxlBase,

        /// <summary>An Anima transformer with its text adapter, run by the
        /// Anima family with its two other files.</summary>
        Anima,

        /// <summary>A Qwen-Image 2.1 transformer, run by the Qwen-Image
        /// family with its two other files.</summary>
        QwenImage,

        /// <summary>The SDXL refiner, which carries only the bigG tower and is a
        /// different pipeline.</summary>
        SdxlRefiner,

        StableDiffusion1,
        StableDiffusion2,

        /// <summary>SD3, Flux, or another diffusion transformer this library
        /// does not run.</summary>
        MmDit,

        Lora,
        ControlNet,

        /// <summary>A standalone VAE, with no diffusion model in the file.</summary>
        VaeOnly,

        /// <summary>A textual-inversion embedding.</summary>
        Embedding,

        Unknown,
    }

    /// <summary>The four models a single-file SDXL checkpoint has to carry.</summary>
    [Flags]
    public enum CheckpointParts
    {
        None = 0,
        Unet = 1,
        Vae = 2,
        TextEncoder = 4,
        TextEncoder2 = 8,
        All = Unet | Vae | TextEncoder | TextEncoder2,
    }

    /// <summary>The verdict on one file.</summary>
    /// <param name="Kind">What the file is.</param>
    /// <param name="Present">Which of the four models it actually carries.</param>
    /// <param name="VelocityPrediction">Whether the UNet predicts the velocity
    /// rather than the noise, from the <c>v_pred</c> marker or the model-spec
    /// metadata. A convention, not a guarantee: a file that lies about this
    /// samples to a washed-out grey image rather than failing.</param>
    /// <param name="ZeroTerminalSnr">Whether the UNet was trained on the
    /// zero-terminal-SNR rescale of the noise schedule, from the <c>ztsnr</c>
    /// marker.</param>
    /// <param name="UnsupportedDataType">The first stored width this library cannot
    /// read, if any — fp8 quantized checkpoints land here.</param>
    /// <param name="VaeLatentChannels">How many channels the VAE in the file
    /// — the checkpoint's own or a standalone one — decodes from: 4 for every
    /// Stable Diffusion up to XL, 16 for SD3 and Flux. 0 without a VAE.</param>
    public sealed record CheckpointReport(
        CheckpointKind Kind,
        CheckpointParts Present,
        bool VelocityPrediction,
        bool ZeroTerminalSnr,
        string? UnsupportedDataType,
        int VaeLatentChannels = 0)
    {
        /// <summary>Whether some family runs this file. The families say
        /// which (<see cref="Pipeline.IModelFamily.Runs"/>) and this asks
        /// them, for a caller that only wants yes or no: a second copy of
        /// their conditions here would be the one a new family forgot.</summary>
        public bool CanRun
        {
            get
            {
                CheckpointReport report = this;
                return ModelFamilies.All.Any(family => family.Runs(report));
            }
        }

        /// <summary>The models a single-file SDXL checkpoint should have had.</summary>
        public CheckpointParts Missing => CheckpointParts.All & ~Present;

        /// <summary>Whether <paramref name="required"/> is entirely there. The
        /// CLI tools that only touch one model ask for only that one.</summary>
        public bool Carries(CheckpointParts required) => (Present & required) == required;
    }

    /// <summary>Reads the header and hands back the verdict. Throws
    /// <see cref="InvalidDataException"/> if the file is not a readable
    /// safetensors file at all.</summary>
    public static CheckpointReport Inspect(string path)
    {
        using var file = new SafetensorsFile(path);
        return Inspect(file);
    }

    /// <summary><see cref="Inspect(string)"/> for a file that may not be
    /// safetensors at all: false rather than an exception, for the callers
    /// that only want a verdict.</summary>
    public static bool TryInspect(string path, out CheckpointReport report)
    {
        if (!SafetensorsInspector.TryOpen(path, out SafetensorsFile? file))
        {
            report = null!;
            return false;
        }
        using (file)
        {
            report = Inspect(file);
            return true;
        }
    }

    /// <inheritdoc cref="Inspect(string)"/>
    public static CheckpointReport Inspect(SafetensorsFile file)
    {
        CheckpointParts present = PartsIn(file);
        return new CheckpointReport(
            Identify(file, present),
            present,
            SafetensorsInspector.IsVelocityPrediction(file),
            SafetensorsInspector.IsZeroTerminalSnr(file),
            SafetensorsInspector.UnreadableDataType(file),
            VaeLatentChannelsIn(file));
    }

    /// <summary>The input width of the decoder's first convolution, under the
    /// checkpoint prefix or bare; 0 when neither is there.</summary>
    private static int VaeLatentChannelsIn(SafetensorsFile file)
    {
        const string Name = "decoder.conv_in.weight";
        foreach (string candidate in new[] { SdxlCheckpoint.VaePrefix + Name, Name })
        {
            if (file.TryGetInfo(candidate, out _, out int[] shape) && shape.Length == 4)
            {
                return shape[1];
            }
        }
        return 0;
    }

    // --- What is in the file -----------------------------------------------

    private static CheckpointParts PartsIn(SafetensorsFile file)
    {
        CheckpointParts present = CheckpointParts.None;
        if (HasPrefix(file, SdxlCheckpoint.UnetPrefix))
        {
            present |= CheckpointParts.Unet;
        }
        if (HasPrefix(file, SdxlCheckpoint.VaePrefix))
        {
            present |= CheckpointParts.Vae;
        }
        if (HasPrefix(file, SdxlCheckpoint.ClipLPrefix))
        {
            present |= CheckpointParts.TextEncoder;
        }
        if (HasPrefix(file, SdxlCheckpoint.ClipGPrefix))
        {
            present |= CheckpointParts.TextEncoder2;
        }
        return present;
    }

    /// <summary>Whole classes of file first — a LoRA or a ControlNet says so in
    /// its own names and never reaches the architecture question — then which
    /// diffusion model it is.</summary>
    private static CheckpointKind Identify(SafetensorsFile file, CheckpointParts present)
    {
        if (IsLora(file))
        {
            return CheckpointKind.Lora;
        }
        if (IsEmbedding(file))
        {
            return CheckpointKind.Embedding;
        }
        if (IsControlNet(file))
        {
            return CheckpointKind.ControlNet;
        }
        // Before the transformer test: Anima is a Cosmos transformer and
        // would answer to it, and Qwen-Image is a transformer the test
        // does not know.
        if (AnimaCheckpoint.IsAnima(file))
        {
            return CheckpointKind.Anima;
        }
        if (QwenImageCheckpoint.IsQwenImage(file))
        {
            return CheckpointKind.QwenImage;
        }
        if (IsDiffusionTransformer(file))
        {
            return CheckpointKind.MmDit;
        }

        if ((present & CheckpointParts.Unet) == 0)
        {
            // A VAE on its own is stored either under the checkpoint prefix or,
            // far more often, under the diffusers names with nothing in front;
            // the Wan-family VAEs Anima and Qwen-Image use are VAEs too.
            return (present & CheckpointParts.Vae) != 0 || IsBareVae(file)
                   || IsWanVae(file)
                ? CheckpointKind.VaeOnly
                : CheckpointKind.Unknown;
        }

        // SDXL conditions on the resolution it is pretending to have been
        // cropped from, and label_emb is where that embedding lives. No earlier
        // Stable Diffusion has it, and every SDXL has it.
        if (Has(file, SdxlCheckpoint.UnetPrefix + "label_emb.0.0.weight"))
        {
            // The refiner drops CLIP-L and puts bigG in slot 0, so the slot a
            // tower sits in is what tells the two apart. A base checkpoint that
            // was pruned of its towers has neither, and stays a base checkpoint
            // missing them, which is the more useful thing to say.
            return HasPrefix(file, "conditioner.embedders.0.model.")
                   && (present & CheckpointParts.TextEncoder) == 0
                ? CheckpointKind.SdxlRefiner
                : CheckpointKind.SdxlBase;
        }

        // Which of the two earlier versions it is, by the width the UNet
        // cross-attends against: 768 for CLIP ViT-L, 1024 for OpenCLIP ViT-H.
        return CrossAttentionWidth(file) == 1024
            ? CheckpointKind.StableDiffusion2
            : CheckpointKind.StableDiffusion1;
    }

    /// <summary>The context width the UNet cross-attends against, from the first
    /// attention block that has one; 0 if the file does not have that block.</summary>
    private static int CrossAttentionWidth(SafetensorsFile file)
    {
        const string Name = SdxlCheckpoint.UnetPrefix +
            "input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight";
        return file.TryGetInfo(Name, out _, out int[] shape) && shape.Length == 2
            ? shape[1]
            : 0;
    }

    // --- Wording -----------------------------------------------------------

    /// <summary>Why the file cannot be run, in English. This is the CLI's
    /// wording and the message on <see cref="CheckpointNotSupportedException"/>;
    /// a front end can say the same things in its own words from the report.
    /// Every line names what to do next, because "not a checkpoint" on its
    /// own leaves the user holding the same file with no idea which one to
    /// reach for instead.</summary>
    public static string Explain(CheckpointReport report)
    {
        if (report.UnsupportedDataType is string dataType)
        {
            return $"the weights are stored as {dataType}; this library reads F16, BF16 and F32, " +
                   "so pick a checkpoint that is not quantized";
        }
        if (report.Kind != CheckpointKind.SdxlBase)
        {
            // Plain ASCII throughout: this goes to a console whose code page
            // turns anything else into mojibake.
            return report.Kind switch
            {
                CheckpointKind.Anima => "an Anima transformer",
                CheckpointKind.QwenImage => "a Qwen-Image 2.1 transformer",
                CheckpointKind.SdxlRefiner =>
                    "this is the SDXL refiner, which carries only the bigG text encoder; " +
                    "pick an SDXL base checkpoint",
                CheckpointKind.StableDiffusion1 =>
                    "this is a Stable Diffusion 1.x checkpoint; this library runs SDXL, Anima and Qwen-Image 2.1",
                CheckpointKind.StableDiffusion2 =>
                    "this is a Stable Diffusion 2.x checkpoint; this library runs SDXL, Anima and Qwen-Image 2.1",
                CheckpointKind.MmDit =>
                    "this is an SD3 or Flux checkpoint, a diffusion transformer this library " +
                    "does not run; this library runs SDXL, Anima and Qwen-Image 2.1",
                CheckpointKind.Lora =>
                    "this is a LoRA, not a checkpoint; pick the checkpoint it was " +
                    "trained against",
                CheckpointKind.ControlNet =>
                    "this is a ControlNet, not a checkpoint; pick a checkpoint",
                CheckpointKind.VaeOnly =>
                    "this file holds only a VAE, with no diffusion model; pick a " +
                    "single-file SDXL checkpoint, an Anima or a Qwen-Image 2.1 transformer",
                CheckpointKind.Embedding =>
                    "this is a textual-inversion embedding, not a checkpoint",
                _ => "no model.diffusion_model.* tensors - not a single-file SDXL " +
                     "checkpoint, an Anima or a Qwen-Image 2.1 transformer",
            };
        }
        return "this SDXL checkpoint is missing " + NameParts(report.Missing) +
               "; pick one that carries all four models";
    }

    /// <summary>The models named as a list, in English.</summary>
    public static string NameParts(CheckpointParts parts)
    {
        var names = new List<string>(4);
        if ((parts & CheckpointParts.Unet) != 0)
        {
            names.Add("the UNet");
        }
        if ((parts & CheckpointParts.Vae) != 0)
        {
            names.Add("the VAE");
        }
        if ((parts & CheckpointParts.TextEncoder) != 0)
        {
            names.Add("the CLIP-L text encoder");
        }
        if ((parts & CheckpointParts.TextEncoder2) != 0)
        {
            names.Add("the CLIP-G text encoder");
        }
        return names.Count == 0 ? "nothing" : string.Join(", ", names);
    }
}
