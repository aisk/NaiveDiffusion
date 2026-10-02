namespace NaiveDiffusion.Weights;

/// <summary>What a .safetensors header says about the file, independent of
/// which diffusion model it might belong to: whether it is a LoRA, a
/// ControlNet, an embedding, a VAE on its own, a diffusion transformer, and
/// whether it is stored at a width this library can read. Each model family's own
/// inspector builds its verdict on top of these.</summary>
public static class SafetensorsInspector
{
    /// <summary>Map a file that may not be safetensors at all: false for
    /// one that is not readable as such — truncated, renamed, a .ckpt in
    /// disguise — where the caller wants a verdict rather than an exception.</summary>
    public static bool TryOpen(string path, [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out SafetensorsFile? file)
    {
        try
        {
            file = new SafetensorsFile(path);
            return true;
        }
        catch (Exception exception) when (exception is InvalidDataException
                                              or System.Text.Json.JsonException or IOException)
        {
            file = null;
            return false;
        }
    }

    public static bool IsLora(SafetensorsFile file)
    {
        if (file.Metadata.TryGetValue("modelspec.architecture", out string? architecture)
            && architecture.EndsWith("/lora", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }
        // The kohya names and the PEFT ones; a LoRA is only ever the pair.
        return AnyKey(file, key =>
            key.Contains(".lora_down.weight", StringComparison.Ordinal)
            || key.Contains(".lora_up.weight", StringComparison.Ordinal)
            || key.Contains(".lora_A.weight", StringComparison.Ordinal)
            || key.Contains(".lora_B.weight", StringComparison.Ordinal)
            || key.StartsWith("lora_unet_", StringComparison.Ordinal)
            || key.StartsWith("lora_te", StringComparison.Ordinal));
    }

    /// <summary>A textual-inversion embedding, under any of the names the
    /// trainers write.</summary>
    public static bool IsEmbedding(SafetensorsFile file) =>
        Has(file, "string_to_param") || Has(file, "emb_params")
        || Has(file, "clip_l") || Has(file, "clip_g");

    public static bool IsControlNet(SafetensorsFile file) =>
        HasPrefix(file, "control_model.") || HasPrefix(file, "input_hint_block.")
        || HasPrefix(file, "zero_convs.");

    /// <summary>SD3 stacks joint blocks, Flux stacks double blocks and the
    /// Cosmos family modulates its blocks with a low-rank adaLN; either way it
    /// is a transformer where the UNet models expect a UNet.</summary>
    public static bool IsDiffusionTransformer(SafetensorsFile file) =>
        AnyKey(file, key => key.Contains("joint_blocks.", StringComparison.Ordinal)
                            || key.Contains("double_blocks.", StringComparison.Ordinal)
                            || key.Contains("blocks.0.adaln_modulation_self_attn.", StringComparison.Ordinal));

    /// <summary>A VAE stored under the diffusers names with nothing in front,
    /// which is how a standalone one is far more often saved.</summary>
    public static bool IsBareVae(SafetensorsFile file) =>
        Has(file, "decoder.conv_out.weight") && Has(file, "encoder.conv_in.weight");

    /// <summary>A Wan-family video VAE — Wan 2.1's, which Anima decodes
    /// with, or the Wan 2.2 layout Qwen-Image 2.1 ships — by the causal 3-D
    /// convolutions its decoder and encoder start with.</summary>
    public static bool IsWanVae(SafetensorsFile file) =>
        Has(file, "decoder.conv1.weight") && Has(file, "decoder.head.2.weight")
        && Has(file, "encoder.conv1.weight");

    /// <summary>The latent width a Wan VAE decodes from: the input channels
    /// of its first decoder convolution. 0 for a file that is not one.</summary>
    public static int WanVaeLatentChannels(SafetensorsFile file) =>
        file.TryGetInfo("decoder.conv1.weight", out _, out int[] shape) && shape.Length == 5
            ? shape[1]
            : 0;

    /// <summary>A Qwen3 language model under transformers' names, with or
    /// without a vision tower beside it: the token embedding and the
    /// per-head query norm Qwen3 has and Qwen2 does not.</summary>
    public static bool IsQwen3TextEncoder(SafetensorsFile file) =>
        Has(file, "model.embed_tokens.weight")
        && Has(file, "model.layers.0.self_attn.q_norm.weight");

    /// <summary>The embedding width of a Llama-shaped text encoder, 0 for a
    /// file that is not one.</summary>
    public static int TextEncoderWidth(SafetensorsFile file) =>
        file.TryGetInfo("model.embed_tokens.weight", out _, out int[] embed) && embed.Length == 2
            ? embed[1]
            : 0;

    /// <summary>Whether the model predicts velocity rather than noise. Both
    /// signals are conventions rather than anything the format guarantees: the
    /// marker tensor is what ComfyUI and A1111 look for, and the metadata key is
    /// what the model-spec trainers write.</summary>
    public static bool IsVelocityPrediction(SafetensorsFile file)
    {
        if (Has(file, "v_pred") || Has(file, "edm_vpred"))
        {
            return true;
        }
        return file.Metadata.TryGetValue("modelspec.prediction_type", out string? prediction)
               && prediction.Trim().StartsWith("v", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>Whether the file says it was trained on the zero-terminal-SNR
    /// rescale of its noise schedule: the <c>ztsnr</c> marker tensor the
    /// v-pred anime finetunes carry, which ComfyUI reads the same way.</summary>
    public static bool IsZeroTerminalSnr(SafetensorsFile file) => Has(file, "ztsnr");

    /// <summary>The first stored width this library cannot read, if the file uses
    /// one. A checkpoint quantized to fp8 is the case that turns up.</summary>
    public static string? UnreadableDataType(SafetensorsFile file)
    {
        foreach (string name in file.Keys)
        {
            if (file.TryGetInfo(name, out string dataType, out _)
                && !SafetensorsFile.IsSupportedDataType(dataType))
            {
                return dataType;
            }
        }
        return null;
    }

    // --- Header lookups ----------------------------------------------------

    public static bool Has(SafetensorsFile file, string name) =>
        file.TryGetInfo(name, out _, out _);

    public static bool HasPrefix(SafetensorsFile file, string prefix) =>
        AnyKey(file, key => key.StartsWith(prefix, StringComparison.Ordinal));

    public static bool AnyKey(SafetensorsFile file, Func<string, bool> match)
    {
        foreach (string name in file.Keys)
        {
            if (match(name))
            {
                return true;
            }
        }
        return false;
    }
}
