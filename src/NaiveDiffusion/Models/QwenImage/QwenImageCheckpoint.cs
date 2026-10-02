using NaiveDiffusion.Weights;
using static NaiveDiffusion.Weights.SafetensorsInspector;

namespace NaiveDiffusion.Models.QwenImage;

/// <summary>The names in Qwen-Image 2.1's files. The transformer ships
/// under diffusers' bare names (Comfy-Org's qwen_image_2.1_bf16) or under
/// ComfyUI's <c>model.diffusion_model.</c>; the text encoder is Qwen3-VL-8B
/// under transformers' names with the language model's prefix already
/// folded away (<c>model.layers.*</c>, with the vision tower beside it under
/// <c>model.visual.</c>); the VAE is under Wan's names. This is where a file
/// is recognized and where the prefix comes off; the graphs read the bare
/// names.</summary>
public static class QwenImageCheckpoint
{
    /// <summary>The prefixes a checkpoint's transformer may sit under.</summary>
    public static readonly string[] DitPrefixes = { "", "model.diffusion_model." };

    /// <summary>The tensors that together say "Qwen-Image 2.1 and no other
    /// transformer": the text projection's zero-centred norm, the one
    /// modulation shared by every block, and a block's query norm. The
    /// earlier Qwen-Image has <c>txt_norm</c> and per-block modulations;
    /// Flux has double blocks.</summary>
    private static readonly string[] Markers =
    {
        "txt_in.text_norm.weight", "modulation.1.weight",
        "transformer_blocks.0.attn.norm_q.weight", "img_in.weight", "proj_out.weight",
    };

    /// <summary>The feed-forward's first projection, fused ([gate; up] in one
    /// matrix, as Comfy-Org saves it) or split (diffusers' own names).</summary>
    public const string FusedGateUp = "img_mlp.gate_up.weight";
    public const string GateProjection = "img_mlp.gate_layer.weight";
    public const string UpProjection = "img_mlp.proj.weight";

    /// <summary>The prefix the transformer sits under in this file, or null
    /// when the file is not a Qwen-Image 2.1 checkpoint.</summary>
    public static string? DitPrefix(SafetensorsFile file)
    {
        foreach (string prefix in DitPrefixes)
        {
            if (Markers.All(marker => Has(file, prefix + marker))
                && (Has(file, $"{prefix}transformer_blocks.0.{FusedGateUp}")
                    || Has(file, $"{prefix}transformer_blocks.0.{GateProjection}")))
            {
                return prefix;
            }
        }
        return null;
    }

    public static bool IsQwenImage(SafetensorsFile file) => DitPrefix(file) is not null;

    /// <summary>What the transformer is built like, from its header: the
    /// model width, the block count and the latent channels it reads. The
    /// release is 4096 wide with 32 blocks over 64 channels.</summary>
    public static (int Width, int Blocks, int InputChannels) Shape(SafetensorsFile file)
    {
        string prefix = DitPrefix(file)
            ?? throw new InvalidDataException("not a Qwen-Image 2.1 checkpoint");
        if (!file.TryGetInfo(prefix + "img_in.weight", out _, out int[] embedding) || embedding.Length != 2)
        {
            throw new InvalidDataException("the image embedding has an unexpected shape");
        }
        int blocks = 0;
        while (Has(file, $"{prefix}transformer_blocks.{blocks}.attn.to_q.weight"))
        {
            blocks++;
        }
        return (embedding[0], blocks, embedding[1]);
    }

    /// <summary>The width of the text encoder rows the transformer projects.</summary>
    public static int ContextWidth(SafetensorsFile file)
    {
        string prefix = DitPrefix(file)
            ?? throw new InvalidDataException("not a Qwen-Image 2.1 checkpoint");
        return file.TryGetInfo(prefix + "txt_in.text_norm.weight", out _, out int[] norm) && norm.Length == 1
            ? norm[0]
            : throw new InvalidDataException("the text norm has an unexpected shape");
    }

    /// <summary>Whether a file holds a Qwen3-VL: a Qwen3 language model
    /// (<see cref="SafetensorsInspector.IsQwen3TextEncoder"/>) with the
    /// vision tower beside it. A plain Qwen3-8B has the width and not the
    /// tower, and reads a prompt into other numbers.</summary>
    public static bool IsQwen3Vl(SafetensorsFile file) =>
        IsQwen3TextEncoder(file) && Has(file, "model.visual.patch_embed.proj.weight");

    /// <summary>Whether a file holds the Qwen3-VL-8B the transformer was
    /// trained against: a Qwen3-VL 4096 wide.</summary>
    public static bool IsTextEncoder(SafetensorsFile file) =>
        IsQwen3Vl(file) && TextEncoderWidth(file) == QwenImageTextEncoder.Width;

    /// <summary>Whether a file is the Qwen-Image 2.1 VAE: a Wan-family VAE
    /// whose decoder reads sixty-four channels and writes four, with a
    /// temporal kernel of one.</summary>
    public static bool IsVae(SafetensorsFile file) =>
        IsWanVae(file)
        && WanVaeLatentChannels(file) == Vae.Wan22Vae.LatentChannels
        && file.TryGetInfo("decoder.head.2.weight", out _, out int[] head)
        && head.Length == 5 && head[0] == Vae.Wan22Vae.ImageChannels && head[2] == 1;
}
