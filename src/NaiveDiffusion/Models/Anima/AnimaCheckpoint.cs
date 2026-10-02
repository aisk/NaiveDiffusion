using NaiveDiffusion.Weights;
using static NaiveDiffusion.Weights.SafetensorsInspector;

namespace NaiveDiffusion.Models.Anima;

/// <summary>The names in Anima's files. The transformer ships under one of
/// two prefixes — <c>net.</c> as the training code saved it, or ComfyUI's
/// <c>model.diffusion_model.</c> — with the adapter that reads the text
/// encoder folded in beside it under <c>llm_adapter.</c>; the text encoder
/// and the VAE are separate files under their own bare names. This is where a
/// file is recognized and where the prefix comes off; the graphs read the
/// bare names.</summary>
public static class AnimaCheckpoint
{
    /// <summary>The prefixes a checkpoint's transformer may sit under.</summary>
    public static readonly string[] DitPrefixes = { "net.", "model.diffusion_model." };

    public const string AdapterPrefix = "llm_adapter.";

    /// <summary>The one weight that says both "Cosmos-shaped transformer" and
    /// "with a text adapter": Anima and nothing else has it.</summary>
    private const string AdapterMarker = AdapterPrefix + "blocks.0.cross_attn.q_proj.weight";

    /// <summary>The patch embedding, whose shape gives the width and the
    /// input channels.</summary>
    private const string PatchEmbedding = "x_embedder.proj.1.weight";

    /// <summary>The prefix the transformer sits under in this file, or null
    /// when the file is not an Anima checkpoint.</summary>
    public static string? DitPrefix(SafetensorsFile file)
    {
        foreach (string prefix in DitPrefixes)
        {
            if (Has(file, prefix + AdapterMarker) && Has(file, prefix + PatchEmbedding))
            {
                return prefix;
            }
        }
        return null;
    }

    public static bool IsAnima(SafetensorsFile file) => DitPrefix(file) is not null;

    /// <summary>What the transformer is built like, from its header: the
    /// model width and the block count. The 2B release is 2048 wide with 28
    /// blocks; a wider one would be another Cosmos size.</summary>
    public static (int Width, int Blocks, int InputChannels) Shape(SafetensorsFile file)
    {
        string prefix = DitPrefix(file)
            ?? throw new InvalidDataException("not an Anima checkpoint");
        if (!file.TryGetInfo(prefix + PatchEmbedding, out _, out int[] patch) || patch.Length != 2)
        {
            throw new InvalidDataException("the patch embedding has an unexpected shape");
        }
        int blocks = 0;
        while (Has(file, $"{prefix}blocks.{blocks}.mlp.layer1.weight"))
        {
            blocks++;
        }
        // Sixteen latent channels plus the padding-mask channel, times a
        // 2×2 patch.
        return (patch[0], blocks, patch[1] / 4 - 1);
    }

    /// <summary>Whether a file holds the Qwen3 text encoder Anima's adapter
    /// was trained against: a Qwen3 language model
    /// (<see cref="SafetensorsInspector.IsQwen3TextEncoder"/>) 1024 wide.
    /// The 0.6B has no vision tower; one with a tower beside the language
    /// model is another model and reads the prompt into other numbers.</summary>
    public static bool IsTextEncoder(SafetensorsFile file) =>
        IsQwen3TextEncoder(file) && TextEncoderWidth(file) == LlmAdapter.Width
        && !HasPrefix(file, "model.visual.");

    /// <summary>Whether a file is the Wan 2.1 VAE, sixteen latent
    /// channels, which Anima decodes with.</summary>
    public static bool IsVae(SafetensorsFile file) =>
        IsWanVae(file) && WanVaeLatentChannels(file) == Vae.WanVae.LatentChannels;
}
