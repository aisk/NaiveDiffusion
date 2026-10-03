using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Models.QwenImage;

/// <summary>How Qwen-Image 2.1's LoRAs name the two models they change, as
/// ComfyUI's <c>lora.py</c> maps them: the transformer as kohya's
/// <c>lora_unet_…</c>, the generic <c>diffusion_model.…</c> that ai-toolkit
/// and ComfyUI write, diffusers' <c>transformer.…</c>, or the bare
/// <c>transformer_blocks.…</c> of DiffSynth, and the Qwen3-VL text encoder
/// as kohya's single <c>lora_te_…</c> or the generic
/// <c>text_encoders.…</c>. SDXL's two CLIP towers are refused on their
/// names. A LoRA for the earlier Qwen-Image shares some of this
/// transformer's names at another width, and is refused on its shapes.</summary>
public static class QwenImageLoras
{
    public const string Dit = "dit";
    public const string TextEncoder = "te";

    public static readonly LoraLayout Layout = new("Qwen-Image 2.1",
        new[]
        {
            new LoraTower(Dit, "transformer", "unet", "diffusion_model", "transformer", ""),
            new LoraTower(TextEncoder, "Qwen3-VL text encoder", "te", "text_encoders"),
        },
        new Dictionary<string, string>
        {
            ["te1"] = "was trained for SDXL, whose text encoders are two CLIP towers " +
                      "where Qwen-Image has Qwen3-VL",
            ["te2"] = "was trained for SDXL, whose text encoders are two CLIP towers " +
                      "where Qwen-Image has Qwen3-VL",
        });

    /// <summary>The names a LoRA has for the two halves of a fused
    /// feed-forward projection: a checkpoint that stores [gate; up] as one
    /// matrix is still trained on as <c>gate_layer</c> and <c>proj</c>, the
    /// first half of the rows and the second. Nothing for any other weight.</summary>
    public static IEnumerable<LoraRows> FusedHalves(string key, int rows)
    {
        if (!key.EndsWith("." + QwenImageCheckpoint.FusedGateUp, StringComparison.Ordinal))
        {
            yield break;
        }
        string block = key[..^QwenImageCheckpoint.FusedGateUp.Length];
        yield return new LoraRows(block + QwenImageCheckpoint.GateProjection, 0, rows / 2);
        yield return new LoraRows(block + QwenImageCheckpoint.UpProjection, rows / 2, rows / 2);
    }

    /// <summary>The other names a text encoder weight goes by in a LoRA,
    /// for a file keyed <c>model.layers.…</c>: kohya drops the
    /// <c>model.</c>, and ComfyUI's generic names put its wrapper,
    /// <c>qwen3vl_8b.transformer.</c>, in front — or only
    /// <c>transformer.</c>.</summary>
    public static IEnumerable<string> TextEncoderAliases(string key)
    {
        if (key.StartsWith("model.", StringComparison.Ordinal))
        {
            yield return key["model.".Length..];
        }
        yield return "transformer." + key;
        yield return "qwen3vl_8b.transformer." + key;
    }
}
