using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Models.Anima;

/// <summary>How Anima's LoRAs name the two models they change, as ComfyUI's
/// <c>lora.py</c> maps them: the transformer — its text adapter included —
/// as kohya's <c>lora_unet_…</c> or the generic <c>diffusion_model.…</c>
/// that ai-toolkit and ComfyUI write, and the Qwen3 text encoder as kohya's
/// single <c>lora_te_…</c> or the generic <c>text_encoders.…</c>. SDXL's
/// two CLIP towers are refused on their names, since a file that also
/// changes the UNet would otherwise be refused only for its UNet half, with
/// a less useful reason.</summary>
public static class AnimaLoras
{
    public const string Dit = "dit";
    public const string TextEncoder = "te";

    public static readonly LoraLayout Layout = new("Anima",
        new[]
        {
            new LoraTower(Dit, "transformer", "unet", "diffusion_model"),
            new LoraTower(TextEncoder, "Qwen3 text encoder", "te", "text_encoders"),
        },
        new Dictionary<string, string>
        {
            ["te1"] = "was trained for SDXL, whose text encoders are two CLIP towers " +
                      "where Anima has Qwen3",
            ["te2"] = "was trained for SDXL, whose text encoders are two CLIP towers " +
                      "where Anima has Qwen3",
        });

    /// <summary>The other names a text encoder weight goes by in a LoRA,
    /// for a file keyed <c>model.layers.…</c>: kohya drops the
    /// <c>model.</c>, and ComfyUI's generic names put its wrapper,
    /// <c>qwen3_06b.transformer.</c>, in front — or only
    /// <c>transformer.</c>.</summary>
    public static IEnumerable<string> TextEncoderAliases(string key)
    {
        if (key.StartsWith("model.", StringComparison.Ordinal))
        {
            yield return key["model.".Length..];
        }
        yield return "transformer." + key;
        yield return "qwen3_06b.transformer." + key;
    }
}
