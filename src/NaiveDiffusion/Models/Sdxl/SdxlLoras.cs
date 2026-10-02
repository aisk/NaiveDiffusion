using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Models.Sdxl;

/// <summary>How SDXL's LoRAs name the three models they change: kohya's
/// unet / te1 / te2, PEFT's unet / text_encoder / text_encoder_2. An SD 1.x
/// or 2.x LoRA has one text tower and calls it "te"; it is refused on that
/// name, because its text half would otherwise fold into CLIP-L without
/// complaint — SD 1.x uses the same tower — and only its UNet half would
/// fail, a stage later.</summary>
public static class SdxlLoras
{
    public const string Unet = "unet";
    public const string TextEncoder = "te1";
    public const string TextEncoder2 = "te2";

    public static readonly LoraLayout Layout = new("SDXL",
        new[]
        {
            new LoraTower(Unet, "UNet", "unet", "unet"),
            new LoraTower(TextEncoder, "CLIP-L", "te1", "text_encoder"),
            new LoraTower(TextEncoder2, "CLIP-G", "te2", "text_encoder_2"),
        },
        new Dictionary<string, string>
        {
            ["te"] = "was trained for Stable Diffusion 1.x or 2.x, which has one text " +
                     "tower where SDXL has two",
        });
}
