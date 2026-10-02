# NaiveDiffusion

Text-to-image on DirectML, written in C#. No Python and no ONNX. It reads the `.safetensors` files ComfyUI and A1111 use, builds the DirectML graphs itself and runs them on whatever GPU Direct3D 12 can see.

## Supported models

| Model | Files |
| --- | --- |
| SDXL (Illustrious, NoobAI, Pony, v-pred included) | one checkpoint, optional VAE |
| Anima | DiT + Qwen3-0.6B text encoder + Qwen-Image VAE |
| Qwen-Image 2.1 | DiT + Qwen3-VL-8B text encoder + 2.1 VAE |

There is also image-to-image, LoRA (SDXL and Anima), tiled VAE, int8 weights, and streaming the weights from system memory when they don't fit on the card.

Windows only, .NET 10.

## Usage

```csharp
using NaiveDiffusion.Dml;
using NaiveDiffusion.Models;
using NaiveDiffusion.Pipeline;

using var device = new DmlDevice();

var image = ModelFamilies.Generate(device, new GenerationOptions
{
    CheckpointPath = @"D:\models\illustriousXL.safetensors",
    Prompt = "1girl, hatsune miku, city street, night",
    Negative = "lowres, bad anatomy",
    Width = 832,
    Height = 1216,
    Steps = 20,
    Seed = 1,
});

// image.Rgb24 holds image.Width * image.Height packed RGB pixels
```

Models with more than one file take the rest through `Components`.

```csharp
var options = new GenerationOptions
{
    CheckpointPath = "anima-base-v1.0.safetensors",
    Components = new Dictionary<string, string>
    {
        ["text_encoder"] = "qwen_3_06b_base.safetensors",
        ["vae"] = "qwen_image_vae.safetensors",
    },
    Schedule = ScheduleKind.Linspace,
    Guidance = 4,
    Prompt = "...",
};
```

## Command line

```
cd src/NaiveDiffusion.Cli
dotnet run -c Release -- devices
dotnet run -c Release -- generate "1girl, hatsune miku" --size 1024 --steps 20 --seed 1 --checkpoint model.safetensors
```

Run it without arguments for the full list of commands. `smoke` checks the graph layers against a CPU reference and needs no model.

## Performance

Measured on a Radeon RX 6800 (16 GB) with an 8-core Ryzen, at 1024×1024. VRAM is the peak while sampling.

| Model | Settings | Per step | Whole run | VRAM |
| --- | --- | --- | --- | --- |
| SDXL | 20 steps, CFG | 2.4 s | 58 s | 6.1 GiB |
| Anima | 20 steps, guidance 4 | 7.0 s | 150 s | 5.7 GiB |
| Anima | same, `--fp16-compute` | 6.0 s | 130 s | 5.7 GiB |
| Anima turbo | 8 steps, guidance 1 | 3.5 s | 40 s | 5.7 GiB |
| Qwen-Image 2.1 | 25 steps, guidance 1 | 12.6 s | 410 s | 12.4 GiB |
| Qwen-Image 2.1 | same, `--fp16-compute` | 10.2 s | 349 s | 11.1 GiB |
| Qwen-Image 2.1 | same, `--fp16-compute --int8` | 10.4 s | 355 s | 8.5 GiB |

Guidance 1 runs the model once per step, anything above runs it twice. The Qwen-Image rows use `--unet-vram auto`, which streams the part of its 13.5 GiB of weights that doesn't fit. Decoding a Qwen-Image picture takes 8.5 GiB on its own, the other two stay under 2 GiB.

On a smaller card the SDXL UNet can keep part of its weights in system memory and stream them every step, or store them as int8. Streaming gives the same image bit for bit, int8 changes fine detail.

| SDXL, 20 steps | Per step | VRAM |
| --- | --- | --- |
| everything in VRAM | 2.3 s | 6.1 GiB |
| 1.7 GiB of the weights in VRAM | 2.8 s | 3.6 GiB |
| `--unet-vram 0` | 3.3 s | 1.6 GiB |
| `--int8` | 2.6 s | 3.4 GiB |
| `--int8 --unet-vram 0` | 2.9 s | 1.4 GiB |

## Building

Some of the DirectML operators used here are newer than the last [Vortice.Windows](https://github.com/amerkoleci/Vortice.Windows) release, so for now it builds against a checkout of Vortice's main branch placed next to this repository.

```
git clone https://github.com/amerkoleci/Vortice.Windows
git clone https://github.com/aisk/NaiveDiffusion
cd NaiveDiffusion
dotnet build
dotnet test
```

The tests don't touch the GPU.

## License

MIT. The embedded tokenizer data has its own licenses, listed in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
