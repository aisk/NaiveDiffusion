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
