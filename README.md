# NaiveDiffusion

<img src="assets/example.png" align="right" width="280" alt="A picture made with Qwen-Image 2.1">

Text-to-image on DirectML, written in C#. No Python and no ONNX. It reads the `.safetensors` files ComfyUI and A1111 use, builds the DirectML graphs itself and runs them on whatever GPU Direct3D 12 can see.

It runs three families of models.

- **SDXL**, Illustrious, NoobAI, Pony and v-pred included, from one checkpoint with an optional VAE
- **Anima**, from its DiT, the Qwen3-0.6B text encoder and the Qwen-Image VAE
- **Qwen-Image 2.1**, from its DiT, the Qwen3-VL-8B text encoder and the 2.1 VAE

There is also image-to-image, LoRA, tiled VAE, int8 weights, and streaming the weights from system memory when they don't fit on the card.

Windows only, .NET 10.

<br clear="right">

## Usage

The picture above came out of this, with Qwen-Image 2.1.

```csharp
using NaiveDiffusion.Dml;
using NaiveDiffusion.Models;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Sampling;

using var device = new DmlDevice();

var image = ModelFamilies.Generate(device, new GenerationOptions
{
    CheckpointPath = "qwen_image_2.1_bf16.safetensors",
    Components = new Dictionary<string, string>
    {
        ["text_encoder"] = "qwen3vl_8b_bf16.safetensors",
        ["vae"] = "qwen_image_2.1_vae_bf16.safetensors",
    },
    Prompt = "Beautiful Asian Frieren cosplayer in a modern data center, photorealistic, " +
             "wearing a safety helmet and holding an open laptop while troubleshooting servers, " +
             "silver-white hair, pointed elf ears, Frieren-inspired outfit, " +
             "Chinese douyin-style cosplay makeup, porcelain skin, defined eyeliner, aegyo-sal, " +
             "long eyelashes, glowing server racks, cinematic lighting, shallow depth of field, " +
             "realistic skin and fabric, professional photography, high detail.",
    Width = 1024,
    Height = 1344,
    Steps = 25,
    Guidance = 1,
    Schedule = ScheduleKind.Linspace,
    Seed = 525712554,
    DenoiserAutoBudget = true,
    DenoiserCompute = ComputePrecision.Float32,
});

image.SavePng("frieren.png");
```

The result is packed RGB pixels in `image.Rgb24`. `ToPng()` gives the encoded bytes, `ToBgra32()` and `ToRgba32()` give the layouts WriteableBitmap, WPF, System.Drawing and most image libraries take.

Qwen-Image and Anima keep the text encoder and the VAE in files of their own, named through `Components`. An SDXL checkpoint has everything in one file.

```csharp
var image = ModelFamilies.Generate(device, new GenerationOptions
{
    CheckpointPath = "illustriousXL.safetensors",
    Prompt = "1girl, hatsune miku, city street, night",
    Negative = "lowres, bad anatomy",
    Width = 832,
    Height = 1216,
    Steps = 20,
    Seed = 1,
});
```

A LoRA is folded into the weights before they are uploaded, at any strength, and several can be stacked. All three families take them, under the names kohya, ai-toolkit, diffusers, DiffSynth and ComfyUI write. LoHa, LoKr and DoRA are refused.

```csharp
Loras = new[] { new LoraSpec("turbo8_lora.safetensors", 1f) },
```

## Command line

```
cd src/NaiveDiffusion.Cli
dotnet run -c Release -- devices
dotnet run -c Release -- generate "1girl, hatsune miku" --size 1024 --steps 20 --seed 1 --checkpoint model.safetensors
dotnet run -c Release -- generate "1girl, hatsune miku" --lora style.safetensors:0.8 --checkpoint model.safetensors
```

Run it without arguments for the full list of commands. `smoke` checks the graph layers against a CPU reference and needs no model.

## Performance

Measured on a Radeon RX 6800 (16 GB) with an 8-core Ryzen, at 1024×1024. VRAM is the peak while sampling.

| Model | Settings | Per step | Whole run | VRAM |
| --- | --- | --- | --- | --- |
| SDXL | 20 steps, CFG | 2.4 s | 58 s | 6.1 GiB |
| Anima | 20 steps, guidance 4 | 6.0 s | 130 s | 5.7 GiB |
| Anima | same, `--compute fp32` | 7.0 s | 150 s | 5.7 GiB |
| Anima turbo | 8 steps, guidance 1, `--compute fp32` | 3.5 s | 40 s | 5.7 GiB |
| Qwen-Image 2.1 | 25 steps, guidance 1 | 10.2 s | 349 s | 11.1 GiB |
| Qwen-Image 2.1 | same, `--compute fp32` | 12.6 s | 410 s | 12.4 GiB |
| Qwen-Image 2.1 | same, `--weights int8` | 10.4 s | 355 s | 8.5 GiB |
| Qwen-Image 2.1 with the Turbo8 LoRA | 8 steps, guidance 1 | 10.4 s | 176 s | 11.1 GiB |

Guidance 1 runs the model once per step, anything above runs it twice. The two transformers compute their blocks at half precision unless `--compute fp32` asks for single, which is what ComfyUI computes at and costs about a fifth more time. SDXL's UNet is half precision throughout. The Qwen-Image rows use `--denoiser-vram auto`, which streams the part of its 13.5 GiB of weights that doesn't fit. Decoding a Qwen-Image picture takes 8.5 GiB on its own, the other two stay under 2 GiB.

On a smaller card the SDXL UNet can keep part of its weights in system memory and stream them every step, or store them as int8. Streaming gives the same image bit for bit, int8 changes fine detail.

| SDXL, 20 steps | Per step | VRAM |
| --- | --- | --- |
| everything in VRAM | 2.3 s | 6.1 GiB |
| 1.7 GiB of the weights in VRAM | 2.8 s | 3.6 GiB |
| `--denoiser-vram 0` | 3.3 s | 1.6 GiB |
| `--weights int8` | 2.6 s | 3.4 GiB |
| `--weights int8 --denoiser-vram 0` | 2.9 s | 1.4 GiB |

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
