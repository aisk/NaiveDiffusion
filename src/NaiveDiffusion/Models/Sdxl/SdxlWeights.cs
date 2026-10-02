using NaiveDiffusion.Tensors;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Models.Sdxl;

/// <summary>Where SDXL's tensors come from: one single-file LDM checkpoint —
/// the format ComfyUI and A1111 use — holding the UNet, the VAE and both text
/// towers, plus, when one is given, a VAE file that stands in for the
/// checkpoint's. <see cref="SdxlCheckpoint"/> translates their names to the
/// ones the graphs want.
///
/// This owns the mapping. The tensors handed out are windows onto it, not
/// copies, so it stays open until they have been uploaded or converted — the
/// caller disposes this after that, never before.
///
/// Reading one model at a time is what keeps the memory discipline: the file is
/// 6.5+ GiB and never has to be resident whole. The VAE is widened to float32 on
/// the way out regardless of how the checkpoint stored them and the UNet is
/// narrowed to the half precision it runs at; the text towers are handed over
/// untouched, because their consumer converts them itself.
///
/// LoRAs are folded in here, on the CPU, into the UNet and the text towers —
/// the VAE is never the target of one. This is the one point every weight
/// passes through between being read and being uploaded, converted or
/// quantized, so a merged weight is indistinguishable downstream from one the
/// checkpoint carried.</summary>
public sealed class SdxlWeights : IDisposable
{
    private readonly SafetensorsFile _checkpoint;
    private readonly string? _vaePath;
    private SafetensorsFile? _vae;

    /// <param name="vaePath">A VAE to use in place of the checkpoint's own —
    /// the fp16-fix, or one a finetune shipped beside it — either bare, as
    /// the standalone files are, or under the checkpoint prefix. Mapped only
    /// when the VAE is asked for, so the towers and the UNet never pay for
    /// it. Null takes the checkpoint's.</param>
    public SdxlWeights(string checkpointPath, string? vaePath = null)
    {
        _checkpoint = new SafetensorsFile(checkpointPath);
        _vaePath = vaePath;
    }

    /// <summary>The VAE's tensors as float32.</summary>
    public Dictionary<string, HostTensor> LoadVae()
    {
        Dictionary<string, HostTensor> raw;
        if (_vaePath is null)
        {
            raw = _checkpoint.ReadPrefix(SdxlCheckpoint.VaePrefix);
        }
        else
        {
            _vae ??= new SafetensorsFile(_vaePath);
            raw = _vae.ReadPrefix(SdxlCheckpoint.VaePrefix);
            if (raw.Count == 0)
            {
                // A standalone VAE names its tensors from the model's root.
                // The trainer's loss head — the discriminator and the LPIPS
                // network some of them still carry — is not part of the
                // model and has no name on the diffusers side.
                raw = _vae.ReadPrefix("");
                foreach (string name in raw.Keys.Where(name =>
                             name.StartsWith("loss.", StringComparison.Ordinal)).ToArray())
                {
                    raw.Remove(name);
                }
            }
        }
        Dictionary<string, HostTensor> tensors = SdxlCheckpoint.Vae(raw);
        if (tensors.Count == 0)
        {
            throw new InvalidDataException(_vaePath is null
                ? "the checkpoint holds no VAE weights; is it an SDXL checkpoint?"
                : $"{_vaePath}: no VAE weights in the file");
        }
        return ConvertAll(tensors, HostDataType.Float32);
    }

    /// <summary>One CLIP tower's tensors, at whatever width the checkpoint
    /// stored them. The towers run on the CPU and widen tensor by tensor as they
    /// build their own float arrays, so widening here as well would mean holding
    /// a whole extra float32 copy of the tower — 3.05 GiB for the pair — that
    /// nothing ever reads.</summary>
    public Dictionary<string, HostTensor> LoadTextEncoder(string name,
        IReadOnlyList<LoraSpec>? loras = null)
    {
        Dictionary<string, HostTensor> tower =
            _checkpoint.ReadPrefix(SdxlCheckpoint.TowerPrefix(name));
        if (tower.Count == 0)
        {
            throw new InvalidDataException(
                $"the checkpoint holds no {name} weights; is it an SDXL checkpoint?");
        }
        bool second = name == "text_encoder_2";
        Dictionary<string, HostTensor> renamed = second
            ? SdxlCheckpoint.ClipG(tower)
            : SdxlCheckpoint.ClipL(tower);
        // A LoRA names both towers' layers the HF way, which is what the
        // renaming just produced — the fused OpenCLIP projections included,
        // which it has split into the three the LoRA has pairs for.
        if (loras is { Count: > 0 })
        {
            LoraMerge.Apply(renamed, SdxlLoras.Layout,
                second ? SdxlLoras.TextEncoder2 : SdxlLoras.TextEncoder, loras);
        }
        return renamed;
    }

    /// <summary>The UNet at half precision, which is what SDXL runs at
    /// everywhere. Finished lazily, weight by weight as the builder reads
    /// them — see <see cref="LazyWeights"/> — so the LoRA merge and any
    /// bfloat16 conversion never hold more than one graph's worth of copies;
    /// the LoRA files are matched against the model here, though, so a wrong
    /// one is refused before the build starts.</summary>
    public LazyWeights LoadUnet(IReadOnlyList<LoraSpec>? loras = null)
    {
        Dictionary<string, HostTensor> raw = _checkpoint.ReadPrefix(SdxlCheckpoint.UnetPrefix);
        Dictionary<string, HostTensor> tensors = SdxlCheckpoint.Unet(raw);
        if (tensors.Count == 0)
        {
            throw new InvalidDataException(
                "the checkpoint holds no UNet weights; is it an SDXL checkpoint?");
        }
        LoraPatchSet? patches = null;
        if (loras is { Count: > 0 })
        {
            // Kohya names the UNet's layers the LDM way and PEFT the diffusers
            // way, so every weight is matched under both.
            var ldmNames = new Dictionary<string, string>(raw.Count);
            foreach (string name in raw.Keys)
            {
                ldmNames[SdxlCheckpoint.UnetKey(name)] = name;
            }
            patches = LoraPatchSet.Prepare(tensors, SdxlLoras.Layout, SdxlLoras.Unet, loras,
                key => ldmNames.TryGetValue(key, out string? ldm)
                    ? new[] { ldm }
                    : Array.Empty<string>());
        }
        return new LazyWeights(tensors, patches, HostDataType.Float16);
    }

    /// <summary>Unmaps the checkpoint, and the VAE file if one was opened.
    /// Every tensor handed out points into those mappings, so nothing may
    /// touch them after this.</summary>
    public void Dispose()
    {
        _vae?.Dispose();
        _checkpoint.Dispose();
    }

    private static Dictionary<string, HostTensor> ConvertAll(
        Dictionary<string, HostTensor> tensors, HostDataType dataType)
    {
        var converted = new Dictionary<string, HostTensor>(tensors.Count);
        foreach ((string name, HostTensor tensor) in tensors)
        {
            converted[name] = tensor.ConvertTo(dataType);
        }
        return converted;
    }
}
