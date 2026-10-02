using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using SdxlNames = NaiveDiffusion.Models.Sdxl.SdxlCheckpoint;

namespace NaiveDiffusion.Tests.Fixtures;

/// <summary>A folder under the temp directory that goes away with the test.</summary>
public sealed class TempFolder : IDisposable
{
    public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(),
        "NaiveDiffusion.Tests", Guid.NewGuid().ToString("N"));

    public TempFolder()
    {
        Directory.CreateDirectory(Path);
    }

    public string File(string name) => System.IO.Path.Combine(Path, name);

    public void Dispose()
    {
        try
        {
            Directory.Delete(Path, recursive: true);
        }
        catch (IOException)
        {
            // A mapping still closing; the temp directory is cleaned up anyway.
        }
    }
}

/// <summary>One tensor of a hand-written safetensors file: a name, a shape
/// (one element unless the test needs to read it), a dtype, and the values
/// when a test reads them rather than only the header.</summary>
public sealed record Tensor(string Name, int[]? Shape = null, string DataType = "F16",
    float[]? Values = null);

/// <summary>Writes safetensors files small enough to hand-write: the 8-byte
/// little-endian header length, the JSON table, then the data laid end to
/// end. The inspectors read the header alone, so most tests only get the
/// key names and the one shape that matters right; the loaders read the
/// data, so a tensor can carry values as well.</summary>
public sealed class FakeSafetensors
{
    private readonly TempFolder _folder;

    public FakeSafetensors(TempFolder folder)
    {
        _folder = folder;
    }

    public string Write(string name, IEnumerable<Tensor> tensors,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        var header = new Dictionary<string, object>();
        if (metadata is not null)
        {
            header["__metadata__"] = metadata;
        }
        // The tensors with values are laid out first and written; the rest
        // are zeros, which the file gets by being extended past them rather
        // than by writing them — the text encoders' vocabularies are
        // gigabytes of zeros, and writing them would make every test that
        // sets them up take seconds.
        var body = new MemoryStream();
        long end = 0;
        foreach (Tensor tensor in tensors.OrderBy(tensor => tensor.Values is null))
        {
            int[] shape = tensor.Shape ?? new[] { 1 };
            long elements = shape.Aggregate(1L, (a, b) => a * b);
            int width = tensor.DataType switch
            {
                "F32" => 4,
                "F8_E4M3" or "F8_E5M2" => 1,
                _ => 2,
            };
            long start = end;
            end += elements * width;
            if (tensor.Values is not null)
            {
                body.Write(Encode(tensor, elements, width));
            }
            header[tensor.Name] = new Dictionary<string, object>
            {
                ["dtype"] = tensor.DataType,
                ["shape"] = shape,
                ["data_offsets"] = new[] { start, end },
            };
        }
        byte[] json = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        string path = _folder.File(name + ".safetensors");
        using var file = File.Create(path);
        file.Write(BitConverter.GetBytes((long)json.Length));
        file.Write(json);
        body.Position = 0;
        body.CopyTo(file);
        file.SetLength(8 + json.Length + end);
        return path;
    }

    private static byte[] Encode(Tensor tensor, long elements, int width)
    {
        var bytes = new byte[elements * width];
        if (tensor.Values!.Length != elements)
        {
            throw new ArgumentException($"{tensor.Name}: {tensor.Values.Length} values for {elements} elements");
        }
        switch (tensor.DataType)
        {
            case "F32":
                MemoryMarshal.AsBytes<float>(tensor.Values).CopyTo(bytes);
                break;
            case "F16":
                for (int i = 0; i < elements; i++)
                {
                    BitConverter.TryWriteBytes(bytes.AsSpan(i * 2), (Half)tensor.Values[i]);
                }
                break;
            case "BF16":
                for (int i = 0; i < elements; i++)
                {
                    uint bits = BitConverter.SingleToUInt32Bits(tensor.Values[i]);
                    BitConverter.TryWriteBytes(bytes.AsSpan(i * 2), (ushort)(bits >> 16));
                }
                break;
            default:
                throw new ArgumentException($"{tensor.Name}: no values for {tensor.DataType}");
        }
        return bytes;
    }

    // --- The files the families recognize, header only ----------------------

    /// <summary>An SDXL base checkpoint with all four models inside.</summary>
    public static Tensor[] SdxlCheckpoint() => new[]
    {
        new Tensor(SdxlNames.UnetPrefix + "label_emb.0.0.weight"),
        new Tensor(SdxlNames.UnetPrefix + "input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight",
            new[] { 320, 2048 }),
        new Tensor(SdxlNames.VaePrefix + "decoder.conv_out.weight"),
        new Tensor(SdxlNames.VaePrefix + "decoder.conv_in.weight", new[] { 512, 4, 3, 3 }),
        new Tensor(SdxlNames.ClipLPrefix + "text_model.embeddings.token_embedding.weight"),
        new Tensor(SdxlNames.ClipGPrefix + "transformer.resblocks.0.attn.in_proj_weight"),
    };

    /// <summary>An Anima transformer under <paramref name="prefix"/>, with
    /// its adapter, two blocks deep.</summary>
    public static Tensor[] AnimaDit(string prefix = "net.") => new[]
    {
        new Tensor(prefix + "x_embedder.proj.1.weight", new[] { 2048, 68 }, "BF16"),
        new Tensor(prefix + "llm_adapter.blocks.0.cross_attn.q_proj.weight", new[] { 1024, 1024 }, "BF16"),
        new Tensor(prefix + "blocks.0.mlp.layer1.weight", new[] { 8192, 2048 }, "BF16"),
        new Tensor(prefix + "blocks.1.mlp.layer1.weight", new[] { 8192, 2048 }, "BF16"),
        new Tensor(prefix + "blocks.0.adaln_modulation_self_attn.1.weight", new[] { 256, 2048 }, "BF16"),
    };

    /// <summary>The Qwen3-0.6B text encoder, 1024 wide.</summary>
    public static Tensor[] Qwen3TextEncoder(int width = 1024) => new[]
    {
        new Tensor("model.embed_tokens.weight", new[] { 151936, width }, "BF16"),
        new Tensor("model.layers.0.self_attn.q_norm.weight", new[] { 128 }, "BF16"),
    };

    /// <summary>The Wan 2.1 VAE, sixteen latent channels.</summary>
    public static Tensor[] WanVae(int channels = 16) => new[]
    {
        new Tensor("decoder.conv1.weight", new[] { 384, channels, 3, 3, 3 }, "BF16"),
        new Tensor("decoder.head.2.weight", new[] { 3, 96, 3, 3, 3 }, "BF16"),
        new Tensor("encoder.conv1.weight", new[] { 96, 3, 3, 3, 3 }, "BF16"),
    };

    /// <summary>A Qwen-Image 2.1 transformer under <paramref name="prefix"/>,
    /// two blocks deep, with the feed-forward fused as Comfy-Org saves it.</summary>
    public static Tensor[] QwenImageDit(string prefix = "") => new[]
    {
        new Tensor(prefix + "img_in.weight", new[] { 4096, 64 }, "BF16"),
        new Tensor(prefix + "txt_in.text_norm.weight", new[] { 4096 }, "BF16"),
        new Tensor(prefix + "modulation.1.weight", new[] { 16384, 4096 }, "BF16"),
        new Tensor(prefix + "proj_out.weight", new[] { 64, 4096 }, "BF16"),
        new Tensor(prefix + "transformer_blocks.0.attn.norm_q.weight", new[] { 128 }, "BF16"),
        new Tensor(prefix + "transformer_blocks.0.attn.to_q.weight", new[] { 4096, 4096 }, "BF16"),
        new Tensor(prefix + "transformer_blocks.1.attn.to_q.weight", new[] { 4096, 4096 }, "BF16"),
        new Tensor(prefix + "transformer_blocks.0.img_mlp.gate_up.weight", new[] { 24576, 4096 }, "BF16"),
    };

    /// <summary>Qwen3-VL-8B as Comfy-Org repacks it: the language model
    /// 4096 wide with the vision tower beside it.</summary>
    public static Tensor[] Qwen3VlTextEncoder(int width = 4096) => new[]
    {
        new Tensor("model.embed_tokens.weight", new[] { 151936, width }, "BF16"),
        new Tensor("model.layers.0.self_attn.q_norm.weight", new[] { 128 }, "BF16"),
        new Tensor("model.visual.patch_embed.proj.weight", new[] { 1152, 3, 2, 16, 16 }, "BF16"),
    };

    /// <summary>The Qwen-Image 2.1 VAE: Wan 2.2's layout, sixty-four latent
    /// channels, four image channels, a temporal kernel of one.</summary>
    public static Tensor[] Wan22Vae(int channels = 64) => new[]
    {
        new Tensor("decoder.conv1.weight", new[] { 1152, channels, 1, 3, 3 }, "BF16"),
        new Tensor("decoder.head.2.weight", new[] { 4, 144, 1, 3, 3 }, "BF16"),
        new Tensor("encoder.conv1.weight", new[] { 96, 4, 1, 3, 3 }, "BF16"),
    };

    /// <summary>An Anima LoRA as ai-toolkit writes it: PEFT pairs under
    /// ComfyUI's generic <c>diffusion_model.</c>, one transformer layer.</summary>
    public static Tensor[] AnimaLora() => new[]
    {
        new Tensor("diffusion_model.blocks.0.mlp.layer1.lora_A.weight", new[] { 4, 2048 }, "BF16"),
        new Tensor("diffusion_model.blocks.0.mlp.layer1.lora_B.weight", new[] { 8192, 4 }, "BF16"),
    };

    /// <summary>A kohya LoRA over one UNet layer and one CLIP-L layer.</summary>
    public static Tensor[] SdxlLora() => new[]
    {
        new Tensor("lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_k.lora_down.weight", new[] { 4, 320 }),
        new Tensor("lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_k.lora_up.weight", new[] { 320, 4 }),
        new Tensor("lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_k.alpha"),
        new Tensor("lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight", new[] { 4, 768 }),
        new Tensor("lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_up.weight", new[] { 3072, 4 }),
    };
}
