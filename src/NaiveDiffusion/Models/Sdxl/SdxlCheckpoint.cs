using System.Text.RegularExpressions;
using NaiveDiffusion.Tensors;

namespace NaiveDiffusion.Models.Sdxl;

/// <summary>SDXL weights under LDM's key names, read as the names the graphs
/// want. The checkpoints ComfyUI and A1111 load are a single file holding all
/// three models under Stability's latent-diffusion names; everything here is
/// renaming — dictionary in, dictionary out.
///
/// Three of the renames are more than a name: OpenCLIP keeps one fused QKV
/// matrix per layer where HF keeps three; OpenCLIP's text projection is stored
/// the way it is used, x @ W, where HF stores the transpose; and the VAE's
/// attention projections are 1x1 convolutions here and plain linears in
/// diffusers, so they lose two trailing axes.
///
/// Anything unrecognized throws rather than being dropped: an unknown key means
/// the table is wrong or the checkpoint is not SDXL, and both are worth hearing
/// about before a graph is built out of half the weights.</summary>
public static partial class SdxlCheckpoint
{
    /// <summary>The single-file prefix each model lives under. Shared with
    /// <see cref="CheckpointInspector"/>, so that what the inspector says is
    /// present is exactly what the loader will go looking for.</summary>
    public const string UnetPrefix = "model.diffusion_model.";

    /// <inheritdoc cref="UnetPrefix"/>
    public const string VaePrefix = "first_stage_model.";

    // --- The UNet ----------------------------------------------------------

    // ResBlock keeps its layers under the Sequentials it built them in.
    private static readonly Dictionary<string, string> ResnetNames = new()
    {
        ["in_layers.0"] = "norm1",
        ["in_layers.2"] = "conv1",
        ["emb_layers.1"] = "time_emb_proj",
        ["out_layers.0"] = "norm2",
        ["out_layers.3"] = "conv2",
        ["skip_connection"] = "conv_shortcut",
    };

    private static readonly Dictionary<string, string> UnetTopNames = new()
    {
        ["input_blocks.0.0"] = "conv_in",
        ["time_embed.0"] = "time_embedding.linear_1",
        ["time_embed.2"] = "time_embedding.linear_2",
        ["label_emb.0.0"] = "add_embedding.linear_1",
        ["label_emb.0.2"] = "add_embedding.linear_2",
        ["out.0"] = "conv_norm_out",
        ["out.2"] = "conv_out",
    };

    [GeneratedRegex(@"^(input_blocks|output_blocks|middle_block)\.(\d+)\.(?:\d+\.)?(.+)$")]
    private static partial Regex UnetBlockPattern();

    /// <summary>The diffusers name for one UNet tensor. LDM numbers the down and
    /// up paths as one flat list of blocks, so the resolution level and the
    /// layer within it are divided back out; which kind of block a tensor
    /// belongs to is decided by its own name, because the sub-index that would
    /// say so shifts depending on whether the level has attention at all.
    /// Public because a LoRA names the UNet's layers the LDM way, so the merge
    /// has to know both names for every tensor.</summary>
    public static string UnetKey(string name)
    {
        (string path, string param) = SplitLast(name);

        if (UnetTopNames.TryGetValue(path, out string? top))
        {
            return $"{top}.{param}";
        }

        Match match = UnetBlockPattern().Match(path);
        if (!match.Success)
        {
            throw new InvalidDataException($"unrecognized UNet key: {name}");
        }
        string where = match.Groups[1].Value;
        int index = int.Parse(match.Groups[2].Value);
        string rest = match.Groups[3].Value;

        if (where == "middle_block")
        {
            // Resnet, transformer, resnet.
            if (index == 1)
            {
                return $"mid_block.attentions.0.{rest}.{param}";
            }
            return $"mid_block.resnets.{index / 2}.{ResnetNames[rest]}.{param}";
        }

        // The down path starts at input_blocks.1, because block 0 is conv_in.
        bool down = where == "input_blocks";
        int flat = down ? index - 1 : index;
        int block = flat / 3, layer = flat % 3;

        if (rest == "op")
        {
            return $"down_blocks.{block}.downsamplers.0.conv.{param}";
        }
        if (rest == "conv")
        {
            return $"up_blocks.{block}.upsamplers.0.conv.{param}";
        }

        string side = down ? "down" : "up";
        if (ResnetNames.TryGetValue(rest, out string? resnet))
        {
            return $"{side}_blocks.{block}.resnets.{layer}.{resnet}.{param}";
        }
        if (rest is "norm" or "proj_in" or "proj_out" || rest.StartsWith("transformer_blocks.", StringComparison.Ordinal))
        {
            return $"{side}_blocks.{block}.attentions.{layer}.{rest}.{param}";
        }
        throw new InvalidDataException($"unrecognized UNet key: {name}");
    }

    /// <summary>model.diffusion_model.* as diffusers.UNet2DConditionModel.</summary>
    public static Dictionary<string, HostTensor> Unet(IReadOnlyDictionary<string, HostTensor> tensors)
    {
        var converted = new Dictionary<string, HostTensor>(tensors.Count);
        foreach ((string name, HostTensor tensor) in tensors)
        {
            converted[UnetKey(name)] = tensor;
        }
        return converted;
    }

    // --- The VAE -----------------------------------------------------------

    // The VAE has four resolutions, and the decoder's are stored lowest-first.
    private const int VaeLevels = 4;

    private static readonly Dictionary<string, string> VaeAttentionNames = new()
    {
        ["norm"] = "group_norm",
        ["q"] = "to_q",
        ["k"] = "to_k",
        ["v"] = "to_v",
        ["proj_out"] = "to_out.0",
    };

    [GeneratedRegex(@"^(encoder|decoder)\.(down|up)\.(\d+)\.(.+)$")]
    private static partial Regex VaeBlockPattern();

    [GeneratedRegex(@"^(encoder|decoder)\.mid\.(block_1|block_2|attn_1)\.(.+)$")]
    private static partial Regex VaeMidPattern();

    [GeneratedRegex(@"^(encoder|decoder)\.conv_(in|out)$")]
    private static partial Regex VaeConvPattern();

    [GeneratedRegex(@"^(encoder|decoder)\.norm_out$")]
    private static partial Regex VaeNormPattern();

    private static string VaeKey(string name)
    {
        (string path, string param) = SplitLast(name);

        Match match = VaeBlockPattern().Match(path);
        if (match.Success)
        {
            string half = match.Groups[1].Value;
            string direction = match.Groups[2].Value;
            int index = int.Parse(match.Groups[3].Value);
            string rest = match.Groups[4].Value;
            if (direction == "up")
            {
                // diffusers numbers the decoder's blocks in the order it runs them.
                index = VaeLevels - 1 - index;
            }
            string prefix = $"{half}.{direction}_blocks.{index}";

            if (rest is "downsample.conv" or "upsample.conv")
            {
                return $"{prefix}.{direction}samplers.0.conv.{param}";
            }
            string[] parts = rest.Split('.', 3);
            if (parts.Length != 3 || parts[0] != "block")
            {
                throw new InvalidDataException($"unrecognized VAE key: {name}");
            }
            string inner = parts[2] == "nin_shortcut" ? "conv_shortcut" : parts[2];
            return $"{prefix}.resnets.{int.Parse(parts[1])}.{inner}.{param}";
        }

        match = VaeMidPattern().Match(path);
        if (match.Success)
        {
            string half = match.Groups[1].Value;
            string which = match.Groups[2].Value;
            string rest = match.Groups[3].Value;
            if (which == "attn_1")
            {
                return $"{half}.mid_block.attentions.0.{VaeAttentionNames[rest]}.{param}";
            }
            return $"{half}.mid_block.resnets.{which[^1] - '1'}.{rest}.{param}";
        }

        if (path is "quant_conv" or "post_quant_conv" || VaeConvPattern().IsMatch(path))
        {
            return $"{path}.{param}";
        }
        match = VaeNormPattern().Match(path);
        if (match.Success)
        {
            return $"{match.Groups[1].Value}.conv_norm_out.{param}";
        }
        throw new InvalidDataException($"unrecognized VAE key: {name}");
    }

    /// <summary>first_stage_model.* as diffusers.AutoencoderKL.</summary>
    public static Dictionary<string, HostTensor> Vae(IReadOnlyDictionary<string, HostTensor> tensors)
    {
        var converted = new Dictionary<string, HostTensor>(tensors.Count);
        foreach ((string name, HostTensor tensor) in tensors)
        {
            string renamed = VaeKey(name);
            HostTensor value = tensor;
            if (renamed.Contains(".attentions.", StringComparison.Ordinal) && tensor.Shape.Length == 4)
            {
                value = tensor.Reshape(tensor.Shape[0], tensor.Shape[1]);
            }
            converted[renamed] = value;
        }
        return converted;
    }

    // --- The two text encoders ---------------------------------------------

    /// <summary>The single-file prefix each tower lives under. Embedder 0 is a
    /// transformers CLIP saved whole; embedder 1 is OpenCLIP's own module.</summary>
    public static string TowerPrefix(string name) => name switch
    {
        "text_encoder" => ClipLPrefix,
        "text_encoder_2" => ClipGPrefix,
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    /// <inheritdoc cref="UnetPrefix"/>
    public const string ClipLPrefix = "conditioner.embedders.0.transformer.";

    /// <inheritdoc cref="UnetPrefix"/>
    public const string ClipGPrefix = "conditioner.embedders.1.model.";

    private static readonly Dictionary<string, string> ClipGTopNames = new()
    {
        ["token_embedding.weight"] = "text_model.embeddings.token_embedding.weight",
        ["positional_embedding"] = "text_model.embeddings.position_embedding.weight",
        ["ln_final.weight"] = "text_model.final_layer_norm.weight",
        ["ln_final.bias"] = "text_model.final_layer_norm.bias",
    };

    private static readonly Dictionary<string, string> ClipGLayerNames = new()
    {
        ["ln_1"] = "layer_norm1",
        ["ln_2"] = "layer_norm2",
        ["mlp.c_fc"] = "mlp.fc1",
        ["mlp.c_proj"] = "mlp.fc2",
        ["attn.out_proj"] = "self_attn.out_proj",
    };

    [GeneratedRegex(@"^transformer\.resblocks\.(\d+)\.(.+)$")]
    private static partial Regex ClipGBlockPattern();

    /// <summary>CLIP ViT-L, which the checkpoint already stores under HF's own names.</summary>
    public static Dictionary<string, HostTensor> ClipL(IReadOnlyDictionary<string, HostTensor> tensors)
    {
        return tensors
            .Where(entry => !entry.Key.EndsWith("position_ids", StringComparison.Ordinal))
            .ToDictionary(entry => entry.Key, entry => entry.Value);
    }

    /// <summary>OpenCLIP ViT-bigG as transformers.CLIPTextModelWithProjection.</summary>
    public static Dictionary<string, HostTensor> ClipG(IReadOnlyDictionary<string, HostTensor> tensors)
    {
        var converted = new Dictionary<string, HostTensor>(tensors.Count);
        foreach ((string name, HostTensor tensor) in tensors)
        {
            if (ClipGTopNames.TryGetValue(name, out string? top))
            {
                converted[top] = tensor;
                continue;
            }
            if (name == "text_projection")
            {
                // Stored the way it is used, x @ W; nn.Linear holds the transpose.
                converted["text_projection.weight"] = Transpose2d(tensor);
                continue;
            }
            if (name == "logit_scale")
            {
                continue;  // Only the image tower it was trained against uses this.
            }

            Match match = ClipGBlockPattern().Match(name);
            if (!match.Success)
            {
                throw new InvalidDataException($"unrecognized OpenCLIP key: {name}");
            }
            string layer = $"text_model.encoder.layers.{match.Groups[1].Value}";
            string rest = match.Groups[2].Value;

            if (rest is "attn.in_proj_weight" or "attn.in_proj_bias")
            {
                // One fused matrix where HF keeps three, stacked query, key, value.
                string param = rest.EndsWith("weight", StringComparison.Ordinal) ? "weight" : "bias";
                int width = tensor.Shape[0] / 3;
                string[] projections = { "q", "k", "v" };
                for (int i = 0; i < 3; i++)
                {
                    converted[$"{layer}.self_attn.{projections[i]}_proj.{param}"] =
                        tensor.SliceRows(i * width, width);
                }
                continue;
            }

            (string path, string parameter) = SplitLast(rest);
            if (ClipGLayerNames.TryGetValue(path, out string? renamed))
            {
                converted[$"{layer}.{renamed}.{parameter}"] = tensor;
                continue;
            }
            throw new InvalidDataException($"unrecognized OpenCLIP key: {name}");
        }
        return converted;
    }

    // --- Tensor helpers ----------------------------------------------------

    private static (string Path, string Param) SplitLast(string name)
    {
        int dot = name.LastIndexOf('.');
        return dot < 0 ? ("", name) : (name[..dot], name[(dot + 1)..]);
    }

    private static HostTensor Transpose2d(HostTensor tensor)
    {
        if (tensor.Shape.Length != 2)
        {
            throw new ArgumentException("only a 2-D tensor can be transposed here");
        }
        int rows = tensor.Shape[0], columns = tensor.Shape[1];
        int itemSize = HostTensor.BytesPerElement(tensor.DataType);
        ReadOnlySpan<byte> source = tensor.Data.Span;
        var data = new byte[tensor.Data.Length];
        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
            {
                source.Slice((row * columns + column) * itemSize, itemSize)
                    .CopyTo(data.AsSpan((column * rows + row) * itemSize, itemSize));
            }
        }
        return new HostTensor(tensor.DataType, new[] { columns, rows }, data);
    }
}
