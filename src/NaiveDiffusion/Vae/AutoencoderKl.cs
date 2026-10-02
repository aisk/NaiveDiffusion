using NaiveDiffusion.Dml;
using NaiveDiffusion.Graph;
using NaiveDiffusion.Tensors;
using Vortice.DirectML;

namespace NaiveDiffusion.Vae;

/// <summary>The variational autoencoder of the latent-diffusion family — the
/// one diffusers calls AutoencoderKL — built as a DirectML graph, under
/// diffusers' weight names. The decoder turns a latent into a [1, 3, H, W]
/// image and the encoder turns an image into the latent mean; the widths and
/// the latent's shape come from a <see cref="VaeConfig"/>.</summary>
public static class AutoencoderKl
{
    /// <summary>The VAE's ResnetBlock2D: two norm/SiLU/conv pairs plus a residual.</summary>
    private static DmlExpression ResnetBlock(ModelBuilder model, DmlExpression x,
        IReadOnlyDictionary<string, HostTensor> parameters, VaeConfig config, string prefix)
    {
        DmlExpression h = Layers.Conv2d(model,
            Layers.Silu(Layers.GroupNorm(model, x, parameters[$"{prefix}.norm1.weight"],
                parameters[$"{prefix}.norm1.bias"], config.NormGroups, config.NormEpsilon)),
            parameters[$"{prefix}.conv1.weight"], parameters[$"{prefix}.conv1.bias"]);
        h = Layers.Conv2d(model,
            Layers.Silu(Layers.GroupNorm(model, h, parameters[$"{prefix}.norm2.weight"],
                parameters[$"{prefix}.norm2.bias"], config.NormGroups, config.NormEpsilon)),
            parameters[$"{prefix}.conv2.weight"], parameters[$"{prefix}.conv2.bias"]);

        // The shortcut is a 1x1 convolution only when the block changes width.
        if (parameters.ContainsKey($"{prefix}.conv_shortcut.weight"))
        {
            x = Layers.Conv2d(model, x, parameters[$"{prefix}.conv_shortcut.weight"],
                parameters[$"{prefix}.conv_shortcut.bias"], padding: 0);
        }
        return x + h;
    }

    /// <summary>Single-head self-attention over every pixel, with a residual.
    /// The projections run in token layout, so the whole block is four matrix
    /// multiplies and a softmax; the only reshapes are stride tricks.</summary>
    private static DmlExpression AttentionBlock(ModelBuilder model, DmlExpression x,
        IReadOnlyDictionary<string, HostTensor> parameters, VaeConfig config, string prefix)
    {
        uint[] shape = x.Shape;
        (uint channels, uint height, uint width) = (shape[1], shape[2], shape[3]);

        DmlExpression normalized = Layers.GroupNorm(model, x,
            parameters[$"{prefix}.group_norm.weight"], parameters[$"{prefix}.group_norm.bias"],
            config.NormGroups, config.NormEpsilon);
        DmlExpression tokens = Layers.ToTokens(normalized);

        DmlExpression query = Layers.Linear(model, tokens,
            parameters[$"{prefix}.to_q.weight"], parameters[$"{prefix}.to_q.bias"]);
        DmlExpression key = Layers.Linear(model, tokens,
            parameters[$"{prefix}.to_k.weight"], parameters[$"{prefix}.to_k.bias"]);
        DmlExpression value = Layers.Linear(model, tokens,
            parameters[$"{prefix}.to_v.weight"], parameters[$"{prefix}.to_v.bias"]);

        DmlExpression scores = DmlOps.Gemm(query, key,
            transB: MatrixTransform.Transpose, alpha: 1.0f / MathF.Sqrt(channels));
        DmlExpression attended = DmlOps.Gemm(
            DmlOps.ActivationSoftmax(scores, new[] { 3 }), value);

        DmlExpression projected = Layers.Linear(model, attended,
            parameters[$"{prefix}.to_out.0.weight"], parameters[$"{prefix}.to_out.0.bias"]);
        return Layers.ToImage(projected, height, width) + x;
    }

    /// <summary>Two resnets with self-attention between them, at the lowest resolution.</summary>
    private static DmlExpression MidBlock(ModelBuilder model, DmlExpression x,
        IReadOnlyDictionary<string, HostTensor> parameters, VaeConfig config, string prefix)
    {
        x = ResnetBlock(model, x, parameters, config, $"{prefix}.resnets.0");
        x = AttentionBlock(model, x, parameters, config, $"{prefix}.attentions.0");
        return ResnetBlock(model, x, parameters, config, $"{prefix}.resnets.1");
    }

    /// <summary>Compile a decoder that produces a height by width image.</summary>
    public static ModelBuilder Decoder(DmlDevice device,
        IReadOnlyDictionary<string, HostTensor> parameters, VaeConfig config, int height,
        int width)
    {
        LatentSpace latent = config.Latent;
        var model = new ModelBuilder(device);
        DmlExpression sample = model.Placeholder(
            new[] { 1, latent.Channels, height / latent.ScaleFactor, width / latent.ScaleFactor });

        DmlExpression h = Layers.Conv2d(model, sample, parameters["post_quant_conv.weight"],
            parameters["post_quant_conv.bias"], padding: 0);
        h = Layers.Conv2d(model, h, parameters["decoder.conv_in.weight"],
            parameters["decoder.conv_in.bias"]);
        h = MidBlock(model, h, parameters, config, "decoder.mid_block");

        for (int i = 0; i < config.BlockOutChannels.Length; i++)
        {
            string prefix = $"decoder.up_blocks.{i}";
            // One more resnet per block than the encoder has — layers_per_block + 1.
            for (int j = 0; j < config.LayersPerBlock + 1; j++)
            {
                h = ResnetBlock(model, h, parameters, config, $"{prefix}.resnets.{j}");
            }
            if (parameters.ContainsKey($"{prefix}.upsamplers.0.conv.weight"))
            {
                h = Layers.Conv2d(model, Layers.UpsampleNearest(h),
                    parameters[$"{prefix}.upsamplers.0.conv.weight"],
                    parameters[$"{prefix}.upsamplers.0.conv.bias"]);
            }
        }

        h = Layers.GroupNorm(model, h, parameters["decoder.conv_norm_out.weight"],
            parameters["decoder.conv_norm_out.bias"], config.NormGroups, config.NormEpsilon);
        DmlExpression image = Layers.Conv2d(model, Layers.Silu(h),
            parameters["decoder.conv_out.weight"], parameters["decoder.conv_out.bias"]);
        return model.Compile(new[] { image });
    }

    /// <summary>Compile an encoder that consumes a height by width image and
    /// produces the latent mean, which is what image-to-image pipelines use.</summary>
    public static ModelBuilder Encoder(DmlDevice device,
        IReadOnlyDictionary<string, HostTensor> parameters, VaeConfig config, int height,
        int width)
    {
        var model = new ModelBuilder(device);
        DmlExpression image = model.Placeholder(new[] { 1, 3, height, width });

        DmlExpression h = Layers.Conv2d(model, image, parameters["encoder.conv_in.weight"],
            parameters["encoder.conv_in.bias"]);

        for (int i = 0; i < config.BlockOutChannels.Length; i++)
        {
            string prefix = $"encoder.down_blocks.{i}";
            for (int j = 0; j < config.LayersPerBlock; j++)
            {
                h = ResnetBlock(model, h, parameters, config, $"{prefix}.resnets.{j}");
            }
            if (parameters.ContainsKey($"{prefix}.downsamplers.0.conv.weight"))
            {
                // Downsample2D pads the bottom and right edges only, then
                // strides by two with no padding of its own.
                h = Layers.Conv2d(model, h, parameters[$"{prefix}.downsamplers.0.conv.weight"],
                    parameters[$"{prefix}.downsamplers.0.conv.bias"],
                    stride: 2, padding: 0, endPadding: 1);
            }
        }

        h = MidBlock(model, h, parameters, config, "encoder.mid_block");
        h = Layers.GroupNorm(model, h, parameters["encoder.conv_norm_out.weight"],
            parameters["encoder.conv_norm_out.bias"], config.NormGroups, config.NormEpsilon);
        h = Layers.Conv2d(model, Layers.Silu(h), parameters["encoder.conv_out.weight"],
            parameters["encoder.conv_out.bias"]);
        DmlExpression moments = Layers.Conv2d(model, h, parameters["quant_conv.weight"],
            parameters["quant_conv.bias"], padding: 0);

        uint[] shape = moments.Shape;
        DmlExpression mean = DmlOps.Slice(moments,
            offsets: new[] { 0, 0, 0, 0 },
            sizes: new[] { 1, config.Latent.Channels, (int)shape[2], (int)shape[3] },
            strides: new[] { 1, 1, 1, 1 });
        return model.Compile(new[] { mean });
    }
}
