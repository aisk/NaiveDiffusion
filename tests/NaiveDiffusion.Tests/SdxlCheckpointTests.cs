using NaiveDiffusion.Models.Sdxl;
using NaiveDiffusion.Tensors;

namespace NaiveDiffusion.Tests;

/// <summary>The LDM names a single-file checkpoint stores its tensors under
/// into the diffusers names the graphs read, the three transformations
/// that are more than a rename, and the rule that a name the table does not
/// know is an error rather than a tensor quietly dropped.</summary>
public class SdxlCheckpointTests
{
    private static HostTensor One() => HostTensor.FromFloats(new[] { 1f }, 1);

    [TestCase("input_blocks.0.0.weight", "conv_in.weight")]
    [TestCase("time_embed.0.bias", "time_embedding.linear_1.bias")]
    [TestCase("label_emb.0.2.weight", "add_embedding.linear_2.weight")]
    [TestCase("out.2.weight", "conv_out.weight")]
    [TestCase("input_blocks.1.0.in_layers.2.weight", "down_blocks.0.resnets.0.conv1.weight")]
    [TestCase("input_blocks.2.0.out_layers.3.weight", "down_blocks.0.resnets.1.conv2.weight")]
    [TestCase("input_blocks.3.0.op.weight", "down_blocks.0.downsamplers.0.conv.weight")]
    [TestCase("input_blocks.4.1.transformer_blocks.0.attn1.to_k.weight", "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.weight")]
    [TestCase("input_blocks.4.1.proj_in.weight", "down_blocks.1.attentions.0.proj_in.weight")]
    [TestCase("input_blocks.4.0.skip_connection.weight", "down_blocks.1.resnets.0.conv_shortcut.weight")]
    [TestCase("middle_block.0.in_layers.0.weight", "mid_block.resnets.0.norm1.weight")]
    [TestCase("middle_block.1.norm.weight", "mid_block.attentions.0.norm.weight")]
    [TestCase("middle_block.2.emb_layers.1.weight", "mid_block.resnets.1.time_emb_proj.weight")]
    [TestCase("output_blocks.2.2.conv.weight", "up_blocks.0.upsamplers.0.conv.weight")]
    [TestCase("output_blocks.5.1.transformer_blocks.1.ff.net.0.proj.weight", "up_blocks.1.attentions.2.transformer_blocks.1.ff.net.0.proj.weight")]
    public void UnetNamesAreTranslatedToDiffusers(string ldm, string diffusers)
    {
        Assert.That(SdxlCheckpoint.UnetKey(ldm), Is.EqualTo(diffusers));
        Assert.That(SdxlCheckpoint.Unet(new Dictionary<string, HostTensor> { [ldm] = One() }).Keys,
            Is.EqualTo(new[] { diffusers }));
    }

    [Test]
    public void AnUnknownNameIsAnErrorNotADroppedTensor()
    {
        Assert.That(() => SdxlCheckpoint.UnetKey("input_blocks.4.1.something_new.weight"),
            Throws.InstanceOf<InvalidDataException>().With.Message.Contains("something_new"));
        Assert.That(() => SdxlCheckpoint.UnetKey("not_a_block.weight"),
            Throws.InstanceOf<InvalidDataException>());
        Assert.That(() => SdxlCheckpoint.Vae(new Dictionary<string, HostTensor> { ["decoder.novel.weight"] = One() }),
            Throws.InstanceOf<InvalidDataException>().With.Message.Contains("novel"));
        Assert.That(() => SdxlCheckpoint.ClipG(new Dictionary<string, HostTensor>
            {
                ["transformer.resblocks.0.attn.novel"] = One(),
            }), Throws.InstanceOf<InvalidDataException>().With.Message.Contains("novel"));
    }

    [Test]
    public void TheVaesBlocksAreRenumberedAndItsAttentionProjectionsBecomeLinear()
    {
        Dictionary<string, HostTensor> vae = SdxlCheckpoint.Vae(new Dictionary<string, HostTensor>
        {
            ["decoder.up.3.block.0.conv1.weight"] = One(),
            ["decoder.up.0.upsample.conv.weight"] = One(),
            ["encoder.down.1.block.1.nin_shortcut.weight"] = One(),
            ["decoder.mid.attn_1.q.weight"] = HostTensor.FromFloats(new float[4], 2, 2, 1, 1),
            ["decoder.mid.attn_1.proj_out.weight"] = HostTensor.FromFloats(new float[4], 2, 2, 1, 1),
            ["decoder.mid.block_2.norm1.weight"] = One(),
            ["decoder.norm_out.weight"] = One(),
            ["post_quant_conv.bias"] = One(),
        });
        Assert.That(vae.Keys, Is.EquivalentTo(new[]
        {
            // diffusers numbers the decoder's levels in the order it runs them.
            "decoder.up_blocks.0.resnets.0.conv1.weight",
            "decoder.up_blocks.3.upsamplers.0.conv.weight",
            "encoder.down_blocks.1.resnets.1.conv_shortcut.weight",
            "decoder.mid_block.attentions.0.to_q.weight",
            "decoder.mid_block.attentions.0.to_out.0.weight",
            "decoder.mid_block.resnets.1.norm1.weight",
            "decoder.conv_norm_out.weight",
            "post_quant_conv.bias",
        }));
        Assert.That(vae["decoder.mid_block.attentions.0.to_q.weight"].Shape, Is.EqualTo(new[] { 2, 2 }),
            "a 1x1 convolution read as a linear layer");
    }

    [Test]
    public void OpenClipsFusedProjectionIsSplitAndItsTextProjectionTransposed()
    {
        // in_proj is [3 × width, width]: query, key, value stacked.
        float[] fused = Enumerable.Range(0, 6 * 2).Select(i => (float)i).ToArray();
        Dictionary<string, HostTensor> clipG = SdxlCheckpoint.ClipG(new Dictionary<string, HostTensor>
        {
            ["transformer.resblocks.0.attn.in_proj_weight"] = HostTensor.FromFloats(fused, 6, 2),
            ["transformer.resblocks.0.ln_1.weight"] = One(),
            ["transformer.resblocks.0.mlp.c_fc.bias"] = One(),
            ["text_projection"] = HostTensor.FromFloats(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3),
            ["ln_final.bias"] = One(),
            ["logit_scale"] = One(),
            ["positional_embedding"] = One(),
        });
        Assert.That(clipG.Keys, Is.EquivalentTo(new[]
        {
            "text_model.encoder.layers.0.self_attn.q_proj.weight",
            "text_model.encoder.layers.0.self_attn.k_proj.weight",
            "text_model.encoder.layers.0.self_attn.v_proj.weight",
            "text_model.encoder.layers.0.layer_norm1.weight",
            "text_model.encoder.layers.0.mlp.fc1.bias",
            "text_projection.weight",
            "text_model.final_layer_norm.bias",
            "text_model.embeddings.position_embedding.weight",
        }), "logit_scale is the image tower's and is dropped");
        Assert.That(clipG["text_model.encoder.layers.0.self_attn.k_proj.weight"].ToFloats(),
            Is.EqualTo(new[] { 4f, 5f, 6f, 7f }), "the second third of the rows, as a view");
        Assert.That(clipG["text_projection.weight"].Shape, Is.EqualTo(new[] { 3, 2 }));
        Assert.That(clipG["text_projection.weight"].ToFloats(), Is.EqualTo(new[] { 1f, 4f, 2f, 5f, 3f, 6f }));
    }

    [Test]
    public void ClipLIsAlreadyNamedTheDiffusersWayExceptForItsPositionIds()
    {
        Dictionary<string, HostTensor> clipL = SdxlCheckpoint.ClipL(new Dictionary<string, HostTensor>
        {
            ["text_model.embeddings.position_ids"] = One(),
            ["text_model.encoder.layers.0.mlp.fc1.weight"] = One(),
        });
        Assert.That(clipL.Keys, Is.EqualTo(new[] { "text_model.encoder.layers.0.mlp.fc1.weight" }));
    }
}
