using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Tests.Fixtures;
using NaiveDiffusion.Weights;
using NaiveDiffusion.Models;
using NaiveDiffusion.Models.Sdxl;
using static NaiveDiffusion.Models.CheckpointInspector;

namespace NaiveDiffusion.Tests;

/// <summary>The verdict on a file from its header alone, so every kind of
/// file the inspector can be pointed at is a hand-written safetensors of a few
/// hundred bytes: the key names and the one shape that matter, one element
/// each.</summary>
public class CheckpointInspectorTests
{
    private TempFolder _folder = null!;
    private FakeSafetensors _files = null!;

    [SetUp]
    public void MakeFolder()
    {
        _folder = new TempFolder();
        _files = new FakeSafetensors(_folder);
    }

    [TearDown]
    public void DropFolder() => _folder.Dispose();

    private string Write(string name, IEnumerable<Tensor> tensors,
        IReadOnlyDictionary<string, string>? metadata = null) => _files.Write(name, tensors, metadata);

    private const string Unet = SdxlCheckpoint.UnetPrefix;
    private const string LabelEmb = Unet + "label_emb.0.0.weight";
    private const string CrossAttention = Unet + "input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight";
    private const string Vae = SdxlCheckpoint.VaePrefix + "decoder.conv_out.weight";
    private const string ClipL = SdxlCheckpoint.ClipLPrefix + "text_model.embeddings.token_embedding.weight";
    private const string ClipG = SdxlCheckpoint.ClipGPrefix + "transformer.resblocks.0.attn.in_proj_weight";

    private static readonly Tensor[] Sdxl =
    {
        new(LabelEmb), new(CrossAttention, Shape: new[] { 320, 2048 }), new(Vae), new(ClipL), new(ClipG),
    };

    [Test]
    public void ASdxlBaseWithAllFourModelsRuns()
    {
        CheckpointReport report = Inspect(Write("sdxl", Sdxl));
        Assert.That(report.Kind, Is.EqualTo(CheckpointKind.SdxlBase));
        Assert.That(report.Present, Is.EqualTo(CheckpointParts.All));
        Assert.That(report.VelocityPrediction, Is.False);
        Assert.That(report.UnsupportedDataType, Is.Null);
        Assert.That(report.CanRun);
    }

    [Test]
    public void AMissingVaeIsNamed()
    {
        CheckpointReport report = Inspect(Write("novae", Sdxl.Where(t => t.Name != Vae)));
        Assert.That(report.Kind, Is.EqualTo(CheckpointKind.SdxlBase));
        Assert.That(report.Missing, Is.EqualTo(CheckpointParts.Vae));
        Assert.That(report.CanRun, Is.False);
        Assert.That(report.Carries(CheckpointParts.Unet | CheckpointParts.TextEncoder));
        Assert.That(Explain(report), Does.Contain("missing the VAE"));
    }

    [Test]
    public void TheRefinerIsToldApartByTheTowerSlot()
    {
        CheckpointReport report = Inspect(Write("refiner", new Tensor[]
        {
            new(LabelEmb), new(Vae), new("conditioner.embedders.0.model.transformer.resblocks.0.attn.in_proj_weight"),
        }));
        Assert.That(report.Kind, Is.EqualTo(CheckpointKind.SdxlRefiner));
        Assert.That(Explain(report), Does.Contain("refiner"));
    }

    [TestCase(768, CheckpointKind.StableDiffusion1)]
    [TestCase(1024, CheckpointKind.StableDiffusion2)]
    public void EarlierVersionsByCrossAttentionWidth(int width, CheckpointKind kind)
    {
        CheckpointReport report = Inspect(Write("sd", new Tensor[]
        {
            new(CrossAttention, Shape: new[] { 320, width }), new(Vae),
            new("cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"),
        }));
        Assert.That(report.Kind, Is.EqualTo(kind));
        Assert.That(report.CanRun, Is.False);
    }

    [Test]
    public void OtherKindsOfFile()
    {
        Assert.That(Inspect(Write("lora", new Tensor[]
        {
            new("lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_k.lora_down.weight"),
            new("lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_k.lora_up.weight"),
        })).Kind, Is.EqualTo(CheckpointKind.Lora));
        Assert.That(Inspect(Write("peft", new Tensor[]
        {
            new("unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.lora_A.weight"),
        })).Kind, Is.EqualTo(CheckpointKind.Lora));
        Assert.That(Inspect(Write("controlnet", new Tensor[]
        {
            new("control_model.input_blocks.0.0.weight"), new("input_hint_block.0.weight"),
        })).Kind, Is.EqualTo(CheckpointKind.ControlNet));
        Assert.That(Inspect(Write("sd3", new Tensor[]
        {
            new(Unet + "joint_blocks.0.x_block.attn.qkv.weight"), new(Vae),
        })).Kind, Is.EqualTo(CheckpointKind.MmDit));
        Assert.That(Inspect(Write("vae", new Tensor[]
        {
            new("decoder.conv_out.weight"), new("encoder.conv_in.weight"),
        })).Kind, Is.EqualTo(CheckpointKind.VaeOnly));
        Assert.That(Inspect(Write("embedding", new Tensor[]
        {
            new("clip_l", Shape: new[] { 1, 768 }), new("clip_g", Shape: new[] { 1, 1280 }),
        })).Kind, Is.EqualTo(CheckpointKind.Embedding));
        Assert.That(Inspect(Write("unknown", new Tensor[] { new("weights") })).Kind,
            Is.EqualTo(CheckpointKind.Unknown));
    }

    [Test]
    public void Fp8IsNamedAsUnreadable()
    {
        CheckpointReport report = Inspect(Write("fp8",
            Sdxl.Select(t => t.Name == CrossAttention ? t with { DataType = "F8_E4M3" } : t)));
        Assert.That(report.Kind, Is.EqualTo(CheckpointKind.SdxlBase));
        Assert.That(report.UnsupportedDataType, Is.EqualTo("F8_E4M3"));
        Assert.That(report.CanRun, Is.False);
        Assert.That(Explain(report), Does.Contain("F8_E4M3"));
    }

    [Test]
    public void VelocityPredictionFromTheMarkerOrTheMetadata()
    {
        Assert.That(Inspect(Write("vpred", Sdxl.Append(new Tensor("v_pred")))).VelocityPrediction);
        Assert.That(Inspect(Write("vpred-meta", Sdxl,
            new Dictionary<string, string> { ["modelspec.prediction_type"] = "v" })).VelocityPrediction);
        Assert.That(Inspect(Write("eps-meta", Sdxl,
            new Dictionary<string, string> { ["modelspec.prediction_type"] = "epsilon" })).VelocityPrediction,
            Is.False);
    }

    [Test]
    public void ZeroTerminalSnrFromItsMarker()
    {
        CheckpointReport report = Inspect(Write("ztsnr",
            Sdxl.Append(new Tensor("v_pred")).Append(new Tensor("ztsnr"))));
        Assert.That(report.VelocityPrediction);
        Assert.That(report.ZeroTerminalSnr);
        Assert.That(report.CanRun);
        Assert.That(Inspect(Write("base", Sdxl)).ZeroTerminalSnr, Is.False);
    }

    private const string DecoderIn = "decoder.conv_in.weight";
    private static readonly int[] FourChannels = { 512, 4, 3, 3 };
    private static readonly int[] SixteenChannels = { 512, 16, 3, 3 };

    [Test]
    public void TheVaesLatentWidthIsReadUnderEitherName()
    {
        Assert.That(Inspect(Write("prefixed", Sdxl.Append(
            new Tensor(SdxlCheckpoint.VaePrefix + DecoderIn, Shape: FourChannels)))).VaeLatentChannels,
            Is.EqualTo(4));
        Assert.That(Inspect(Write("bare", new Tensor[]
        {
            new("decoder.conv_out.weight"), new("encoder.conv_in.weight"),
            new(DecoderIn, Shape: SixteenChannels),
        })).VaeLatentChannels, Is.EqualTo(16));
        Assert.That(Inspect(Write("none", Sdxl.Where(t => t.Name != Vae))).VaeLatentChannels,
            Is.EqualTo(0));
    }

    /// <summary>What SDXL makes of a file offered as its VAE: a standalone
    /// one or a checkpoint carrying one will do, at four latent channels;
    /// anything else is named for what it is.</summary>
    [Test]
    public void SdxlTakesAVaeFromAVaeFileOrACheckpointAndNothingElse()
    {
        ModelComponent vae = SdxlFamily.VaeComponent;
        Assert.That(SdxlFamily.Instance.Components, Is.EqualTo(new[] { vae }));
        Assert.That(vae.Required, Is.False, "the checkpoint carries one");

        ComponentVerdict Verdict(string name, IEnumerable<Tensor> tensors, string dataType = "F16") =>
            SdxlFamily.Instance.InspectComponent(vae, Write(name, tensors));

        Assert.That(Verdict("bare", new Tensor[]
        {
            new("decoder.conv_out.weight"), new("encoder.conv_in.weight"),
            new(DecoderIn, Shape: FourChannels),
        }), Is.EqualTo(ComponentVerdict.Accepted));
        Assert.That(Verdict("checkpoint", Sdxl.Append(
            new Tensor(SdxlCheckpoint.VaePrefix + DecoderIn, Shape: FourChannels))),
            Is.EqualTo(ComponentVerdict.Accepted));
        Assert.That(Verdict("flux", new Tensor[]
        {
            new("decoder.conv_out.weight"), new("encoder.conv_in.weight"),
            new(DecoderIn, Shape: SixteenChannels),
        }), Is.EqualTo(ComponentVerdict.WrongArchitecture));
        Assert.That(Verdict("lora", new Tensor[]
        {
            new("lora_unet_input_blocks_4_1_proj_in.lora_down.weight"),
        }), Is.EqualTo(ComponentVerdict.NotThisComponent));
        Assert.That(Verdict("novae", Sdxl.Where(t => t.Name != Vae)),
            Is.EqualTo(ComponentVerdict.NotThisComponent));
        Assert.That(Verdict("fp8", new Tensor[]
        {
            new("decoder.conv_out.weight", DataType: "F8_E4M3"), new("encoder.conv_in.weight"),
            new(DecoderIn, Shape: FourChannels),
        }), Is.EqualTo(ComponentVerdict.UnsupportedDataType));

        string junk = _folder.File( "junk.safetensors");
        File.WriteAllBytes(junk, new byte[] { 1, 2, 3 });
        Assert.That(SdxlFamily.Instance.InspectComponent(vae, junk),
            Is.EqualTo(ComponentVerdict.Unreadable));
        Assert.That(ComponentVerdict.WrongArchitecture.Explain(vae), Does.Contain("VAE"));
    }

    [Test]
    public void FilesThatAreNotSafetensorsAreRefusedWithoutReadingPastTheEnd()
    {
        string empty = _folder.File( "empty.safetensors");
        File.WriteAllBytes(empty, new byte[2]);
        string random = _folder.File( "random.safetensors");
        File.WriteAllBytes(random, Enumerable.Range(0, 4096).Select(i => (byte)(i * 131 + 7)).ToArray());
        string truncated = _folder.File( "truncated.safetensors");
        File.WriteAllBytes(truncated, File.ReadAllBytes(Write("whole", Sdxl))[..40]);
        // A header that promises data the file does not have.
        string overrun = _folder.File( "overrun.safetensors");
        byte[] whole = File.ReadAllBytes(Write("whole2", Sdxl));
        File.WriteAllBytes(overrun, whole[..^1]);

        // JSON that parses but is not a tensor table: the same refusal, not
        // whatever the JSON reader throws for the shape it did not find.
        string notATable = WithHeader("notatable", "{\"a\":1}");
        string noShape = WithHeader("noshape", "{\"a\":{\"dtype\":\"F16\",\"data_offsets\":[0,0]}}");
        string noDtype = WithHeader("nodtype",
            "{\"a\":{\"dtype\":null,\"shape\":[0],\"data_offsets\":[0,0]}}");
        string array = WithHeader("array", "[1,2,3]");

        foreach (string path in new[]
                 {
                     empty, random, truncated, overrun, notATable, noShape, noDtype, array,
                 })
        {
            Assert.That(() => Inspect(path), Throws.InstanceOf<InvalidDataException>(),
                Path.GetFileName(path));
        }

        string WithHeader(string name, string json)
        {
            string path = _folder.File(name + ".safetensors");
            byte[] header = System.Text.Encoding.UTF8.GetBytes(json);
            File.WriteAllBytes(path, BitConverter.GetBytes((long)header.Length).Concat(header).ToArray());
            return path;
        }
    }
}
