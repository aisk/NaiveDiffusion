using NaiveDiffusion.Models.Sdxl;
using NaiveDiffusion.Tensors;
using NaiveDiffusion.Tests.Fixtures;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Tests;

/// <summary>Reading a LoRA's pairs under either naming, refusing the files
/// that are not plain LoRAs for this architecture, and folding the deltas
/// into a weight on the CPU: W' = W + weight × alpha / rank × up · down,
/// checked against the arithmetic done by hand on a tiny matrix.</summary>
public class LoraFileTests
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

    private static LoraLayout Layout => SdxlLoras.Layout;

    [Test]
    public void KohyaPairsAreFiledByTowerAndLayerWithTheirAlpha()
    {
        using var file = new LoraFile(_files.Write("kohya", FakeSafetensors.SdxlLora()
            .Select(t => t.Name.EndsWith(".alpha") ? t with { Values = new[] { 2f } } : t)), Layout);
        Assert.That(file.Name, Is.EqualTo("kohya"));
        IReadOnlyDictionary<string, LoraFile.Module> unet = file.Modules(SdxlLoras.Unet);
        Assert.That(unet.Keys, Is.EquivalentTo(new[] { "input_blocks_4_1_transformer_blocks_0_attn1_to_k" }));
        LoraFile.Module module = unet.Values.Single();
        Assert.That(module.Down.Shape, Is.EqualTo(new[] { 4, 320 }));
        Assert.That(module.Up.Shape, Is.EqualTo(new[] { 320, 4 }));
        Assert.That(module.Alpha, Is.EqualTo(2f));
        Assert.That(file.Modules(SdxlLoras.TextEncoder).Values.Single().Alpha, Is.Null, "no alpha: rank");
        Assert.That(file.Modules(SdxlLoras.TextEncoder2), Is.Empty);
    }

    [Test]
    public void PeftNamesAreReadIntoTheSameTowers()
    {
        using var file = new LoraFile(_files.Write("peft", new[]
        {
            new Tensor("unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.lora_A.weight", new[] { 4, 320 }),
            new Tensor("unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.lora_B.weight", new[] { 320, 4 }),
            new Tensor("text_encoder_2.text_model.encoder.layers.0.mlp.fc1.lora_A.weight", new[] { 4, 1280 }),
            new Tensor("text_encoder_2.text_model.encoder.layers.0.mlp.fc1.lora_B.weight", new[] { 5120, 4 }),
        }), Layout);
        Assert.That(file.Modules(SdxlLoras.Unet).Keys,
            Is.EquivalentTo(new[] { "down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k" }));
        Assert.That(file.Modules(SdxlLoras.TextEncoder2).Keys,
            Is.EquivalentTo(new[] { "text_model_encoder_layers_0_mlp_fc1" }));
    }

    [Test]
    public void TheFilesThatAreNotPlainSdxlLorasAreRefusedWithTheReason()
    {
        string sd15 = _files.Write("sd15", new[]
        {
            new Tensor("lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight", new[] { 4, 768 }),
            new Tensor("lora_te_text_model_encoder_layers_0_mlp_fc1.lora_up.weight", new[] { 3072, 4 }),
        });
        Assert.That(() => new LoraFile(sd15, Layout),
            Throws.InstanceOf<InvalidDataException>().With.Message.Contains("Stable Diffusion 1.x"));

        string dora = _files.Write("dora", FakeSafetensors.SdxlLora()
            .Append(new Tensor("lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_k.dora_scale", new[] { 320 })));
        Assert.That(() => new LoraFile(dora, Layout),
            Throws.InstanceOf<InvalidDataException>().With.Message.Contains("DoRA"));

        string loha = _files.Write("loha", new[]
        {
            new Tensor("lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_k.hada_w1_a", new[] { 4, 320 }),
        });
        Assert.That(() => new LoraFile(loha, Layout),
            Throws.InstanceOf<InvalidDataException>().With.Message.Contains("LoHa"));

        string half = _files.Write("half", FakeSafetensors.SdxlLora().Take(1));
        Assert.That(() => new LoraFile(half, Layout),
            Throws.InstanceOf<InvalidDataException>().With.Message.Contains("no up matrix"));
    }

    [Test]
    public void TheDeltaIsFoldedIntoTheWeightAtTheStoredWidth()
    {
        // W is 2x3; down is [rank 1, 3], up is [2, rank 1]: the delta is the
        // outer product up · down, scaled by weight × alpha / rank.
        string lora = _files.Write("delta", new[]
        {
            new Tensor("lora_unet_conv_in.lora_down.weight", new[] { 1, 3 }, "F32", new[] { 1f, 2f, 3f }),
            new Tensor("lora_unet_conv_in.lora_up.weight", new[] { 2, 1 }, "F32", new[] { 10f, 100f }),
            new Tensor("lora_unet_conv_in.alpha", Values: new[] { 0.5f }, DataType: "F32"),
        });
        var tensors = new Dictionary<string, HostTensor>
        {
            ["conv_in.weight"] = HostTensor.FromFloats(new[] { 1f, 1f, 1f, 2f, 2f, 2f }, 2, 3),
            ["conv_out.weight"] = HostTensor.FromFloats(new[] { 7f }, 1, 1),
        };
        HostTensor untouched = tensors["conv_out.weight"];

        LoraMerge.Apply(tensors, Layout, SdxlLoras.Unet, new[] { new LoraSpec(lora, 2f) });

        // scale = 2 × 0.5 / 1 = 1: row 0 gets 10 × [1 2 3], row 1 gets 100 × [1 2 3].
        Assert.That(tensors["conv_in.weight"].ToFloats(),
            Is.EqualTo(new[] { 11f, 21f, 31f, 102f, 202f, 302f }));
        Assert.That(tensors["conv_in.weight"].DataType, Is.EqualTo(HostDataType.Float32));
        Assert.That(tensors["conv_out.weight"], Is.SameAs(untouched), "a weight no LoRA names is the same object");

        // A half-precision weight stays half precision, and bfloat16 becomes half.
        var halves = new Dictionary<string, HostTensor>
        {
            ["conv_in.weight"] = HostTensor.FromHalves(new Half[] { (Half)1, (Half)1, (Half)1, (Half)2, (Half)2, (Half)2 }, 2, 3),
        };
        LoraMerge.Apply(halves, Layout, SdxlLoras.Unet, new[] { new LoraSpec(lora, 1f) });
        Assert.That(halves["conv_in.weight"].DataType, Is.EqualTo(HostDataType.Float16));
        Assert.That(halves["conv_in.weight"].ToFloats(), Is.EqualTo(new[] { 6f, 11f, 16f, 52f, 102f, 152f }));
    }

    [Test]
    public void AFileNoneOfWhoseLayersFitIsRefusedAndAWeightOfZeroIsNotRead()
    {
        string lora = _files.Write("elsewhere", new[]
        {
            new Tensor("lora_unet_somewhere_else.lora_down.weight", new[] { 1, 3 }, "F32", new[] { 1f, 1f, 1f }),
            new Tensor("lora_unet_somewhere_else.lora_up.weight", new[] { 2, 1 }, "F32", new[] { 1f, 1f }),
        });
        var tensors = new Dictionary<string, HostTensor>
        {
            ["conv_in.weight"] = HostTensor.FromFloats(new float[6], 2, 3),
        };
        Assert.That(() => LoraMerge.Apply(tensors, Layout, SdxlLoras.Unet, new[] { new LoraSpec(lora, 1f) }),
            Throws.InstanceOf<InvalidDataException>().With.Message.Contains("SDXL"));
        // At weight zero nothing is even opened, so a file that is not there is fine.
        Assert.That(() => LoraMerge.Apply(tensors, Layout, SdxlLoras.Unet,
                new[] { new LoraSpec(_folder.File("gone.safetensors"), 0f) }),
            Throws.Nothing);
    }

    [Test]
    public void AliasesLetKohyaNamesFindDiffusersKeys()
    {
        string lora = _files.Write("aliased", new[]
        {
            new Tensor("lora_unet_input_blocks_0_0.lora_down.weight", new[] { 1, 2 }, "F32", new[] { 1f, 1f }),
            new Tensor("lora_unet_input_blocks_0_0.lora_up.weight", new[] { 1, 1 }, "F32", new[] { 1f }),
        });
        var tensors = new Dictionary<string, HostTensor>
        {
            ["conv_in.weight"] = HostTensor.FromFloats(new[] { 0f, 0f }, 1, 2),
        };
        using LoraPatchSet patches = LoraPatchSet.Prepare(tensors, Layout, SdxlLoras.Unet,
            new[] { new LoraSpec(lora, 1f) },
            key => key == "conv_in.weight" ? new[] { "input_blocks.0.0.weight" } : Array.Empty<string>());
        Assert.That(patches.Touches("conv_in.weight"));
        Assert.That(patches.Apply("conv_in.weight", tensors["conv_in.weight"]).ToFloats(),
            Is.EqualTo(new[] { 1f, 1f }));
    }
}
