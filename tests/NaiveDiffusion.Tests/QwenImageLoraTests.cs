using NaiveDiffusion.Models.QwenImage;
using NaiveDiffusion.Tensors;
using NaiveDiffusion.Tests.Fixtures;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Tests;

/// <summary>Qwen-Image 2.1's LoRAs under the names ComfyUI reads them by:
/// the transformer as <c>diffusion_model.…</c>, <c>transformer.…</c> or
/// <c>lora_unet_…</c>, with the pair's own names written any of the ways
/// the trainers write them; the two halves of the fused feed-forward
/// projection under the names they are trained by; the Qwen3-VL encoder as
/// <c>lora_te_…</c> or <c>text_encoders.…</c>; and the files made for
/// another model refused.</summary>
public class QwenImageLoraTests
{
    private TempFolder _folder = null!;
    private FakeSafetensors _files = null!;
    private string _checkpoint = "";
    private string _textEncoder = "";

    private const string Attention = "transformer_blocks.1.attn.to_k.weight";
    private const string Fused = "transformer_blocks.1.img_mlp.gate_up.weight";
    private const string Encoder = "model.layers.0.self_attn.q_proj.weight";

    private static readonly float[] Weight = { 1f, 1f, 1f, 2f, 2f, 2f };
    private static readonly float[] Merged = { 11f, 21f, 31f, 102f, 202f, 302f };

    [SetUp]
    public void WriteTheFiles()
    {
        _folder = new TempFolder();
        _files = new FakeSafetensors(_folder);
        _checkpoint = _files.Write("qwen21", FakeSafetensors.QwenImageDit().Concat(new[]
        {
            new Tensor(Attention, new[] { 2, 3 }, "F32", Weight),
            // [gate; up]: the gate's two rows, then the up projection's.
            new Tensor(Fused, new[] { 4, 3 }, "F32", Weight.Concat(Weight).ToArray()),
        }));
        _textEncoder = _files.Write("qwen3vl", FakeSafetensors.Qwen3VlTextEncoder().Append(
            new Tensor(Encoder, new[] { 2, 3 }, "F32", Weight)));
    }

    [TearDown]
    public void DropFolder() => _folder.Dispose();

    /// <summary>A rank-1 pair whose product is <paramref name="up"/> ⊗ [1, 2, 3].</summary>
    private static IEnumerable<Tensor> Pair(string module, string downName, string upName, params float[] up) => new[]
    {
        new Tensor($"{module}.{downName}", new[] { 1, 3 }, "F32", new[] { 1f, 2f, 3f }),
        new Tensor($"{module}.{upName}", new[] { up.Length, 1 }, "F32", up),
    };

    private static IEnumerable<Tensor> Peft(string module, params float[] up) =>
        Pair(module, "lora_A.weight", "lora_B.weight", up);

    [Test]
    public void EveryWayOfNamingTheTransformerReachesIt()
    {
        var files = new Dictionary<string, IEnumerable<Tensor>>
        {
            ["ai-toolkit"] = Peft("diffusion_model.transformer_blocks.1.attn.to_k", 10f, 100f),
            ["diffusers"] = Peft("transformer.transformer_blocks.1.attn.to_k", 10f, 100f),
            ["comfyui"] = Pair("diffusion_model.transformer_blocks.1.attn.to_k",
                "lora_down.weight", "lora_up.weight", 10f, 100f),
            ["diffsynth"] = Pair("transformer_blocks.1.attn.to_k",
                "lora_A.default.weight", "lora_B.default.weight", 10f, 100f),
            ["kohya"] = Pair("lora_unet_transformer_blocks_1_attn_to_k",
                "lora_down.weight", "lora_up.weight", 10f, 100f),
        };
        using var weights = new QwenImageWeights(_checkpoint, _textEncoder);
        foreach ((string name, IEnumerable<Tensor> tensors) in files)
        {
            var loras = new[] { new LoraSpec(_files.Write(name, tensors), 1f) };
            using LazyWeights dit = weights.LoadDit(loras);
            Assert.That(dit[Attention].DataType, Is.EqualTo(HostDataType.Float16), name);
            Assert.That(dit[Attention].ToFloats(), Is.EqualTo(Merged), name);
            Assert.That(dit[Fused].ToFloats(), Is.EqualTo(Weight.Concat(Weight)), name);
        }
    }

    [Test]
    public void TheHalvesOfTheFusedProjectionTakeTheirOwnPairs()
    {
        string lora = _files.Write("halves",
            Peft("diffusion_model.transformer_blocks.1.img_mlp.gate_layer", 10f, 100f)
                .Concat(Peft("diffusion_model.transformer_blocks.1.img_mlp.proj", 50f, 0f)));
        using var weights = new QwenImageWeights(_checkpoint, _textEncoder);
        using (LazyWeights dit = weights.LoadDit(new[] { new LoraSpec(lora, 1f) }))
        {
            Assert.That(dit[Fused].ToFloats(), Is.EqualTo(new[]
            {
                11f, 21f, 31f, 102f, 202f, 302f,
                51f, 101f, 151f, 2f, 2f, 2f,
            }));
        }

        // One half alone leaves the other as it was, and a pair trained on
        // the fused matrix itself covers all four rows.
        string gate = _files.Write("gate", Peft("diffusion_model.transformer_blocks.1.img_mlp.gate_layer", 10f, 100f));
        using (LazyWeights dit = weights.LoadDit(new[] { new LoraSpec(gate, 1f) }))
        {
            Assert.That(dit[Fused].ToFloats(), Is.EqualTo(Merged.Concat(Weight)));
        }
        string whole = _files.Write("whole",
            Peft("diffusion_model.transformer_blocks.1.img_mlp.gate_up", 10f, 100f, 10f, 100f));
        using (LazyWeights dit = weights.LoadDit(new[] { new LoraSpec(whole, 1f) }))
        {
            Assert.That(dit[Fused].ToFloats(), Is.EqualTo(Merged.Concat(Merged)));
        }
    }

    [Test]
    public void ACheckpointThatKeepsTheProjectionsApartIsReadByTheirNames()
    {
        const string gateLayer = "transformer_blocks.1.img_mlp.gate_layer.weight";
        const string projection = "transformer_blocks.1.img_mlp.proj.weight";
        string split = _files.Write("split", FakeSafetensors.QwenImageDit().Concat(new[]
        {
            new Tensor(gateLayer, new[] { 2, 3 }, "F32", Weight),
            new Tensor(projection, new[] { 2, 3 }, "F32", Weight),
        }));
        string lora = _files.Write("gate", Peft("diffusion_model.transformer_blocks.1.img_mlp.gate_layer", 10f, 100f));
        using var weights = new QwenImageWeights(split, _textEncoder);
        using LazyWeights dit = weights.LoadDit(new[] { new LoraSpec(lora, 1f) });
        Assert.That(dit[gateLayer].ToFloats(), Is.EqualTo(Merged));
        Assert.That(dit[projection].ToFloats(), Is.EqualTo(Weight));
    }

    [Test]
    public void TheTextEncoderIsReachedUnderKohyasNamesAndComfyUis()
    {
        var files = new Dictionary<string, IEnumerable<Tensor>>
        {
            ["kohya"] = Pair("lora_te_layers_0_self_attn_q_proj", "lora_down.weight", "lora_up.weight", 10f, 100f),
            ["wrapped"] = Peft("text_encoders.qwen3vl_8b.transformer.model.layers.0.self_attn.q_proj", 10f, 100f),
            ["bare"] = Peft("text_encoders.transformer.model.layers.0.self_attn.q_proj", 10f, 100f),
        };
        using var weights = new QwenImageWeights(_checkpoint, _textEncoder);
        foreach ((string name, IEnumerable<Tensor> tensors) in files)
        {
            var loras = new[] { new LoraSpec(_files.Write(name, tensors), 1f) };
            Assert.That(weights.LoadTextEncoder(loras)[Encoder].ToFloats(), Is.EqualTo(Merged), name);
        }

        // A file with only transformer layers leaves the encoder as stored.
        string dit = _files.Write("dit", Peft("diffusion_model.transformer_blocks.1.attn.to_k", 10f, 100f));
        Assert.That(weights.LoadTextEncoder(new[] { new LoraSpec(dit, 1f) })[Encoder].DataType,
            Is.EqualTo(HostDataType.Float32));
    }

    [Test]
    public void AFileMadeForAnotherModelIsRefusedBeforeAnythingIsBuilt()
    {
        using var weights = new QwenImageWeights(_checkpoint, _textEncoder);
        string sdxl = _files.Write("sdxl", FakeSafetensors.SdxlLora());
        Assert.That(() => weights.LoadDit(new[] { new LoraSpec(sdxl, 1f) }),
            Throws.InstanceOf<InvalidDataException>().With.Message.Contains("SDXL"));
        string anima = _files.Write("anima", FakeSafetensors.AnimaLora());
        Assert.That(() => weights.LoadDit(new[] { new LoraSpec(anima, 1f) }),
            Throws.InstanceOf<InvalidDataException>().With.Message.Contains("different architecture than Qwen-Image 2.1"));

        // The earlier Qwen-Image has an attention of the same name, 3072
        // wide where this one is 4096.
        string earlier = _files.Write("qwen_image", new[]
        {
            new Tensor("diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight", new[] { 4, 3072 }, "BF16"),
            new Tensor("diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight", new[] { 3072, 4 }, "BF16"),
        });
        Assert.That(() => weights.LoadDit(new[] { new LoraSpec(earlier, 1f) }),
            Throws.InstanceOf<InvalidDataException>().With.Message.Contains("different model"));
    }
}
