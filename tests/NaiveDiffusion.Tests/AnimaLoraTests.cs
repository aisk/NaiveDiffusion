using NaiveDiffusion.Models.Anima;
using NaiveDiffusion.Models.Sdxl;
using NaiveDiffusion.Tensors;
using NaiveDiffusion.Tests.Fixtures;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Tests;

/// <summary>Anima's LoRAs under the names ComfyUI reads them by: the
/// transformer as <c>diffusion_model.…</c> or <c>lora_unet_…</c>, its text
/// adapter as part of it, the Qwen3 encoder as <c>lora_te_…</c> or
/// <c>text_encoders.…</c>; and the files made for SDXL refused with a
/// reason that says so, in both directions.</summary>
public class AnimaLoraTests
{
    private TempFolder _folder = null!;
    private FakeSafetensors _files = null!;
    private string _checkpoint = "";
    private string _textEncoder = "";

    private const string Block = "blocks.0.self_attn.q_proj.weight";
    private const string Adapter = "blocks.0.self_attn.q_proj.weight";
    private const string Encoder = "model.layers.0.self_attn.q_proj.weight";

    [SetUp]
    public void WriteTheFiles()
    {
        _folder = new TempFolder();
        _files = new FakeSafetensors(_folder);
        // One small weight with values in each place a LoRA can reach.
        var weight = new[] { 1f, 1f, 1f, 2f, 2f, 2f };
        _checkpoint = _files.Write("anima", FakeSafetensors.AnimaDit().Concat(new[]
        {
            new Tensor("net." + Block, new[] { 2, 3 }, "F32", weight),
            new Tensor("net.llm_adapter." + Adapter, new[] { 2, 3 }, "F32", weight),
        }));
        _textEncoder = _files.Write("qwen", FakeSafetensors.Qwen3TextEncoder().Append(
            new Tensor(Encoder, new[] { 2, 3 }, "F32", weight)));
    }

    [TearDown]
    public void DropFolder() => _folder.Dispose();

    /// <summary>A rank-1 pair whose product is [10, 100] ⊗ [1, 2, 3].</summary>
    private static IEnumerable<Tensor> Pair(string module, string down, string up) => new[]
    {
        new Tensor($"{module}.{down}", new[] { 1, 3 }, "F32", new[] { 1f, 2f, 3f }),
        new Tensor($"{module}.{up}", new[] { 2, 1 }, "F32", new[] { 10f, 100f }),
    };

    private static IEnumerable<Tensor> Peft(string module) => Pair(module, "lora_A.weight", "lora_B.weight");

    private static IEnumerable<Tensor> Kohya(string module) => Pair(module, "lora_down.weight", "lora_up.weight");

    private static readonly float[] Merged = { 11f, 21f, 31f, 102f, 202f, 302f };

    [Test]
    public void AiToolkitsNamesReachTheTransformerAndItsAdapterWithTheirAlpha()
    {
        string lora = _files.Write("peft", Peft("diffusion_model.blocks.0.self_attn.q_proj")
            .Concat(Peft("diffusion_model.llm_adapter.blocks.0.self_attn.q_proj"))
            .Append(new Tensor("diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.alpha",
                DataType: "F32", Values: new[] { 0.5f })));
        var loras = new[] { new LoraSpec(lora, 1f) };
        using var weights = new AnimaWeights(_checkpoint, _textEncoder);

        using (LazyWeights dit = weights.LoadDit(loras))
        {
            Assert.That(dit[Block].DataType, Is.EqualTo(HostDataType.Float16));
            Assert.That(dit[Block].ToFloats(), Is.EqualTo(Merged));
            Assert.That(dit.Keys.Any(key => key.StartsWith("llm_adapter.")), Is.False);
        }
        // The alpha of 0.5 over rank 1 halves the adapter's delta.
        Assert.That(weights.LoadAdapter(loras)[Adapter].ToFloats(),
            Is.EqualTo(new[] { 6f, 11f, 16f, 52f, 102f, 152f }));
        HostTensor untouched = weights.LoadTextEncoder(loras)[Encoder];
        Assert.That(untouched.DataType, Is.EqualTo(HostDataType.Float32), "a LoRA with no encoder layers leaves it");
    }

    [Test]
    public void KohyaNamesReachAllThree()
    {
        string lora = _files.Write("kohya", Kohya("lora_unet_blocks_0_self_attn_q_proj")
            .Concat(Kohya("lora_unet_llm_adapter_blocks_0_self_attn_q_proj"))
            .Concat(Kohya("lora_te_layers_0_self_attn_q_proj")));
        var loras = new[] { new LoraSpec(lora, 1f) };
        using var weights = new AnimaWeights(_checkpoint, _textEncoder);

        using (LazyWeights dit = weights.LoadDit(loras))
        {
            Assert.That(dit[Block].ToFloats(), Is.EqualTo(Merged));
        }
        Assert.That(weights.LoadAdapter(loras)[Adapter].ToFloats(), Is.EqualTo(Merged));
        Assert.That(weights.LoadTextEncoder(loras)[Encoder].ToFloats(), Is.EqualTo(Merged));
    }

    [Test]
    public void ComfyUisGenericEncoderNamesAreReadWithOrWithoutTheWrapper()
    {
        foreach (string prefix in new[] { "text_encoders.qwen3_06b.transformer.model.", "text_encoders.transformer.model." })
        {
            string lora = _files.Write("generic", Peft(prefix + "layers.0.self_attn.q_proj"));
            using var weights = new AnimaWeights(_checkpoint, _textEncoder);
            Assert.That(weights.LoadTextEncoder(new[] { new LoraSpec(lora, 1f) })[Encoder].ToFloats(),
                Is.EqualTo(Merged), prefix);
        }
    }

    [Test]
    public void AnSdxlLoraIsRefusedForAnimaAndAnAnimaLoraForSdxl()
    {
        using var weights = new AnimaWeights(_checkpoint, _textEncoder);
        // Its CLIP-L half gives it away on reading.
        string sdxl = _files.Write("sdxl", FakeSafetensors.SdxlLora());
        Assert.That(() => weights.LoadDit(new[] { new LoraSpec(sdxl, 1f) }),
            Throws.InstanceOf<InvalidDataException>().With.Message.Contains("SDXL"));
        // A UNet-only one has none of the transformer's layers.
        string unet = _files.Write("unet", FakeSafetensors.SdxlLora().Where(t => t.Name.StartsWith("lora_unet_")));
        Assert.That(() => weights.LoadDit(new[] { new LoraSpec(unet, 1f) }),
            Throws.InstanceOf<InvalidDataException>().With.Message.Contains("different architecture than Anima"));

        // SDXL reads none of an Anima LoRA's names; the file says what it is for.
        string anima = _files.Write("anima_lora", FakeSafetensors.AnimaLora(),
            new Dictionary<string, string> { ["modelspec.architecture"] = "anima/lora" });
        Assert.That(() => new LoraFile(anima, SdxlLoras.Layout),
            Throws.InstanceOf<InvalidDataException>().With.Message.Contains("anima/lora"));
    }
}
