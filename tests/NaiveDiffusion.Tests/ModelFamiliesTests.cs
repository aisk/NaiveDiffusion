using NaiveDiffusion.Models;
using NaiveDiffusion.Models.Anima;
using NaiveDiffusion.Models.QwenImage;
using NaiveDiffusion.Models.Sdxl;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Tests.Fixtures;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Tests;

/// <summary>Which family a file belongs to, from hand-written safetensors
/// headers the way <see cref="CheckpointInspectorTests"/> writes them: an
/// Anima transformer under either prefix, a Qwen-Image 2.1 transformer
/// under either of its own, their text encoders and VAEs — which each
/// family takes only its own of — and the files that are none of those.</summary>
public class ModelFamiliesTests
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

    private string Write(string name, IEnumerable<Tensor> tensors) => _files.Write(name, tensors);

    private static IEnumerable<Tensor> AnimaDit(string prefix) => FakeSafetensors.AnimaDit(prefix);

    [TestCase("net.")]
    [TestCase("model.diffusion_model.")]
    public void AnAnimaTransformerIsRecognizedUnderEitherPrefix(string prefix)
    {
        string path = Write("anima", AnimaDit(prefix));
        Assert.That(ModelFamilies.Identify(path, out _), Is.SameAs(AnimaFamily.Instance));
        using var file = new SafetensorsFile(path);
        Assert.That(AnimaCheckpoint.DitPrefix(file), Is.EqualTo(prefix));
        Assert.That(AnimaCheckpoint.Shape(file), Is.EqualTo((2048, 2, 16)));
    }

    [Test]
    public void AnFp8AnimaIsTurnedAway()
    {
        string path = Write("anima8", AnimaDit("net.").Select(t => t with { DataType = "F8_E4M3" }));
        Assert.That(ModelFamilies.Identify(path, out CheckpointInspector.CheckpointReport report), Is.Null);
        Assert.That(report.UnsupportedDataType, Is.EqualTo("F8_E4M3"));
    }

    [TestCase("")]
    [TestCase("model.diffusion_model.")]
    public void AQwenImageTransformerIsRecognizedUnderEitherPrefix(string prefix)
    {
        string path = Write("qwen21", FakeSafetensors.QwenImageDit(prefix));
        Assert.That(ModelFamilies.Identify(path, out CheckpointInspector.CheckpointReport report),
            Is.SameAs(QwenImageFamily.Instance));
        Assert.That(report.Kind, Is.EqualTo(CheckpointInspector.CheckpointKind.QwenImage));
        using var file = new SafetensorsFile(path);
        Assert.That(QwenImageCheckpoint.DitPrefix(file), Is.EqualTo(prefix));
        Assert.That(QwenImageCheckpoint.Shape(file), Is.EqualTo((4096, 2, 64)));
        Assert.That(QwenImageCheckpoint.ContextWidth(file), Is.EqualTo(4096));
    }

    [Test]
    public void AnFp8QwenImageIsTurnedAway()
    {
        string path = Write("qwen8", FakeSafetensors.QwenImageDit().Select(t => t with { DataType = "F8_E4M3" }));
        Assert.That(ModelFamilies.Identify(path, out CheckpointInspector.CheckpointReport report), Is.Null);
        Assert.That(report.Kind, Is.EqualTo(CheckpointInspector.CheckpointKind.QwenImage));
        Assert.That(report.UnsupportedDataType, Is.EqualTo("F8_E4M3"));
        Assert.That(CheckpointInspector.Explain(report), Does.Contain("quantized"));
    }

    [Test]
    public void AnSdxlCheckpointStillGoesToSdxl()
    {
        string path = Write("sdxl", FakeSafetensors.SdxlCheckpoint());
        Assert.That(ModelFamilies.Identify(path, out _), Is.SameAs(SdxlFamily.Instance));
    }

    [Test]
    public void TheComponentsAreToldApartFromTheirHeaders()
    {
        AnimaFamily anima = AnimaFamily.Instance;
        string textEncoder = Write("qwen", FakeSafetensors.Qwen3TextEncoder());
        string widerEncoder = Write("qwen4b", FakeSafetensors.Qwen3TextEncoder(width: 2560));
        string wanVae = Write("wanvae", FakeSafetensors.WanVae());
        string sdVae = Write("sdvae", new[]
        {
            new Tensor("decoder.conv_out.weight", new[] { 3, 128, 3, 3 }),
            new Tensor("decoder.conv_in.weight", new[] { 512, 4, 3, 3 }),
            new Tensor("encoder.conv_in.weight", new[] { 128, 3, 3, 3 }),
        });
        string dit = Write("dit", AnimaDit("net."));

        Assert.That(anima.InspectComponent(AnimaFamily.TextEncoderComponent, textEncoder),
            Is.EqualTo(ComponentVerdict.Accepted));
        Assert.That(anima.InspectComponent(AnimaFamily.TextEncoderComponent, widerEncoder),
            Is.EqualTo(ComponentVerdict.WrongArchitecture));
        Assert.That(anima.InspectComponent(AnimaFamily.TextEncoderComponent, wanVae),
            Is.EqualTo(ComponentVerdict.NotThisComponent));
        Assert.That(anima.InspectComponent(AnimaFamily.VaeComponent, wanVae),
            Is.EqualTo(ComponentVerdict.Accepted));
        Assert.That(anima.InspectComponent(AnimaFamily.VaeComponent, sdVae),
            Is.EqualTo(ComponentVerdict.WrongArchitecture));
        Assert.That(anima.InspectComponent(AnimaFamily.VaeComponent, dit),
            Is.EqualTo(ComponentVerdict.NotThisComponent));
        Assert.That(anima.InspectComponent(AnimaFamily.VaeComponent, _folder.File("missing.safetensors")),
            Is.EqualTo(ComponentVerdict.Unreadable));
        // And the other way round: SDXL will not take the video VAE, which
        // is a VAE, for another family.
        Assert.That(SdxlFamily.Instance.InspectComponent(SdxlFamily.VaeComponent, wanVae),
            Is.EqualTo(ComponentVerdict.WrongArchitecture));
    }

    /// <summary>Qwen-Image's parts, and the two families' parts against
    /// each other: each is the other's kind of file and not its own.</summary>
    [Test]
    public void QwenImagesComponentsAreToldApartFromAnimas()
    {
        QwenImageFamily qwen = QwenImageFamily.Instance;
        AnimaFamily anima = AnimaFamily.Instance;
        string vl = Write("qwen3vl", FakeSafetensors.Qwen3VlTextEncoder());
        string smallVl = Write("qwen3vl4b", FakeSafetensors.Qwen3VlTextEncoder(width: 2560));
        string plainQwen = Write("qwen06", FakeSafetensors.Qwen3TextEncoder());
        string vae = Write("qwen21vae", FakeSafetensors.Wan22Vae());
        string wanVae = Write("wanvae", FakeSafetensors.WanVae());
        string dit = Write("qwen21", FakeSafetensors.QwenImageDit());

        Assert.That(qwen.InspectComponent(QwenImageFamily.TextEncoderComponent, vl),
            Is.EqualTo(ComponentVerdict.Accepted));
        Assert.That(qwen.InspectComponent(QwenImageFamily.TextEncoderComponent, smallVl),
            Is.EqualTo(ComponentVerdict.WrongArchitecture));
        Assert.That(qwen.InspectComponent(QwenImageFamily.TextEncoderComponent, plainQwen),
            Is.EqualTo(ComponentVerdict.WrongArchitecture), "Anima's Qwen3 has no vision tower");
        Assert.That(qwen.InspectComponent(QwenImageFamily.TextEncoderComponent, vae),
            Is.EqualTo(ComponentVerdict.NotThisComponent));
        Assert.That(qwen.InspectComponent(QwenImageFamily.VaeComponent, vae),
            Is.EqualTo(ComponentVerdict.Accepted));
        Assert.That(qwen.InspectComponent(QwenImageFamily.VaeComponent, wanVae),
            Is.EqualTo(ComponentVerdict.WrongArchitecture), "Wan 2.1's sixteen channels");
        Assert.That(qwen.InspectComponent(QwenImageFamily.VaeComponent, dit),
            Is.EqualTo(ComponentVerdict.NotThisComponent));

        Assert.That(anima.InspectComponent(AnimaFamily.TextEncoderComponent, vl),
            Is.EqualTo(ComponentVerdict.WrongArchitecture));
        Assert.That(anima.InspectComponent(AnimaFamily.VaeComponent, vae),
            Is.EqualTo(ComponentVerdict.WrongArchitecture));
        Assert.That(ModelFamilies.Identify(vae, out CheckpointInspector.CheckpointReport report), Is.Null);
        Assert.That(report.Kind, Is.EqualTo(CheckpointInspector.CheckpointKind.VaeOnly));
    }

    [Test]
    public void EveryFamilysPartsAreListedOncePerId()
    {
        Assert.That(ModelFamilies.AllComponents.Select(part => part.Id),
            Is.EquivalentTo(new[] { "vae", "text_encoder" }));
        Assert.That(AnimaFamily.Instance.MaxClipSkip, Is.Zero);
        Assert.That(QwenImageFamily.Instance.MaxClipSkip, Is.Zero);
        Assert.That(SdxlFamily.Instance.MaxClipSkip, Is.EqualTo(12));
    }
}
