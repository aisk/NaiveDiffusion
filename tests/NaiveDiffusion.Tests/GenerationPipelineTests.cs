using NaiveDiffusion.Models;
using NaiveDiffusion.Models.Anima;
using NaiveDiffusion.Models.QwenImage;
using NaiveDiffusion.Models.Sdxl;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Sampling;
using NaiveDiffusion.Text;
using NaiveDiffusion.Tests.Fixtures;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Tests;

/// <summary>What the pipeline refuses before a weight is loaded, from
/// hand-written headers: the hard invariants — a prompt
/// that still names a snippet or a LoRA, an option the family does not
/// have, a file that is not what it was given as — each stop the run with
/// a sentence, and a run that passes gets its family's sampling.</summary>
public class GenerationPipelineTests
{
    private TempFolder _folder = null!;
    private FakeSafetensors _files = null!;
    private string _sdxl = "";
    private string _anima = "";
    private string _qwen = "";
    private string _wanVae = "";
    private string _qwenImage = "";
    private string _qwen3Vl = "";
    private string _qwenVae = "";

    [SetUp]
    public void WriteTheFiles()
    {
        _folder = new TempFolder();
        _files = new FakeSafetensors(_folder);
        _sdxl = _files.Write("sdxl", FakeSafetensors.SdxlCheckpoint());
        _anima = _files.Write("anima", FakeSafetensors.AnimaDit());
        _qwen = _files.Write("qwen", FakeSafetensors.Qwen3TextEncoder());
        _wanVae = _files.Write("wanvae", FakeSafetensors.WanVae());
        _qwenImage = _files.Write("qwen21", FakeSafetensors.QwenImageDit());
        _qwen3Vl = _files.Write("qwen3vl", FakeSafetensors.Qwen3VlTextEncoder());
        _qwenVae = _files.Write("qwen21vae", FakeSafetensors.Wan22Vae());
    }

    [TearDown]
    public void DropTheFiles() => _folder.Dispose();

    private static GenerationPipeline Sdxl => ModelFamilies.PipelineFor(SdxlFamily.Instance);

    private static GenerationPipeline Anima => ModelFamilies.PipelineFor(AnimaFamily.Instance);

    private static GenerationPipeline QwenImage => ModelFamilies.PipelineFor(QwenImageFamily.Instance);

    private GenerationOptions SdxlOptions() => new()
    {
        Prompt = "1girl", CheckpointPath = _sdxl, Width = 1024, Height = 1024,
    };

    private GenerationOptions AnimaOptions() => new()
    {
        Prompt = "1girl", CheckpointPath = _anima, Width = 1024, Height = 1024,
        Components = new Dictionary<string, string>
        {
            [AnimaFamily.TextEncoderComponent.Id] = _qwen,
            [AnimaFamily.VaeComponent.Id] = _wanVae,
        },
    };

    private GenerationOptions QwenImageOptions() => new()
    {
        Prompt = "1girl", CheckpointPath = _qwenImage, Width = 1024, Height = 1024, Guidance = 1f,
        Components = new Dictionary<string, string>
        {
            [QwenImageFamily.TextEncoderComponent.Id] = _qwen3Vl,
            [QwenImageFamily.VaeComponent.Id] = _qwenVae,
        },
    };

    [Test]
    public void ARunThatPassesGetsItsFamilysSampling()
    {
        Assert.That(Sdxl.Prepare(SdxlOptions()).Prediction, Is.SameAs(EpsilonPrediction.Instance));
        Assert.That(Anima.Prepare(AnimaOptions()).Prediction, Is.SameAs(FlowPrediction.Instance));
        Assert.That(QwenImage.Prepare(QwenImageOptions()).Prediction, Is.SameAs(FlowPrediction.Instance));
    }

    /// <summary>The shift follows the image: the release's scheduler puts
    /// the log-shift on a line from 0.5 at 256 tokens to 0.9 at 8192, and
    /// a sigma read off the shifted table shows it.</summary>
    [Test]
    public void QwenImagesShiftFollowsTheImageSize()
    {
        Assert.That(QwenImageFamily.Mu(256), Is.EqualTo(0.5).Within(1e-9));
        Assert.That(QwenImageFamily.Mu(8192), Is.EqualTo(0.9).Within(1e-9));
        Assert.That(QwenImageFamily.Mu(4096), Is.EqualTo(0.6936).Within(1e-3), "1024², ComfyUI's 0.69");

        ModelSampling at1024 = QwenImage.Prepare(QwenImageOptions());
        ModelSampling at512 = QwenImage.Prepare(QwenImageOptions() with { Width = 512, Height = 512 });
        Assert.That(at1024.Levels.SigmaMax, Is.EqualTo(1.0).Within(1e-9), "a flow model starts from noise");
        double shift = Math.Exp(QwenImageFamily.Mu(4096));
        double shifted = at1024.Levels.SigmaAt(500);
        Assert.That(shifted, Is.EqualTo(shift * 0.501 / (1 + (shift - 1) * 0.501)).Within(1e-6));
        Assert.That(at512.Levels.SigmaAt(500), Is.LessThan(shifted), "a smaller image shifts less");
    }

    /// <summary>Guidance 1 is one branch and no negative prompt for
    /// Qwen-Image and Anima; SDXL keeps both whatever the scale. Each
    /// family's default is its reference's, and Qwen-Image's, released to
    /// run without guidance, is one branch.</summary>
    [Test]
    public void GuidanceAtOneRunsOneBranchWhereTheFamilySaysSo()
    {
        Assert.That(QwenImageFamily.Instance.Guidance.Branches(1f), Is.EqualTo(1));
        Assert.That(QwenImageFamily.Instance.Guidance.Branches(4f), Is.EqualTo(2));
        Assert.That(SdxlFamily.Instance.Guidance.Branches(1f), Is.EqualTo(2));
        Assert.That(AnimaFamily.Instance.Guidance.Branches(1f), Is.EqualTo(1));
        Assert.That(AnimaFamily.Instance.Guidance.Branches(4f), Is.EqualTo(2));
        Assert.That(SdxlFamily.Instance.DefaultGuidance, Is.EqualTo(5f));
        Assert.That(AnimaFamily.Instance.DefaultGuidance, Is.EqualTo(4f));
        Assert.That(QwenImageFamily.Instance.DefaultGuidance, Is.EqualTo(1f));
        Assert.That(QwenImageFamily.Instance.Guidance.Branches(QwenImageFamily.Instance.DefaultGuidance),
            Is.EqualTo(1));
        float[] cond = { 1f, 2f }, uncond = { 0f, 0f };
        Assert.That(OptionalClassifierFreeGuidance.Instance.Combine(new[] { cond }, 1f), Is.SameAs(cond));
        Assert.That(OptionalClassifierFreeGuidance.Instance.Combine(new[] { uncond, cond }, 2f),
            Is.EqualTo(new[] { 2f, 4f }));
    }

    [Test]
    public void AVelocityCheckpointSamplesAsSuch()
    {
        string vpred = _files.Write("vpred", FakeSafetensors.SdxlCheckpoint()
            .Append(new Tensor("v_pred")).Append(new Tensor("ztsnr")));
        ModelSampling sampling = Sdxl.Prepare(SdxlOptions() with { CheckpointPath = vpred });
        Assert.That(sampling.Prediction, Is.SameAs(VelocityPrediction.Instance));
        Assert.That(sampling.Levels.SigmaMax, Is.GreaterThan(1000), "the zero-terminal-SNR rescale");
    }

    [Test]
    public void AnUnexpandedSnippetOrLoraTagIsRefused()
    {
        Assert.That(() => Sdxl.Prepare(SdxlOptions() with { Prompt = "1girl, {miku}" }),
            Throws.ArgumentException.With.Message.Contains("{miku}"));
        Assert.That(() => Sdxl.Prepare(SdxlOptions() with { Negative = "{bad hands}" }),
            Throws.ArgumentException.With.Message.Contains("negative prompt"));
        Assert.That(() => Sdxl.Prepare(SdxlOptions() with { Prompt = "1girl <lora:watercolor:0.8>" }),
            Throws.ArgumentException.With.Message.Contains("<lora:watercolor>"));
    }

    /// <summary>Whether a LoRA's layers fit is the merge's to say; the
    /// pipeline asks only that the file be a LoRA, for every family that
    /// takes one.</summary>
    [Test]
    public void AnimaTakesALoraAndRefusesACheckpointGivenAsOne()
    {
        string lora = _files.Write("anima_lora", FakeSafetensors.AnimaLora(),
            new Dictionary<string, string> { ["modelspec.architecture"] = "anima/lora" });
        Assert.That(() => Anima.Prepare(AnimaOptions() with { Loras = new[] { new LoraSpec(lora, 0.8f) } }),
            Throws.Nothing);
        Assert.That(() => Anima.Prepare(AnimaOptions() with { Loras = new[] { new LoraSpec(_anima, 1f) } }),
            Throws.ArgumentException.With.Message.Contains("not a LoRA"));
    }

    [Test]
    public void TheOptionsAFamilyDoesNotHaveAreRefused()
    {
        Assert.That(() => Anima.Prepare(AnimaOptions() with { ClipSkip = 1 }),
            Throws.ArgumentException.With.Message.Contains("clip skip"));
        Assert.That(() => Anima.Prepare(AnimaOptions() with { Schedule = ScheduleKind.AlignYourSteps }),
            Throws.ArgumentException.With.Message.Contains("Align"));
        Assert.That(() => Anima.Prepare(AnimaOptions() with { Width = 1032 }),
            Throws.ArgumentException.With.Message.Contains("16"));
        Assert.That(() => QwenImage.Prepare(QwenImageOptions() with { ClipSkip = 1 }),
            Throws.ArgumentException.With.Message.Contains("clip skip"));
        Assert.That(() => QwenImage.Prepare(QwenImageOptions() with
            {
                Loras = new[] { new LoraSpec(_files.Write("lora", FakeSafetensors.SdxlLora()), 1f) },
            }), Throws.ArgumentException.With.Message.Contains("LoRAs are not supported"));
        Assert.That(() => QwenImage.Prepare(QwenImageOptions() with { Schedule = ScheduleKind.AlignYourSteps }),
            Throws.ArgumentException.With.Message.Contains("Align"));
        Assert.That(() => QwenImage.Prepare(QwenImageOptions() with { Height = 1000 }),
            Throws.ArgumentException.With.Message.Contains("16"));
        // Anima's files are not Qwen-Image's, and each required part is asked for.
        Assert.That(() => QwenImage.Prepare(QwenImageOptions() with
            {
                Components = new Dictionary<string, string>
                {
                    [QwenImageFamily.TextEncoderComponent.Id] = _qwen,
                    [QwenImageFamily.VaeComponent.Id] = _qwenVae,
                },
            }), Throws.ArgumentException.With.Message.Contains("text encoder"));
        Assert.That(() => QwenImage.Prepare(QwenImageOptions() with
            {
                Components = new Dictionary<string, string>
                {
                    [QwenImageFamily.TextEncoderComponent.Id] = _qwen3Vl,
                    [QwenImageFamily.VaeComponent.Id] = _wanVae,
                },
            }), Throws.ArgumentException.With.Message.Contains("VAE"));
        Assert.That(() => QwenImage.Prepare(QwenImageOptions() with
            {
                Components = new Dictionary<string, string>(),
            }), Throws.ArgumentException);
        Assert.That(() => Anima.Prepare(AnimaOptions() with { CheckpointPath = _qwenImage }),
            Throws.InstanceOf<CheckpointNotSupportedException>());
        // SDXL takes all of them, within their ranges.
        Assert.That(() => Sdxl.Prepare(SdxlOptions() with { ClipSkip = 13 }),
            Throws.ArgumentException.With.Message.Contains("clip skip"));
        Assert.That(() => Sdxl.Prepare(SdxlOptions() with { ClipSkip = 1, Width = 1028 }),
            Throws.ArgumentException.With.Message.Contains("8"));
        Assert.That(() => Sdxl.Prepare(SdxlOptions() with { ClipSkip = 12, Schedule = ScheduleKind.AlignYourSteps }),
            Throws.Nothing);

        // Half-precision blocks are a transformer's choice; the UNet
        // computes at half precision as it is.
        Assert.That(() => Sdxl.Prepare(SdxlOptions() with { DenoiserHalfCompute = true }),
            Throws.ArgumentException.With.Message.Contains("half-precision"));
        Assert.That(() => Anima.Prepare(AnimaOptions() with { DenoiserHalfCompute = true }), Throws.Nothing);
        Assert.That(() => QwenImage.Prepare(QwenImageOptions() with { DenoiserHalfCompute = true }),
            Throws.Nothing);
    }

    [Test]
    public void TheFilesAreCheckedAgainstWhatTheyWereGivenAs()
    {
        // The other family's checkpoint.
        Assert.That(() => Sdxl.Prepare(SdxlOptions() with { CheckpointPath = _anima }),
            Throws.InstanceOf<CheckpointNotSupportedException>());
        Assert.That(() => Anima.Prepare(AnimaOptions() with { CheckpointPath = _sdxl }),
            Throws.InstanceOf<CheckpointNotSupportedException>());
        // A part the family does not have.
        Assert.That(() => Sdxl.Prepare(SdxlOptions() with
            {
                Components = new Dictionary<string, string> { ["text_encoder"] = _qwen },
            }), Throws.ArgumentException.With.Message.Contains("text_encoder"));
        // A part the family needs, left out.
        Assert.That(() => Anima.Prepare(AnimaOptions() with
            {
                Components = new Dictionary<string, string> { ["vae"] = _wanVae },
            }), Throws.ArgumentException.With.Message.Contains("text encoder"));
        // A file that is not that part.
        Assert.That(() => Anima.Prepare(AnimaOptions() with
            {
                Components = new Dictionary<string, string> { ["text_encoder"] = _qwen, ["vae"] = _sdxl },
            }), Throws.ArgumentException.With.Message.Contains("VAE"));
        // A LoRA that is a checkpoint.
        Assert.That(() => Sdxl.Prepare(SdxlOptions() with { Loras = new[] { new LoraSpec(_sdxl, 1f) } }),
            Throws.ArgumentException.With.Message.Contains("not a LoRA"));
        // A file that is gone.
        Assert.That(() => Sdxl.Prepare(SdxlOptions() with { CheckpointPath = _folder.File("gone.safetensors") }),
            Throws.InstanceOf<FileNotFoundException>());
    }

    [Test]
    public void TheNumbersHaveRanges()
    {
        Assert.That(() => Sdxl.Prepare(SdxlOptions() with { Steps = 0 }), Throws.ArgumentException);
        Assert.That(() => Sdxl.Prepare(SdxlOptions() with { Steps = 1001 }), Throws.ArgumentException);
        Assert.That(() => Sdxl.Prepare(SdxlOptions() with { Guidance = float.NaN }), Throws.ArgumentException);
        Assert.That(() => Sdxl.Prepare(SdxlOptions() with { Width = 0 }), Throws.ArgumentException);
    }

    [Test]
    public void TheFamilyIsToldFromTheHeader()
    {
        Assert.That(ModelFamilies.Identify(_sdxl, out _), Is.SameAs(SdxlFamily.Instance));
        Assert.That(ModelFamilies.Identify(_anima, out _), Is.SameAs(AnimaFamily.Instance));
        Assert.That(ModelFamilies.Identify(_qwen, out CheckpointInspector.CheckpointReport report), Is.Null);
        Assert.That(report.Kind, Is.EqualTo(CheckpointInspector.CheckpointKind.Unknown));
        Assert.That(ModelFamilies.Require(_anima), Is.SameAs(AnimaFamily.Instance));
        Assert.That(() => ModelFamilies.Require(_wanVae),
            Throws.InstanceOf<CheckpointNotSupportedException>()
                .With.Property("Report").Property("Kind").EqualTo(CheckpointInspector.CheckpointKind.VaeOnly));
    }

    [Test]
    public void EveryFamilyAnswersItsCapabilities()
    {
        foreach (IModelFamily family in ModelFamilies.All)
        {
            Assert.That(family.SizeAlignment % family.Codec.Latent.ScaleFactor, Is.Zero, family.Name);
            Assert.That(family.SupportsSchedule(ScheduleKind.Leading), family.Name);
            Assert.That(family.Name, Is.EqualTo(family.Name.ToLowerInvariant()));
        }
        Assert.That(SdxlFamily.Instance.SupportsLoras);
        Assert.That(AnimaFamily.Instance.SupportsLoras);
        Assert.That(AnimaFamily.Instance.SizeAlignment, Is.EqualTo(16));
        Assert.That(QwenImageFamily.Instance.SupportsLoras, Is.False);
        Assert.That(SdxlFamily.Instance.SupportsHalfCompute, Is.False);
        Assert.That(AnimaFamily.Instance.SupportsHalfCompute);
        Assert.That(QwenImageFamily.Instance.SupportsHalfCompute);
        Assert.That(QwenImageFamily.Instance.SizeAlignment, Is.EqualTo(16));
        Assert.That(QwenImageFamily.Instance.DefaultSchedule, Is.EqualTo(ScheduleKind.Linspace));
        // Tags for the two whose encoders cut and weight the prompt, sentences
        // for the one whose template takes it as written.
        Assert.That(SdxlFamily.Instance.PromptStyle, Is.EqualTo(PromptStyle.Tags));
        Assert.That(AnimaFamily.Instance.PromptStyle, Is.EqualTo(PromptStyle.Tags));
        Assert.That(QwenImageFamily.Instance.PromptStyle, Is.EqualTo(PromptStyle.Sentences));
    }

    /// <summary>A caller that runs only some parts of the family is asked
    /// only for their files: the codec alone wants the VAE, the denoiser
    /// alone wants nothing beyond the checkpoint. A file that is given is
    /// still checked whatever runs.</summary>
    [Test]
    public void PrepareAsksForTheFilesOfThePartsThatRun()
    {
        Assert.That(AnimaFamily.TextEncoderComponent.Part, Is.EqualTo(PipelineParts.Conditioner));
        Assert.That(AnimaFamily.VaeComponent.Part, Is.EqualTo(PipelineParts.Codec));
        Assert.That(SdxlFamily.VaeComponent.Part, Is.EqualTo(PipelineParts.Codec));

        GenerationOptions bare = AnimaOptions() with { Components = new Dictionary<string, string>() };
        Assert.That(() => Anima.Prepare(bare), Throws.ArgumentException.With.Message.Contains("required"));
        Assert.That(() => Anima.Prepare(bare, PipelineParts.Codec),
            Throws.ArgumentException.With.Message.Contains("VAE"));
        Assert.That(() => Anima.Prepare(bare, PipelineParts.Conditioner | PipelineParts.Denoiser),
            Throws.ArgumentException.With.Message.Contains("text encoder"));
        Assert.That(Anima.Prepare(bare, PipelineParts.Denoiser).Prediction, Is.SameAs(FlowPrediction.Instance));
        Assert.That(Anima.Prepare(AnimaOptions() with
        {
            Components = new Dictionary<string, string> { [AnimaFamily.VaeComponent.Id] = _wanVae },
        }, PipelineParts.Codec).Prediction, Is.SameAs(FlowPrediction.Instance));
        Assert.That(() => Anima.Prepare(AnimaOptions() with
        {
            Components = new Dictionary<string, string> { [AnimaFamily.VaeComponent.Id] = _qwen },
        }, PipelineParts.Denoiser), Throws.ArgumentException.With.Message.Contains("VAE"), "given, so checked");
    }
}
