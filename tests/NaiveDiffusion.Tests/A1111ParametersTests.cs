using System.Globalization;
using NaiveDiffusion.Images;
using NaiveDiffusion.Sampling;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Models.Sdxl;
using NaiveDiffusion.Text;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Tests;

/// <summary>The parameters block a saved PNG carries: A1111's layout, English
/// names, invariant numbers, and a prompt that pastes back in.</summary>
public class A1111ParametersTests
{
    [Test]
    [SetCulture("de-DE")]
    public void LaysOutLikeA1111WhateverTheCulture()
    {
        var options = new GenerationOptions
        {
            Prompt = "1girl, smile",
            Negative = "lowres",
            Width = 832,
            Height = 1216,
            Steps = 15,
            Guidance = 5.5f,
            Sampler = SamplerKind.EulerAncestral,
            Schedule = ScheduleKind.Karras,
            CheckpointPath = @"C:\models\illustriousXL_v1.safetensors",
            Loras = new[] { new LoraSpec(@"C:\loras\watercolor_style.safetensors", 0.8f) },
        };

        string text = A1111Parameters.Format(options, seed: 7, SdxlFamily.Instance);

        Assert.That(text.Split('\n'), Is.EqualTo(new[]
        {
            "1girl, smile <lora:watercolor_style:0.8>",
            "Negative prompt: lowres",
            "Steps: 15, Sampler: Euler a, Schedule type: Karras, CFG scale: 5.5, Seed: 7, " +
            "Size: 832x1216, Model: illustriousXL_v1, Clip skip: 2, Version: NaiveDiffusion",
        }));
    }

    [Test]
    public void TheVaeGoesInByNameUnderItsA1111Key()
    {
        var options = new GenerationOptions
        {
            CheckpointPath = @"C:\models\m.safetensors",
            Components = new Dictionary<string, string>
            {
                [SdxlFamily.VaeComponent.Id] = @"C:\models\vae\sdxl_vae_fp16_fix.safetensors",
            },
            ClipSkip = 1,
        };
        string text = A1111Parameters.Format(options, 1, SdxlFamily.Instance);
        Assert.That(text, Does.Contain(", Model: m, VAE: sdxl_vae_fp16_fix, Clip skip: 1,"));
        // A file not given is not named.
        Assert.That(A1111Parameters.Format(options with { Components = new Dictionary<string, string>() },
            1, SdxlFamily.Instance), Does.Not.Contain("VAE"));
    }

    [Test]
    public void ThePromptLinePastesBackIn()
    {
        var options = new GenerationOptions
        {
            Prompt = "1girl",
            CheckpointPath = "model.safetensors",
            Loras = new[]
            {
                new LoraSpec("a.safetensors", 1f),
                new LoraSpec(@"sub\b.safetensors", 0.65f),
            },
        };
        string prompt = A1111Parameters.Format(options, 0, SdxlFamily.Instance).Split('\n')[0];
        string stripped = LoraTags.Extract(prompt, out IReadOnlyList<LoraTag> loras);
        Assert.That(stripped, Is.EqualTo("1girl"));
        Assert.That(loras, Is.EqualTo(new[] { new LoraTag("a", 1f), new LoraTag("b", 0.65f) }));
    }

    [Test]
    public void ImageToImageRecordsTheStrength()
    {
        var options = new GenerationOptions
        {
            Width = 8,
            Height = 8,
            CheckpointPath = "m.safetensors",
            ReferenceImage = new ImageResult(8, 8, new byte[8 * 8 * 3]),
            Strength = 0.45f,
        };
        Assert.That(A1111Parameters.Format(options, 1, SdxlFamily.Instance),
            Does.Contain(", Denoising strength: 0.45,"));
        Assert.That(A1111Parameters.Format(options with { ReferenceImage = null }, 1, SdxlFamily.Instance),
            Does.Not.Contain("Denoising"));
    }
}
