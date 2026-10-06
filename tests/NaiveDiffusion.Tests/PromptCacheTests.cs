using NaiveDiffusion.Models.Anima;
using NaiveDiffusion.Models.QwenImage;
using NaiveDiffusion.Models.Sdxl;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Tests.Fixtures;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Tests;

/// <summary>What the prompt cache keeps and for how long, and what a text
/// is kept under: the encoder's files and settings, and nothing a run sets
/// that the encoder does not read.</summary>
public class PromptCacheTests
{
    private static PromptCache.Key Key(string text) => new("encoder", text, 0);

    private static float[][] Megabytes(int count) => new[] { new float[count << 18] };

    [Test]
    public void WhatWasAddedIsFoundAgain()
    {
        var cache = new PromptCache();
        float[][] encoded = { new float[8], new float[2] };
        cache.Add(Key("1girl"), encoded);

        Assert.That(cache.Find(Key("1girl")), Is.SameAs(encoded));
        Assert.That(cache.HeldBytes, Is.EqualTo(40));
        Assert.That(cache.Find(Key("1boy")), Is.Null);
        Assert.That(cache.Find(new PromptCache.Key("another encoder", "1girl", 0)), Is.Null);
        Assert.That(cache.Find(new PromptCache.Key("encoder", "1girl", 2)), Is.Null);
    }

    [Test]
    public void TheTextUsedLongestAgoGoesFirst()
    {
        var cache = new PromptCache(3L << 20);
        cache.Add(Key("a"), Megabytes(1));
        cache.Add(Key("b"), Megabytes(1));
        cache.Add(Key("c"), Megabytes(1));
        Assert.That(cache.Find(Key("a")), Is.Not.Null);

        cache.Add(Key("d"), Megabytes(1));

        Assert.That(cache.Find(Key("b")), Is.Null);
        Assert.That(cache.Find(Key("a")), Is.Not.Null);
        Assert.That(cache.Find(Key("c")), Is.Not.Null);
        Assert.That(cache.Find(Key("d")), Is.Not.Null);
        Assert.That(cache.HeldBytes, Is.EqualTo(3L << 20));
    }

    [Test]
    public void AddingUnderAKeyAgainReplacesWhatWasThere()
    {
        var cache = new PromptCache();
        cache.Add(Key("a"), Megabytes(2));
        float[][] second = Megabytes(1);
        cache.Add(Key("a"), second);

        Assert.That(cache.Find(Key("a")), Is.SameAs(second));
        Assert.That(cache.Count, Is.EqualTo(1));
        Assert.That(cache.HeldBytes, Is.EqualTo(1L << 20));
    }

    [Test]
    public void TheCapacityBoundsWhatIsHeld()
    {
        var cache = new PromptCache(2L << 20);
        cache.Add(Key("too large"), Megabytes(3));
        Assert.That(cache.Count, Is.Zero);

        cache.Add(Key("a"), Megabytes(1));
        cache.Add(Key("b"), Megabytes(1));
        cache.CapacityBytes = 1L << 20;
        Assert.That(cache.Find(Key("a")), Is.Null);
        Assert.That(cache.Find(Key("b")), Is.Not.Null);

        cache.CapacityBytes = 0;
        Assert.That(cache.Count, Is.Zero);
        cache.Add(Key("c"), Megabytes(1));
        Assert.That(cache.Find(Key("c")), Is.Null);

        Assert.That(() => cache.CapacityBytes = -1, Throws.InstanceOf<ArgumentOutOfRangeException>());
    }

    [Test]
    public void TheStampFollowsTheEncoderAndNotTheRun()
    {
        using var folder = new TempFolder();
        var files = new FakeSafetensors(folder);
        string checkpoint = files.Write("anima", FakeSafetensors.AnimaDit());
        string textEncoder = files.Write("qwen", FakeSafetensors.Qwen3TextEncoder());
        string other = files.Write("qwen-other", FakeSafetensors.Qwen3TextEncoder());
        string vae = files.Write("wanvae", FakeSafetensors.WanVae());
        string lora = files.Write("lora", FakeSafetensors.WanVae());
        GenerationOptions Options(string encoder, string vaePath) => new()
        {
            Prompt = "1girl", CheckpointPath = checkpoint, Width = 1024, Height = 1024,
            Components = new Dictionary<string, string>
            {
                [AnimaFamily.TextEncoderComponent.Id] = encoder,
                [AnimaFamily.VaeComponent.Id] = vaePath,
            },
        };
        GenerationOptions options = Options(textEncoder, vae);
        string stamp = PromptCache.EncoderStamp(AnimaFamily.Instance, options);

        // What the text encoder never reads.
        Assert.That(PromptCache.EncoderStamp(AnimaFamily.Instance, options with
        {
            Prompt = "1boy", Negative = "lowres", Seed = 7, Steps = 8, Guidance = 1f, Width = 512, Height = 768,
        }), Is.EqualTo(stamp));
        Assert.That(PromptCache.EncoderStamp(AnimaFamily.Instance, Options(textEncoder, other)),
            Is.EqualTo(stamp));

        // What it does.
        Assert.That(PromptCache.EncoderStamp(AnimaFamily.Instance, Options(other, vae)),
            Is.Not.EqualTo(stamp));
        Assert.That(PromptCache.EncoderStamp(AnimaFamily.Instance, options with
        {
            Loras = new[] { new LoraSpec(lora, 0.8f) },
        }), Is.Not.EqualTo(stamp));
        Assert.That(PromptCache.EncoderStamp(SdxlFamily.Instance, options with { ClipSkip = 1 }),
            Is.Not.EqualTo(PromptCache.EncoderStamp(SdxlFamily.Instance, options)));
    }

    /// <summary>The families' requests: one text per branch, the negative
    /// first, each with the length the run's longest prompt sets.</summary>
    [Test]
    public void ARunAsksForOneTextPerBranch()
    {
        string longPrompt = string.Join(", ", Enumerable.Repeat("lowres, bad anatomy, blurry", 30));
        var options = new GenerationOptions { Prompt = "1girl", Negative = longPrompt };

        Assert.That(SdxlFamily.Instance.Conditioner.Requests(options, 2), Is.EqualTo(new[]
        {
            new PromptRequest(longPrompt, SdxlTextEncoders.ChunkCount(longPrompt)),
            new PromptRequest("1girl", SdxlTextEncoders.ChunkCount(longPrompt)),
        }));
        Assert.That(SdxlTextEncoders.ChunkCount(longPrompt), Is.GreaterThan(1));
        Assert.That(SdxlFamily.Instance.Conditioner.Requests(options with { Negative = "" }, 2),
            Is.EqualTo(new[] { new PromptRequest("", 1), new PromptRequest("1girl", 1) }));

        Assert.That(AnimaFamily.Instance.Conditioner.Requests(options, 1),
            Is.EqualTo(new[] { new PromptRequest("1girl", AnimaPrompt.ContextLength) }));
        Assert.That(AnimaFamily.Instance.Conditioner.Requests(options, 2).Select(request => request.Text),
            Is.EqualTo(new[] { longPrompt, "1girl" }));

        Assert.That(QwenImageFamily.Instance.Conditioner.Requests(options, 1),
            Is.EqualTo(new[] { new PromptRequest("1girl", 0) }));
        Assert.That(QwenImageFamily.Instance.Conditioner.Requests(options, 2),
            Is.EqualTo(new[] { new PromptRequest(longPrompt, 0), new PromptRequest("1girl", 0) }));
    }
}
