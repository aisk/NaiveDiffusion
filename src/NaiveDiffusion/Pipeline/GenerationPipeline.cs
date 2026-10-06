using System.Diagnostics;
using NaiveDiffusion.Dml;
using NaiveDiffusion.Images;
using NaiveDiffusion.Models;
using NaiveDiffusion.Sampling;
using NaiveDiffusion.Tensors;
using NaiveDiffusion.Text;
using NaiveDiffusion.Vae;
using NaiveDiffusion.Weights;
using static NaiveDiffusion.Models.CheckpointInspector;

namespace NaiveDiffusion.Pipeline;

/// <summary>Generate an image from a prompt, end to end, with whichever
/// <see cref="IModelFamily"/> this was built around. The stages run in
/// sequence — the text encoders turn the prompt into conditioning, the
/// diffusion model predicts and removes noise over a few dozen steps, and the
/// VAE decoder turns the final latent into pixels — and each is built only
/// when needed and released once done: all of them resident at once do not
/// comfortably fit. The order, the memory discipline between the stages, the
/// progress reporting and the cache are what live here; everything that
/// knows what a model looks like is the family's.</summary>
public sealed class GenerationPipeline
{
    public const string NoCheckpointMessage = "pick a single-file checkpoint (.safetensors) first";

    /// <summary>How a stage tells the caller where it is. The run-wide fields —
    /// the clock and the image count — are the reporter's own.</summary>
    private delegate void Reporter(Stage stage, int step = 0, int total = 0, int image = 0,
        ImageResult? preview = null);

    private readonly IModelFamily _family;

    public GenerationPipeline(IModelFamily family)
    {
        _family = family;
    }

    public IModelFamily Family => _family;

    /// <summary>Run the whole pipeline for one image, seeded from the options.</summary>
    public ImageResult Generate(DmlDevice device, GenerationOptions options,
        IProgress<Snapshot>? progress = null, CancellationToken cancellation = default,
        DenoiserCache? cache = null, PromptCache? prompts = null)
        => Generate(device, options, new[] { options.Seed }, progress, cancellation,
            cache: cache, prompts: prompts)[0];

    /// <summary>Run the whole pipeline once per seed. Everything but the
    /// sampling is shared: the prompts are encoded once, the diffusion model is
    /// built once, and the VAE decoder is built once. Under classifier-free
    /// guidance the model runs twice per step: once on the prompt and once on
    /// the negative prompt; the difference between the two is what gets
    /// amplified. The model and the decoder do not comfortably fit together,
    /// so every seed samples before anything decodes; <paramref name="onImage"/>
    /// hears about each image as its decode completes, on the calling thread.
    ///
    /// <paramref name="cache"/> extends that sharing past the end of the run:
    /// given one, the model is borrowed from it if a previous run left a
    /// matching one there, and offered back at the end. Without one the model
    /// is built at the start and released before decoding.
    ///
    /// <paramref name="prompts"/> does the same for the text side: given
    /// one, a prompt an earlier run encoded with the same text encoder is
    /// taken from it, and what this run encodes is left there. Without one
    /// every run opens the text encoder.</summary>
    public ImageResult[] Generate(DmlDevice device, GenerationOptions options,
        IReadOnlyList<int> seeds, IProgress<Snapshot>? progress = null,
        CancellationToken cancellation = default, Action<int, ImageResult>? onImage = null,
        DenoiserCache? cache = null, PromptCache? prompts = null)
    {
        if (seeds.Count == 0)
        {
            throw new ArgumentException("at least one seed", nameof(seeds));
        }
        ModelSampling sampling = Prepare(options);

        var stopwatch = Stopwatch.StartNew();
        void Report(Stage stage, int step = 0, int total = 0, int image = 0,
            ImageResult? preview = null)
        {
            // The device labels every video memory reading with this, so a peak
            // can be blamed on the stage that caused it.
            device.CurrentStage = stage.Label();
            progress?.Report(new Snapshot(stage, step, total, stopwatch.Elapsed,
                image, seeds.Count, preview));
        }

        ImageResult[] results;
        try
        {
            Conditioning conditioning = Encode(device, options, prompts, Report, cancellation);
            cancellation.ThrowIfCancellationRequested();

            float[]? reference = null;
            if (options.ReferenceImage is ImageResult image)
            {
                Report(Stage.EncodingImage);
                reference = _family.Codec.EncodeReference(device, options, image);
                ReleaseHostWeights();
                device.Sample("VAE encoder released");
                cancellation.ThrowIfCancellationRequested();
            }

            float[][] sampled = Sample(device, options, sampling, seeds, conditioning, reference,
                cache, Report, cancellation);

            cancellation.ThrowIfCancellationRequested();
            results = Decode(device, options, sampled, Report, cancellation, onImage);
        }
        finally
        {
            // The pool keeps the last dispatch's scratch — a decoded tile's, a
            // gigabyte and a half, or several without tiling — until something
            // asks for another size. A device can outlive the run, and an
            // idle one is not to sit on video memory nothing will read;
            // the next run's first dispatch allocates what it needs either way.
            device.ReleasePooledBuffers();
        }
        device.Sample("pooled buffers released");

        Report(Stage.Done);
        return results;
    }

    /// <summary>Everything that can be known about a run before a weight is
    /// loaded, checked, and the sampling the run will use. What every family
    /// shares is checked here — the sizes and the step count, that the files
    /// exist and are what they were given as, that every option applies to
    /// this family — and the family adds its own. Throws on the first
    /// problem, so a run that cannot finish never starts: a missing VAE
    /// would otherwise surface only after every image had been sampled.</summary>
    public ModelSampling Prepare(GenerationOptions options, PipelineParts parts = PipelineParts.All)
    {
        if (options.Steps < 1)
        {
            throw new ArgumentException("steps must be at least 1");
        }
        if (!float.IsFinite(options.Guidance))
        {
            throw new ArgumentException("the guidance scale must be a number");
        }
        int alignment = _family.SizeAlignment;
        if (options.Width <= 0 || options.Height <= 0
            || options.Width % alignment != 0 || options.Height % alignment != 0)
        {
            throw new ArgumentException(
                $"each side must be a positive multiple of {alignment}");
        }
        if (options.ReferenceImage is ImageResult reference
            && (reference.Width != options.Width || reference.Height != options.Height))
        {
            throw new ArgumentException(
                $"the reference image is {reference.Width}x{reference.Height}; " +
                $"it must be {options.Width}x{options.Height}, the size being generated");
        }
        if (options.ReferenceImage is not null
            && (!(options.Strength > 0.0f) || options.Strength > 1.0f))
        {
            throw new ArgumentException("strength must be above 0 and at most 1");
        }
        if (!_family.SupportsSchedule(options.Schedule))
        {
            throw new ArgumentException(
                $"{options.Schedule} has no table for {_family.Name}; pick another schedule");
        }

        // The options a family may not have: clip skip where the text side
        // has one layer to read, LoRAs where nothing merges them, a weight
        // storage or a compute precision its model is not built for.
        if (_family.MaxClipSkip == 0)
        {
            if (options.ClipSkip != GenerationOptions.DefaultClipSkip)
            {
                throw new ArgumentException(
                    $"{_family.Name} reads its text encoder's last layer only; clip skip does not apply");
            }
        }
        else if (options.ClipSkip < 1 || options.ClipSkip > _family.MaxClipSkip)
        {
            throw new ArgumentException(
                $"clip skip must be between 1 and {_family.MaxClipSkip}");
        }
        if (!_family.SupportsLoras && options.Loras.Count > 0)
        {
            throw new ArgumentException($"LoRAs are not supported for {_family.Name}");
        }
        if (!_family.SupportsWeights(options.DenoiserWeights))
        {
            throw new ArgumentException(
                $"{_family.Name} cannot store its weights as {options.DenoiserWeights.Label()}; it takes " +
                string.Join(" or ", Enum.GetValues<WeightStorage>()
                    .Where(_family.SupportsWeights).Select(weights => weights.Label())));
        }
        if (!_family.SupportsCompute(options.DenoiserCompute))
        {
            throw new ArgumentException(
                $"{_family.Name} cannot compute at {options.DenoiserCompute.Label()}; it computes at " +
                string.Join(" or ", Enum.GetValues<ComputePrecision>()
                    .Where(_family.SupportsCompute).Select(compute => compute.Label())));
        }

        // A tag names a file, and only the caller knows where files are: it
        // resolves the tags (LoraLibrary.Resolve) before handing the prompt
        // over. One still here would reach the text encoder as words.
        if (LoraTags.Contains(options.Prompt))
        {
            LoraTags.Extract(options.Prompt, out IReadOnlyList<LoraTag> tags);
            throw new ArgumentException(
                $"the prompt still names a LoRA, <lora:{tags[0].Name}>; " +
                "resolve the tag to a file or take it out before generating");
        }
        // A reference names a snippet, and only the caller has the library:
        // it expands the prompts (PromptTemplate.Expand) before handing them
        // over. One still here would reach the text encoder as a word in
        // braces.
        foreach ((string label, string text) in new[] { ("prompt", options.Prompt), ("negative prompt", options.Negative) })
        {
            if (PromptTemplate.References(text) is { Count: > 0 } names)
            {
                throw new ArgumentException(
                    $"the {label} still refers to a snippet, {PromptTemplate.Format(names[0])}; " +
                    "expand it or take it out before generating");
            }
        }
        foreach (LoraSpec lora in options.Loras)
        {
            if (!File.Exists(lora.Path))
            {
                throw new FileNotFoundException("the LoRA file is gone", lora.Path);
            }
            if (!float.IsFinite(lora.Weight))
            {
                throw new ArgumentException($"{lora.Path}: the LoRA weight must be a number");
            }
            // The header says whether a file is a LoRA at all; whether its
            // layers fit this checkpoint only the merge can say, and it does.
            if (Inspect(lora.Path).Kind != CheckpointKind.Lora)
            {
                throw new ArgumentException($"{lora.Path}: not a LoRA");
            }
        }

        // The files: the checkpoint is this family's and whole, every part
        // given is one the family has and is the file it was given as, and
        // every part the family needs was given. All from headers — a page
        // fault each.
        if (!File.Exists(options.CheckpointPath))
        {
            throw new FileNotFoundException(NoCheckpointMessage, options.CheckpointPath);
        }
        CheckpointReport report = CheckpointInspector.Inspect(options.CheckpointPath);
        if (!_family.Runs(report))
        {
            throw new CheckpointNotSupportedException(report,
                $"{options.CheckpointPath}: {CheckpointInspector.Explain(report)}");
        }
        foreach ((string id, string path) in options.Components)
        {
            ModelComponent component = _family.Components.FirstOrDefault(part => part.Id == id)
                ?? throw new ArgumentException($"{_family.Name} has no component named {id}");
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"the {id} file is gone", path);
            }
            ComponentVerdict verdict = _family.InspectComponent(component, path);
            if (verdict != ComponentVerdict.Accepted)
            {
                throw new ArgumentException($"{path}: {verdict.Explain(component)}");
            }
        }
        // Every part the family needs was given — of the parts that will
        // run: a VAE roundtrip does not ask for the text encoder, and one
        // forward pass on conditioning from a file asks for neither.
        foreach (ModelComponent component in _family.Components)
        {
            if (component.Required && (component.Part & parts) != 0
                && options.ComponentPath(component.Id) is null)
            {
                throw new ArgumentException($"a {component.Name} file is required");
            }
        }

        _family.Validate(options);

        ModelSampling sampling = _family.Sampling(report, options);
        if (options.Schedule == ScheduleKind.AlignYourSteps
            && sampling.Levels.AlignedNoiseLevels is null)
        {
            throw new ArgumentException("no Align Your Steps table exists for this model");
        }
        // The step count arrives as the caller typed it, and the schedulers
        // divide by it.
        if (options.Steps > sampling.Levels.TrainTimesteps)
        {
            throw new ArgumentException(
                $"steps must be between 1 and {sampling.Levels.TrainTimesteps}");
        }
        return sampling;
    }

    /// <summary>The run's prompts as conditioning. Each text the
    /// <paramref name="prompts"/> cache has from an earlier run is taken
    /// from it, and the text encoder is opened only when one is missing —
    /// not at all for a run that changed nothing the encoder reads.</summary>
    private Conditioning Encode(DmlDevice device, GenerationOptions options, PromptCache? prompts,
        Reporter report, CancellationToken cancellation)
    {
        IConditioner conditioner = _family.Conditioner;
        int branches = _family.Guidance.Branches(options.Guidance);
        IReadOnlyList<PromptRequest> requests = conditioner.Requests(options, branches);
        string stamp = prompts is null ? "" : PromptCache.EncoderStamp(_family, options);
        var encoded = new float[requests.Count][][];
        for (int i = 0; i < encoded.Length; i++)
        {
            encoded[i] = prompts?.Find(new PromptCache.Key(stamp, requests[i]))!;
        }

        if (Array.TrueForAll(encoded, found => found is not null))
        {
            report(Stage.EncodingPrompt);
            device.Sample("prompts reused");
        }
        else
        {
            // Two stages for the text side: reading a large encoder's weights
            // is seconds on its own, and the status should say so rather than
            // claim the prompt is being encoded.
            report(Stage.LoadingTextEncoder);
            using (IPromptEncoder encoder = conditioner.Open(options))
            {
                cancellation.ThrowIfCancellationRequested();
                report(Stage.EncodingPrompt);
                for (int i = 0; i < encoded.Length; i++)
                {
                    if (encoded[i] is not null)
                    {
                        continue;
                    }
                    // The same text twice in one run — an empty prompt
                    // against an empty negative — is encoded once.
                    int earlier = requests.Take(i).ToList().IndexOf(requests[i]);
                    encoded[i] = earlier >= 0 ? encoded[earlier] : encoder.Encode(requests[i]);
                    prompts?.Add(new PromptCache.Key(stamp, requests[i]), encoded[i]);
                }
            }
            ReleaseHostWeights();
            // Sampled after every handover, where the interesting question is not
            // how much a stage took but whether it gave it all back.
            device.Sample("text encoders released");
        }
        return conditioner.Condition(options, encoded).CheckedFor(branches);
    }

    private (int Height, int Width, int Length) LatentShape(GenerationOptions options)
    {
        LatentSpace latent = _family.Codec.Latent;
        int height = options.Height / latent.ScaleFactor;
        int width = options.Width / latent.ScaleFactor;
        return (height, width, latent.Channels * height * width);
    }

    /// <summary>Drop the host copies the stage that just finished read in. Each
    /// model's weights arrive as arrays of gigabytes — SDXL's UNet alone is
    /// 4.78 GiB — and once DirectML has them nothing references those arrays,
    /// but they sit on the large object heap until a collection runs. The next
    /// stage's load would stack on top of them, so the handoff between stages is
    /// where the collection has to happen. Aggressive, so the segments go back
    /// to the OS as well: a plain collection frees the arrays but keeps the
    /// address space committed against the next allocation, and with LoRAs
    /// folded in that is over four gigabytes of merged copies of the UNet
    /// still counted against the process all through sampling.</summary>
    public static void ReleaseHostWeights() =>
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Aggressive, blocking: true, compacting: true);

    /// <summary>The denoiser for this run, its resident budget resolved
    /// first: the figure the options give, or one taken from what the card
    /// has free right now when they ask for that — now, because the previous
    /// run's model and the pool's buffers are gone at this point and nothing
    /// this run needs is there yet; taken any earlier it counts memory the
    /// run is about to release, and streams weights the card had room for.</summary>
    public IDenoiser BuildDenoiser(DmlDevice device, GenerationOptions options,
        Conditioning conditioning, CancellationToken cancellation)
    {
        ulong? budget = options.DenoiserResidentBytes;
        if (budget is null && options.DenoiserAutoBudget)
        {
            device.ReleasePooledBuffers();
            budget = DenoiserBudget.FromFreeMemory(device, options, _family);
            device.Sample($"denoiser budget: {budget >> 20} MiB");
        }
        IDenoiser denoiser = _family.BuildDenoiser(device, options, conditioning, budget,
            cancellation);
        ReleaseHostWeights();
        return denoiser;
    }

    /// <summary>Whether the denoiser can stay resident through the decode that
    /// follows it. Without a cache the question never comes up: the model is
    /// released the moment sampling ends, and the two stages never overlap.
    ///
    /// The margin is wide because the two ways of being wrong do not cost the
    /// same. Releasing a model that would have fit costs fifteen seconds on the
    /// next run. Keeping one that does not fit costs the run and possibly the
    /// device: an over-budget working set is served out of system memory across
    /// PCIe, and a dispatch slow enough trips the driver's timeout.
    ///
    /// Twice the estimate, then — half again was measured to be too little.
    /// A 1024² whole-image decode passed that test by 60 MiB on a 16 GiB card,
    /// ran at a peak of 11.4 GiB, and the budget itself fell from 15.2 to 13.4
    /// GiB while it did: the denominator shrinks under exactly the pressure
    /// this is deciding about. An adapter that will not say what the budget is
    /// gets a no.</summary>
    private bool FitsAlongsideDecode(DmlDevice device, GenerationOptions options)
    {
        (ulong usage, ulong budget) = device.VideoMemory();
        ulong needed = _family.Codec.EstimateDecodeBytes(options);
        return budget != 0 && usage + 2 * needed <= budget;
    }

    /// <summary>Sample one latent per seed, all through the same model. With a
    /// <paramref name="reference"/> latent every seed starts from it, noised to
    /// the level the strength picks, rather than from noise alone.</summary>
    private float[][] Sample(DmlDevice device, GenerationOptions options,
        ModelSampling sampling, IReadOnlyList<int> seeds, Conditioning conditioning,
        float[]? reference, DenoiserCache? cache, Reporter report,
        CancellationToken cancellation)
    {
        (int latentHeight, int latentWidth, int latentLength) = LatentShape(options);

        var key = DenoiserCache.Key.For(_family.Name, options, conditioning);
        // Borrowing empties the cache, and releases what was in it when it does
        // not match: the old graphs and their replacement do not fit at once.
        IDenoiser? cached = cache?.Borrow(device, key);
        IDenoiser denoiser;
        if (cached is not null && options.DenoiserResidentBytes is ulong budget &&
            budget < cached.ResidentBytes)
        {
            // The budget is not part of the key, because a model built under a
            // looser one is still right under any budget it fits. One that no
            // longer fits an explicit budget is rebuilt; where its weights
            // live was fixed when they were uploaded. An automatic budget
            // never rebuilds: the cached model fit the card when it was built,
            // and it is the largest thing this process holds.
            cached.Dispose();
            cached = null;
            device.Sample("denoiser released: over the new budget");
        }
        else if (cached is not null && cached.StreamedBytes > 0 &&
                 options.DenoiserResidentBytes is null && !options.DenoiserAutoBudget)
        {
            // No budget at all asks for every weight in video memory, which a
            // model built under a budget that streamed some of them is not:
            // reusing it would keep a run that switched low-VRAM mode off
            // crossing the bus at every step.
            cached.Dispose();
            cached = null;
            device.Sample("denoiser released: streamed, now wanted resident");
        }
        if (cached is not null)
        {
            denoiser = cached;
            device.Sample("denoiser reused");
        }
        else
        {
            report(Stage.BuildingDenoiser);
            denoiser = BuildDenoiser(device, options, conditioning, cancellation);
            device.Sample("denoiser built");
        }
        device.Sample($"denoiser: {denoiser.GraphCount} graphs, {denoiser.PersistentBytes >> 20} MiB " +
                      $"of weights, {denoiser.StreamedBytes >> 20} MiB streamed, " +
                      $"{denoiser.TemporaryBytes >> 20} MiB scratch");

        // A device failure leaves the compiled graphs in a state nothing should
        // be reused from. Cancellation does not — the model is untouched, and
        // cancelling to change a setting and going again is exactly the case
        // worth keeping it for.
        bool broken = false;
        try
        {
            var sampled = new float[seeds.Count][];
            for (int image = 0; image < seeds.Count; image++)
            {
                sampled[image] = SampleOne(denoiser, options, sampling, seeds[image],
                    conditioning, reference, latentHeight, latentWidth, latentLength, image,
                    report, cancellation);
            }
            return sampled;
        }
        catch (Exception exception) when (exception is not OperationCanceledException)
        {
            broken = true;
            throw;
        }
        finally
        {
            // A model that was offered back is the cache's to dispose — it does
            // so itself when caching was turned off while the run was in
            // flight — so only one that was never offered is disposed here.
            bool offered = cache is not null && !broken && FitsAlongsideDecode(device, options);
            if (offered && cache!.Return(device, key, denoiser))
            {
                device.Sample("denoiser kept");
            }
            else
            {
                if (!offered)
                {
                    denoiser.Dispose();
                }
                ReleaseHostWeights();
                device.Sample("denoiser released");
            }
        }
    }

    /// <summary>One image's sampling loop: the starting latent from the seed,
    /// then every step of the schedule, the denoiser run once per guidance
    /// branch and the branches combined before the sampler steps.</summary>
    private float[] SampleOne(IDenoiser denoiser, GenerationOptions options,
        ModelSampling sampling, int seed, Conditioning conditioning, float[]? reference,
        int latentHeight, int latentWidth, int latentLength, int image, Reporter report,
        CancellationToken cancellation)
    {
        LatentSpace latent = _family.Codec.Latent;
        GuidanceStrategy guidance = _family.Guidance;

        // One stream per image: the starting latent first, then whatever
        // an ancestral sampler puts back along the way.
        var noise = new GaussianGenerator(seed);
        SigmaScheduler scheduler = SigmaScheduler.Create(
            options.Sampler, noise, options.Schedule, sampling);
        scheduler.SetTimesteps(options.Steps);

        float[] latents = noise.Fill(latentLength);
        if (reference is null)
        {
            for (int i = 0; i < latents.Length; i++)
            {
                latents[i] *= scheduler.InitNoiseSigma;
            }
        }
        else
        {
            // The reference stands in for every step above the level
            // the run starts at, so the noise put on it is that
            // level's — the same draw the image would have started
            // from, on the same stream, so the seed still means what
            // it means from noise.
            scheduler.Skip(options.Strength);
            latents = sampling.Prediction.NoiseReference(reference, latents, scheduler.Sigmas[0]);
        }

        var predictions = new float[guidance.Branches(options.Guidance)][];
        for (int step = 0; step < scheduler.StepCount; step++)
        {
            cancellation.ThrowIfCancellationRequested();
            report(Stage.Sampling, step, scheduler.StepCount, image);

            float[] modelInput = scheduler.ScaleModelInput(latents, step);
            var latentTensor = HostTensor.FromFloats(modelInput,
                1, latent.Channels, latentHeight, latentWidth);
            double timestep = scheduler.Timesteps[step];

            for (int branch = 0; branch < predictions.Length; branch++)
            {
                if (branch > 0)
                {
                    cancellation.ThrowIfCancellationRequested();
                }
                predictions[branch] = denoiser.Predict(latentTensor, timestep,
                    scheduler.Sigmas[step], conditioning.Branch(branch)).ToFloats();
            }
            float[] prediction = guidance.Combine(predictions, options.Guidance);

            // Taken before the step, from the same prediction it is
            // about to apply: this is where the model says the image is
            // right now, and it costs one pass over the latent.
            var preview = new ImageResult(latentWidth, latentHeight,
                latent.PreviewRgb24(
                    scheduler.PredictedSample(prediction, step, latents),
                    latentHeight, latentWidth));
            scheduler.Step(prediction, step, latents);
            report(Stage.Sampling, step + 1, scheduler.StepCount, image, preview);
        }
        return latents;
    }

    /// <summary>Every sampled latent through the VAE decoder, in order.</summary>
    private ImageResult[] Decode(DmlDevice device, GenerationOptions options,
        float[][] sampled, Reporter report, CancellationToken cancellation,
        Action<int, ImageResult>? onImage)
    {
        var results = new ImageResult[sampled.Length];
        ILatentDecoder? decoder = null;
        try
        {
            // Reported, or the status would sit on the last sampling step for
            // the seconds the VAE takes to read and compile.
            report(Stage.LoadingDecoder);
            decoder = _family.Codec.OpenDecoder(device, options);
            for (int image = 0; image < sampled.Length; image++)
            {
                cancellation.ThrowIfCancellationRequested();
                report(Stage.Decoding, 0, 0, image);
                results[image] = new ImageResult(options.Width, options.Height,
                    Pixels.ToRgb24(decoder.Decode(sampled[image]), options.Height, options.Width));
                onImage?.Invoke(image, results[image]);
            }
        }
        finally
        {
            decoder?.Dispose();
            ReleaseHostWeights();
            device.Sample("decoder released");
        }
        return results;
    }
}
