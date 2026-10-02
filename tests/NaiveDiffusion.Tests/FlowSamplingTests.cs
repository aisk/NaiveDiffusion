using NaiveDiffusion.Sampling;
using NaiveDiffusion.Vae;

namespace NaiveDiffusion.Tests;

/// <summary>Rectified flow as the sampler sees it — the shifted line of
/// noise levels, the prediction that reads a velocity — and the standardized
/// latent space beside it. Checked against ComfyUI's ModelSamplingDiscreteFlow
/// and CONST arithmetic, which is what Anima was trained to be sampled with.</summary>
public class FlowSamplingTests
{
    [Test]
    public void TheShiftedLineEndsAtOneAndBendsTowardsNoise()
    {
        var levels = new FlowShiftSchedule(3.0);
        Assert.That(levels.TrainTimesteps, Is.EqualTo(1000));
        Assert.That(levels.SigmaMax, Is.EqualTo(1.0).Within(1e-12));
        // t = 1/1000 through the shift: 3t / (1 + 2t).
        Assert.That(levels.SigmaMin, Is.EqualTo(3 * 0.001 / (1 + 2 * 0.001)).Within(1e-12));
        // Half way along the timesteps sits well above half the noise.
        Assert.That(levels.SigmaAt(499), Is.EqualTo(0.75).Within(1e-3));
        // Shift 1 is the plain line.
        Assert.That(new FlowShiftSchedule(1.0).SigmaAt(499), Is.EqualTo(0.5).Within(1e-12));
    }

    /// <summary>A flow model starts from noise alone, which is sigma 1: the
    /// spacing a family calls its default has to begin there, at any step
    /// count. SDXL's leading spacing on Anima's levels does not — 0.98 on
    /// twenty steps, 0.95 on eight — which is why the default is the
    /// family's to say.</summary>
    [Test]
    public void AFlowFamilysDefaultSpacingStartsFromPureNoise()
    {
        NaiveDiffusion.Pipeline.IModelFamily anima = NaiveDiffusion.Models.Anima.AnimaFamily.Instance;
        foreach (int steps in new[] { 1, 8, 20, 50 })
        {
            var scheduler = new EulerScheduler(anima.DefaultSchedule,
                new ModelSampling(new FlowShiftSchedule(3.0), FlowPrediction.Instance));
            scheduler.SetTimesteps(steps);
            Assert.That(scheduler.InitNoiseSigma, Is.EqualTo(1f), $"{steps} steps");
        }

        var leading = new EulerScheduler(ScheduleKind.Leading,
            new ModelSampling(new FlowShiftSchedule(3.0), FlowPrediction.Instance));
        leading.SetTimesteps(8);
        Assert.That(leading.InitNoiseSigma, Is.LessThan(0.96f), "what the default must not be");
    }

    [Test]
    public void ASamplerRunsTheLineDownToCleanData()
    {
        var levels = new FlowShiftSchedule(3.0);
        var scheduler = new EulerScheduler(ScheduleKind.Linspace,
            new ModelSampling(levels, FlowPrediction.Instance));
        scheduler.SetTimesteps(8);
        Assert.That(scheduler.Sigmas[0], Is.EqualTo(1f));
        Assert.That(scheduler.Sigmas[^1], Is.EqualTo(0f));
        Assert.That(scheduler.InitNoiseSigma, Is.EqualTo(1f));

        // A model that knows the clean latent x0 exactly predicts the flow
        // noise − x0 at every level; the line then lands on x0.
        float[] clean = { 0.3f, -1.2f, 2.0f, 0.05f };
        float[] noise = { 1.0f, -0.5f, 0.25f, -2.0f };
        var sample = (float[])noise.Clone();
        for (int step = 0; step < scheduler.StepCount; step++)
        {
            float[] input = scheduler.ScaleModelInput(sample, step);
            Assert.That(input, Is.EqualTo(sample), "a flow model reads the sample as it is");
            float sigma = scheduler.Sigmas[step];
            // x = (1 − σ) x0 + σ ε, so ε = (x − (1 − σ) x0) / σ.
            var velocity = new float[4];
            for (int i = 0; i < 4; i++)
            {
                float epsilon = (sample[i] - (1 - sigma) * clean[i]) / sigma;
                velocity[i] = epsilon - clean[i];
            }
            Assert.That(scheduler.PredictedSample(velocity, step, sample), Is.EqualTo(clean).Within(1e-4f));
            scheduler.Step(velocity, step, sample);
        }
        Assert.That(sample, Is.EqualTo(clean).Within(1e-4f));
    }

    [Test]
    public void TheAncestralStepKeepsTheSignalOnTheLine()
    {
        // After a step to σ_next the sample has to be (1 − σ_next)·x0 plus
        // noise of scale σ_next, whatever noise the step put back. The
        // variance-exploding split Stable Diffusion uses leaves the signal at
        // (1 − σ_down)·x0 instead, a twentieth too strong on the first step
        // here and worse later, which twenty steps turn into an image blown
        // out to white; ComfyUI's rectified-flow euler a does not.
        var levels = new FlowShiftSchedule(3.0);
        var noise = new GaussianGenerator(seed: 11);
        var scheduler = new EulerAncestralScheduler(noise, ScheduleKind.Linspace,
            new ModelSampling(levels, FlowPrediction.Instance));
        scheduler.SetTimesteps(8);

        float[] clean = Enumerable.Range(0, 4096).Select(i => 10 * MathF.Sin(i * 0.37f)).ToArray();
        float[] sample = noise.Fill(clean.Length);
        for (int step = 0; step < scheduler.StepCount; step++)
        {
            float sigma = scheduler.Sigmas[step];
            float sigmaNext = scheduler.Sigmas[step + 1];
            var velocity = new float[clean.Length];
            for (int i = 0; i < clean.Length; i++)
            {
                float epsilon = (sample[i] - (1 - sigma) * clean[i]) / sigma;
                velocity[i] = epsilon - clean[i];
            }
            scheduler.Step(velocity, step, sample);

            double[] residual = sample.Select((x, i) => (double)(x - (1 - sigmaNext) * clean[i])).ToArray();
            double mean = residual.Average();
            double rms = Math.Sqrt(residual.Sum(r => r * r) / residual.Length);
            Assert.That(mean, Is.EqualTo(0.0).Within(0.1), $"signal off the line after step {step}");
            Assert.That(rms, Is.EqualTo(sigmaNext).Within(0.05 * sigmaNext + 1e-6),
                $"noise scale after step {step}");
        }
        Assert.That(sample, Is.EqualTo(clean).Within(1e-4f));
    }

    [Test]
    public void AReferenceIsNoisedAlongTheLineNotOnTopOfIt()
    {
        float[] reference = { 1f, 2f };
        float[] noise = { -1f, 0.5f };
        Assert.That(FlowPrediction.Instance.NoiseReference(reference, noise, 0.25f),
            Is.EqualTo(new[] { 0.75f - 0.25f, 1.5f + 0.125f }).Within(1e-6f));
        // Stable Diffusion's convention is untouched: reference + σ·noise.
        Assert.That(EpsilonPrediction.Instance.NoiseReference(reference, noise, 0.25f),
            Is.EqualTo(new[] { 0.75f, 2.125f }).Within(1e-6f));
    }

    [Test]
    public void AStandardizedLatentGoesInAndComesBackPerChannel()
    {
        var space = new StandardizedLatentSpace(Channels: 2, ScaleFactor: 8,
            Means: new[] { 1f, -1f }, Deviations: new[] { 2f, 4f },
            PreviewColors: new[] { new[] { 1f, 0f, 0f }, new[] { 0f, 1f, 0f } },
            PreviewBias: new float[3]);
        float[] latent = { 3f, 5f, -1f, 7f };   // two channels of two cells
        space.ToModelScale(latent);
        Assert.That(latent, Is.EqualTo(new[] { 1f, 2f, 0f, 2f }));
        Assert.That(space.FromModelScale(latent), Is.EqualTo(new[] { 3f, 5f, -1f, 7f }));
    }

    [Test]
    public void WanLatentIsSixteenChannelsAtEightPixels()
    {
        Assert.That(WanVae.Latent.Channels, Is.EqualTo(16));
        Assert.That(WanVae.Latent.ScaleFactor, Is.EqualTo(8));
        var latent = new float[16 * 4];
        WanVae.Latent.ToModelScale(latent);
        // Zero on the VAE's scale is minus the mean over the deviation.
        Assert.That(latent[0], Is.EqualTo(0.7571f / 2.8184f).Within(1e-5f));
    }
}
