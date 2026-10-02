using NaiveDiffusion.Sampling;

namespace NaiveDiffusion.Tests;

/// <summary>The noise schedules and the samplers over them. The schedules are
/// checked against the numbers the ecosystem agrees on — SDXL's sigma range,
/// diffusers' leading timesteps, the Align Your Steps table — and the
/// samplers against the one thing every correct sampler has to do: with a
/// model that predicts exactly — the noise, or the velocity — land on the
/// clean latent.</summary>
public class SchedulersTests
{
    /// <summary>The base model's sampling: scaled-linear betas, epsilon prediction.</summary>
    private static readonly ModelSampling Sdxl =
        new(VpScaledLinearSchedule.Sdxl, EpsilonPrediction.Instance);

    /// <summary>SDXL's noise range under its scaled-linear betas, the
    /// ends of the Align Your Steps table.</summary>
    private const double SigmaMax = 14.6146412293, SigmaMin = 0.0291671582;

    private static readonly ScheduleKind[] Schedules =
        (ScheduleKind[])Enum.GetValues(typeof(ScheduleKind));

    private static readonly SamplerKind[] Samplers =
        (SamplerKind[])Enum.GetValues(typeof(SamplerKind));

    [Test]
    public void LinspaceSpansTheTrainingRange()
    {
        var scheduler = new EulerScheduler(ScheduleKind.Linspace, Sdxl);
        scheduler.SetTimesteps(5);
        Assert.That(scheduler.Timesteps, Is.EqualTo(new[] { 999.0, 749.25, 499.5, 249.75, 0.0 }).Within(1e-9));
        Assert.That(scheduler.Sigmas[0], Is.EqualTo(SigmaMax).Within(1e-4));
        Assert.That(scheduler.Sigmas[4], Is.EqualTo(SigmaMin).Within(1e-4));
        Assert.That(scheduler.Sigmas[5], Is.Zero);

        // One step is the noisy end alone, not 0/0.
        scheduler.SetTimesteps(1);
        Assert.That(scheduler.Timesteps, Is.EqualTo(new[] { 999.0 }));
        Assert.That(scheduler.Sigmas, Is.EqualTo(new[] { (float)SigmaMax, 0f }).Within(1e-4));
    }

    [Test]
    public void LeadingIsDiffusersDefault()
    {
        // stride 50, offset 1: 951, 901, …, 1.
        var scheduler = new EulerScheduler(ScheduleKind.Leading, Sdxl);
        scheduler.SetTimesteps(20);
        Assert.That(scheduler.Timesteps,
            Is.EqualTo(Enumerable.Range(0, 20).Select(i => 951.0 - 50 * i)));
        Assert.That(scheduler.InitNoiseSigma,
            Is.EqualTo(Math.Sqrt(scheduler.Sigmas[0] * (double)scheduler.Sigmas[0] + 1)).Within(1e-6));
    }

    [Test]
    public void KarrasIsEvenInTheSeventhRoot()
    {
        var scheduler = new EulerScheduler(ScheduleKind.Karras, Sdxl);
        scheduler.SetTimesteps(10);
        Assert.That(scheduler.Sigmas[0], Is.EqualTo(SigmaMax).Within(1e-4));
        Assert.That(scheduler.Sigmas[9], Is.EqualTo(SigmaMin).Within(1e-4));
        Assert.That(scheduler.Sigmas[10], Is.Zero);
        double[] roots = scheduler.Sigmas.Take(10).Select(sigma => Math.Pow(sigma, 1.0 / 7)).ToArray();
        for (int i = 1; i < roots.Length; i++)
        {
            Assert.That(roots[i] - roots[i - 1], Is.EqualTo(roots[1] - roots[0]).Within(1e-6));
        }
        Assert.That(scheduler.InitNoiseSigma, Is.EqualTo(scheduler.Sigmas[0]));
    }

    [Test]
    public void ExponentialIsEvenInLogSigma()
    {
        var scheduler = new EulerScheduler(ScheduleKind.Exponential, Sdxl);
        scheduler.SetTimesteps(8);
        double[] logs = scheduler.Sigmas.Take(8).Select(sigma => Math.Log(sigma)).ToArray();
        for (int i = 1; i < logs.Length; i++)
        {
            Assert.That(logs[i] - logs[i - 1], Is.EqualTo(logs[1] - logs[0]).Within(1e-5));
        }
    }

    [Test]
    public void AlignYourStepsAtTenStepsIsTheTable()
    {
        double[] table =
        {
            14.6146412293, 6.3184485287, 3.7681790315, 2.1811480769, 1.3405244945,
            0.8620721141, 0.5550693289, 0.3798540708, 0.2332364134, 0.1114188177,
        };
        var scheduler = new EulerScheduler(ScheduleKind.AlignYourSteps, Sdxl);
        scheduler.SetTimesteps(10);
        Assert.That(scheduler.Sigmas.Take(10), Is.EqualTo(table).Within(1e-5));
    }

    [Test]
    public void EverySchedule_TimestepsFollowTheSigmas()
    {
        foreach (ScheduleKind schedule in Schedules)
        {
            var scheduler = new EulerScheduler(schedule, Sdxl);
            scheduler.SetTimesteps(12);
            Assert.That(scheduler.StepCount, Is.EqualTo(12));
            Assert.That(scheduler.Sigmas, Has.Length.EqualTo(13));
            for (int i = 1; i < 12; i++)
            {
                Assert.That(scheduler.Sigmas[i], Is.LessThan(scheduler.Sigmas[i - 1]), schedule.ToString());
                Assert.That(scheduler.Timesteps[i], Is.LessThan(scheduler.Timesteps[i - 1]), schedule.ToString());
            }
            // Leading starts a stride short of the top by design; the rest
            // put their first level at the training range's end.
            Assert.That(scheduler.Timesteps[0],
                Is.InRange(schedule == ScheduleKind.Leading ? 900.0 : 998.0, 999.0), schedule.ToString());
            // Align Your Steps stretches a ten-entry table, so at any other
            // count its last level sits short of the table's end.
            Assert.That(scheduler.Timesteps[^1],
                Is.InRange(0.0, schedule == ScheduleKind.AlignYourSteps ? 30.0 : 1.0), schedule.ToString());
        }
    }

    [Test]
    public void SkipKeepsTheTailOfTheSchedule()
    {
        var scheduler = new EulerScheduler(ScheduleKind.Karras, Sdxl);
        scheduler.SetTimesteps(20);
        double[] timesteps = scheduler.Timesteps.ToArray();
        float[] sigmas = scheduler.Sigmas.ToArray();

        Assert.That(scheduler.Skip(0.5f), Is.EqualTo(10));
        Assert.That(scheduler.Timesteps, Is.EqualTo(timesteps[10..]));
        Assert.That(scheduler.Sigmas, Is.EqualTo(sigmas[10..]));

        // Anything above zero keeps at least one step; 1 keeps them all.
        scheduler.SetTimesteps(20);
        Assert.That(scheduler.Skip(0.001f), Is.EqualTo(1));
        Assert.That(scheduler.Sigmas, Has.Length.EqualTo(2));
        scheduler.SetTimesteps(20);
        Assert.That(scheduler.Skip(1f), Is.EqualTo(20));
    }

    [Test]
    public void EverySamplerLandsOnTheCleanLatentGivenPerfectNoisePredictions()
    {
        // The model output an epsilon model would give if it knew the clean
        // latent: the noise the sample carries, scaled to the level. Each
        // sampler's update must then walk down to that latent exactly —
        // Euler because the derivative points straight at it, the DPM++
        // solvers because every prediction is the same point, the ancestral
        // ones because the noise they add is noise the next prediction sees.
        EverySamplerLandsOnTheCleanLatent(EpsilonPrediction.Instance, VpScaledLinearSchedule.Sdxl,
            (sample, clean, sigma) => (sample - clean) / sigma);
    }

    [Test]
    public void EverySamplerLandsOnTheCleanLatentGivenPerfectVelocityPredictions()
    {
        // The velocity is sqrt(alpha) * noise - sqrt(1 - alpha) * signal, with
        // sqrt(alpha) = 1 / sqrt(sigma^2 + 1) under the variance-preserving
        // scaling. The same walk must land on the same latent, on the
        // published table and on the zero-terminal-SNR one whose top level
        // is thousands: the sample starts at that level and the arithmetic
        // has to survive it.
        foreach (NoiseSchedule levels in new[]
                 {
                     VpScaledLinearSchedule.Sdxl,
                     VpScaledLinearSchedule.Sdxl.WithZeroTerminalSnr(),
                 })
        {
            EverySamplerLandsOnTheCleanLatent(VelocityPrediction.Instance, levels,
                (sample, clean, sigma) =>
                {
                    float epsilon = (sample - clean) / sigma;
                    return (epsilon - sigma * clean) / MathF.Sqrt(sigma * sigma + 1);
                });
        }
    }

    private static void EverySamplerLandsOnTheCleanLatent(Prediction prediction,
        NoiseSchedule levels, Func<float, float, float, float> perfectOutput)
    {
        float[] clean = Enumerable.Range(0, 64).Select(i => MathF.Sin(i * 0.37f)).ToArray();
        foreach (SamplerKind kind in Samplers)
        {
            foreach (ScheduleKind schedule in Schedules)
            {
                var noise = new GaussianGenerator(seed: 3);
                SigmaScheduler scheduler = SigmaScheduler.Create(kind, noise, schedule,
                    new ModelSampling(levels, prediction));
                scheduler.SetTimesteps(15);

                float[] sample = noise.Fill(clean.Length);
                for (int i = 0; i < sample.Length; i++)
                {
                    sample[i] = clean[i] + sample[i] * scheduler.Sigmas[0];
                }
                for (int step = 0; step < scheduler.StepCount; step++)
                {
                    float sigma = scheduler.Sigmas[step];
                    float[] output = sample.Select((x, i) => perfectOutput(x, clean[i], sigma))
                        .ToArray();
                    Assert.That(scheduler.PredictedSample(output, step, sample),
                        Is.EqualTo(clean).Within(2e-3f), $"{kind} {schedule} preview at {step}");
                    scheduler.Step(output, step, sample);
                }
                Assert.That(sample, Is.EqualTo(clean).Within(2e-3f), $"{kind} {schedule}");
            }
        }
    }

    [Test]
    public void ZeroTerminalSnrLeavesTheBottomAndSendsTheTopToNoise()
    {
        // ComfyUI's rescale: alpha_cumprod at the top pinned to 4.9e-8, so
        // sigma_max is about 4519 rather than 14.6, and the bottom level is
        // what it was. The Align Your Steps table carries over unchanged.
        NoiseSchedule levels = VpScaledLinearSchedule.Sdxl.WithZeroTerminalSnr();
        Assert.That(levels.TrainTimesteps, Is.EqualTo(1000));
        Assert.That(levels.SigmaMin, Is.EqualTo(SigmaMin).Within(1e-6));
        Assert.That(levels.SigmaMax, Is.EqualTo(4519.0).Within(1.0));
        Assert.That(levels.SigmaAt(500), Is.GreaterThan(VpScaledLinearSchedule.Sdxl.SigmaAt(500)));
        Assert.That(levels.AlignedNoiseLevels, Is.EqualTo(VpScaledLinearSchedule.Sdxl.AlignedNoiseLevels));
        for (int t = 1; t < 1000; t++)
        {
            Assert.That(levels.SigmaAt(t), Is.GreaterThan(levels.SigmaAt(t - 1)), $"at {t}");
        }
        // The inverse lookup still lands where it left from, top included.
        Assert.That(levels.TimestepAt(levels.SigmaAt(731.5)), Is.EqualTo(731.5).Within(1e-3));
        Assert.That(levels.TimestepAt(levels.SigmaMax), Is.EqualTo(999));
    }

    [Test]
    public void ScaleModelInputNormalizesTheVariance()
    {
        var scheduler = new EulerScheduler(ScheduleKind.Leading, Sdxl);
        scheduler.SetTimesteps(4);
        float sigma = scheduler.Sigmas[1];
        float[] scaled = scheduler.ScaleModelInput(new[] { 1f, -2f }, 1);
        float expected = 1f / MathF.Sqrt(sigma * sigma + 1);
        Assert.That(scaled, Is.EqualTo(new[] { expected, -2 * expected }).Within(1e-6f));
    }

    [Test]
    public void GaussianGeneratorIsSeededAndStandard()
    {
        float[] first = new GaussianGenerator(11).Fill(1000);
        float[] again = new GaussianGenerator(11).Fill(1000);
        Assert.That(again, Is.EqualTo(first));
        Assert.That(new GaussianGenerator(12).Fill(1000), Is.Not.EqualTo(first));

        float[] many = new GaussianGenerator(5).Fill(200_000);
        double mean = many.Average();
        double variance = many.Select(x => (x - mean) * (x - mean)).Average();
        Assert.That(mean, Is.EqualTo(0).Within(0.01));
        Assert.That(variance, Is.EqualTo(1).Within(0.02));
    }
}
