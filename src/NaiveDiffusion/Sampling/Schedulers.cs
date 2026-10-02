namespace NaiveDiffusion.Sampling;

/// <summary>How the noise levels a run passes through are chosen. The first two
/// space the training timesteps and read the levels off them; Karras places the
/// levels directly and works out which timesteps they correspond to.</summary>
public enum ScheduleKind
{
    /// <summary>Evenly spaced from 0, shifted by one: SDXL's published
    /// config, and its default here.</summary>
    Leading,
    /// <summary>The whole range from the last training timestep down to 0, which
    /// ComfyUI and A1111 call the normal schedule.</summary>
    Linspace,
    /// <summary>Karras et al. 2022: evenly spaced in sigma^(1/7) instead, which
    /// spends more of the budget at the low noise levels where the image is
    /// actually settled. The karras half of "DPM++ 2M Karras".</summary>
    Karras,
    /// <summary>Evenly spaced in log sigma — the same idea as Karras with the
    /// bend taken out, and the mildest of the three.</summary>
    Exponential,
    /// <summary>Align Your Steps (NVIDIA, 2024): levels solved for offline
    /// against SDXL itself rather than derived from a formula, published as ten
    /// numbers and stretched to whatever step count is asked for. The point of
    /// it is the low step counts.</summary>
    AlignYourSteps,
}

public enum SamplerKind
{
    /// <summary>Plain Euler: one multiply-add per step, same latent in, same
    /// image out.</summary>
    Euler,
    /// <summary>Euler with the noise put back that the step took out.</summary>
    EulerAncestral,
    /// <summary>DPM-Solver++ 2M, the second-order multistep solver most SDXL
    /// workflows default to.</summary>
    DpmPlusPlus2M,
    /// <summary>The same solver on the stochastic differential equation: it
    /// pulls harder towards the prediction and puts noise back to make up the
    /// difference, which keeps detail coming in late in a run.</summary>
    DpmPlusPlus2MSde,
}

/// <summary>The noise levels a run passes through and the parts every sampler
/// shares. The schedule of sigmas is the content; a sampler is just how one
/// integrates from one of them to the next. Which levels a model was trained
/// at is the <see cref="NoiseSchedule"/>'s, and what its output means is the
/// <see cref="Prediction"/>'s; the family hands the two over as one
/// <see cref="ModelSampling"/>, and the spacing (<see cref="ScheduleKind"/>)
/// and the solvers below are written against the two of them alone.</summary>
public abstract class SigmaScheduler
{
    private const int StepsOffset = 1;

    private readonly NoiseSchedule _levels;
    private readonly Prediction _prediction;
    private readonly ScheduleKind _schedule;

    public double[] Timesteps { get; private set; } = Array.Empty<double>();
    public float[] Sigmas { get; private set; } = Array.Empty<float>();

    protected SigmaScheduler(ScheduleKind schedule, ModelSampling sampling)
    {
        _schedule = schedule;
        _levels = sampling.Levels;
        _prediction = sampling.Prediction;
        if (schedule == ScheduleKind.AlignYourSteps && _levels.AlignedNoiseLevels is null)
        {
            throw new ArgumentException("no Align Your Steps table exists for this model");
        }
    }

    private int TrainTimesteps => _levels.TrainTimesteps;

    /// <summary>The sampler a kind names, ready to run. <paramref name="noise"/>
    /// is the stream the starting latent came from; an ancestral sampler keeps
    /// drawing from it.</summary>
    public static SigmaScheduler Create(SamplerKind kind, GaussianGenerator noise,
        ScheduleKind schedule, ModelSampling sampling) => kind switch
    {
        SamplerKind.EulerAncestral => new EulerAncestralScheduler(noise, schedule, sampling),
        SamplerKind.DpmPlusPlus2M => new DpmPlusPlus2MScheduler(schedule, sampling),
        SamplerKind.DpmPlusPlus2MSde => new DpmPlusPlus2MSdeScheduler(noise, schedule, sampling),
        _ => new EulerScheduler(schedule, sampling),
    };

    public int StepCount => Timesteps.Length;

    public void SetTimesteps(int steps)
    {
        Timesteps = new double[steps];
        // A trailing zero so the final step integrates all the way to clean data.
        Sigmas = new float[steps + 1];

        // These three place the noise levels first. The timesteps are then
        // whatever those levels happen to sit at, since the UNet still has to be
        // told the noise level it is looking at.
        if (_schedule is ScheduleKind.Karras or ScheduleKind.Exponential
            or ScheduleKind.AlignYourSteps)
        {
            for (int i = 0; i < steps; i++)
            {
                double sigma = _schedule switch
                {
                    ScheduleKind.Karras => KarrasSigma(i, steps),
                    ScheduleKind.Exponential => ExponentialSigma(i, steps),
                    _ => AlignedSigma(i, steps),
                };
                Sigmas[i] = (float)sigma;
                Timesteps[i] = _levels.TimestepAt(sigma);
            }
            return;
        }

        if (_schedule == ScheduleKind.Linspace)
        {
            // One step is the last training timestep alone, not 0/0.
            for (int i = 0; i < steps; i++)
            {
                Timesteps[i] = steps == 1
                    ? TrainTimesteps - 1
                    : (TrainTimesteps - 1) * (1.0 - (double)i / (steps - 1));
            }
        }
        else
        {
            int stride = TrainTimesteps / steps;
            for (int i = 0; i < steps; i++)
            {
                Timesteps[i] = (steps - 1 - i) * stride + StepsOffset;
            }
        }
        for (int i = 0; i < steps; i++)
        {
            Sigmas[i] = (float)_levels.SigmaAt(Timesteps[i]);
        }
    }

    /// <summary>Evenly spaced in sigma^(1/rho) between the training range's
    /// ends. rho = 7 is what the paper settled on and what everyone ships: it
    /// bends the schedule towards the low noise levels.</summary>
    private double KarrasSigma(int index, int steps)
    {
        const double rho = 7.0;
        double low = Math.Pow(_levels.SigmaMin, 1.0 / rho);
        double high = Math.Pow(_levels.SigmaMax, 1.0 / rho);
        return Math.Pow(high + Ramp(index, steps) * (low - high), rho);
    }

    /// <summary>Evenly spaced in log sigma, ends included.</summary>
    private double ExponentialSigma(int index, int steps)
    {
        double low = Math.Log(_levels.SigmaMin);
        double high = Math.Log(_levels.SigmaMax);
        return Math.Exp(high + Ramp(index, steps) * (low - high));
    }

    /// <summary>The published ten-step table, log-linearly interpolated to the
    /// step count asked for — what ComfyUI's AYS node does. A run of exactly
    /// ten steps lands on the table entries themselves.</summary>
    private double AlignedSigma(int index, int steps)
    {
        double[] levels = _levels.AlignedNoiseLevels!;
        int last = levels.Length - 1;
        double position = (double)last * index / steps;
        int low = Math.Min((int)position, last - 1);
        double fraction = position - low;
        double from = Math.Log(levels[low]);
        double to = Math.Log(levels[low + 1]);
        return Math.Exp(from + fraction * (to - from));
    }

    /// <summary>0 at the first step, 1 at the last. A single-step run stays at
    /// the noisy end.</summary>
    private static double Ramp(int index, int steps) =>
        steps > 1 ? (double)index / (steps - 1) : 0.0;

    /// <summary>Drop the schedule's first steps so that only the last
    /// <paramref name="strength"/> of them run, which is how an image-to-image
    /// run starts part-way down: the reference latent is noised to the first
    /// remaining level (<see cref="Sigmas"/>[0]) and sampled from there, and
    /// everything that would have happened above that level is taken as
    /// already done by the reference. Diffusers' and A1111's convention — the
    /// schedule is the one the step count asks for, truncated by index — so a
    /// strength tuned there means the same thing here. Strength 1 keeps every
    /// step; anything above 0 keeps at least one, so the run always moves.
    /// Returns how many steps remain.</summary>
    public int Skip(float strength)
    {
        int steps = Timesteps.Length;
        int keep = Math.Clamp((int)(steps * strength), 1, steps);
        int drop = steps - keep;
        Timesteps = Timesteps[drop..];
        Sigmas = Sigmas[drop..];
        return keep;
    }

    /// <summary>The standard deviation to draw the starting latent with.</summary>
    public float InitNoiseSigma
    {
        get
        {
            // Leading follows diffusers and folds in the variance-preserving
            // term; the other schedules are k-diffusion's, where the level is
            // the scale. The two differ by 0.2% at SDXL's sigma_max either way.
            return _prediction.InitNoiseSigma(Sigmas[0],
                variancePreserving: _schedule == ScheduleKind.Leading);
        }
    }

    /// <summary>Normalize a latent to the input variance the model expects.</summary>
    public float[] ScaleModelInput(float[] sample, int step)
    {
        float scale = _prediction.ModelInputScale(Sigmas[step]);
        var scaled = new float[sample.Length];
        for (int i = 0; i < sample.Length; i++)
        {
            scaled[i] = sample[i] * scale;
        }
        return scaled;
    }

    /// <summary>One step, from the noise level of <paramref name="step"/> to the
    /// next one, in place on <paramref name="sample"/>.</summary>
    public abstract void Step(float[] modelOutput, int step, float[] sample);

    /// <summary>The image as it currently stands: where the model says the clean
    /// latent is, from this step's noisy one. This is what a preview shows — the
    /// sample itself still carries the noise the run has not removed yet, and
    /// looks like static until the very end.</summary>
    public float[] PredictedSample(float[] modelOutput, int step, float[] sample) =>
        Denoise(modelOutput, Sigmas[step], sample);

    /// <summary>Whether a multistep solver should take its second-order term at
    /// this step. It should not on the one that lands on the schedule's lowest
    /// noise level: leading and linspace put five to eight times the log-sigma
    /// distance into that step as into the one before it, and extrapolating
    /// across a gap that much longer than the one it was measured over
    /// overshoots — visibly so once an SDE puts noise back on top of it. This is
    /// diffusers' lower_order_final, which is on by default there for the same
    /// reason. A schedule with even steps, Karras among them, is unaffected
    /// either way.</summary>
    protected bool SecondOrderAt(int step) => step + 2 < StepCount;

    /// <summary>The clean latent the model is pointing at from here.</summary>
    protected float[] Denoise(float[] modelOutput, double sigma, float[] sample) =>
        _prediction.Denoise(modelOutput, sigma, sample);

    /// <summary>d(sample)/d(sigma) here — what an Euler step moves along.</summary>
    protected float[] Derivative(float[] modelOutput, double sigma, float[] sample) =>
        _prediction.Derivative(modelOutput, sigma, sample);

    /// <summary>What a stochastic sampler does to a sample it has integrated
    /// down to <paramref name="sigmaDown"/> to bring it back up to
    /// <paramref name="sigmaNext"/>: multiply by the first, add unit noise
    /// times the second. The prediction decides; see
    /// <see cref="Prediction.Renoise"/>.</summary>
    protected (double Rescale, double NoiseScale) Renoise(double sigmaDown, double sigmaNext) =>
        _prediction.Renoise(sigmaDown, sigmaNext);
}

/// <summary>The deterministic Euler sampler. One step is one multiply-add.</summary>
public sealed class EulerScheduler : SigmaScheduler
{
    public EulerScheduler(ScheduleKind schedule, ModelSampling sampling)
        : base(schedule, sampling)
    {
    }

    public override void Step(float[] modelOutput, int step, float[] sample)
    {
        float[] derivative = Derivative(modelOutput, Sigmas[step], sample);
        float delta = Sigmas[step + 1] - Sigmas[step];
        for (int i = 0; i < sample.Length; i++)
        {
            sample[i] += derivative[i] * delta;
        }
    }
}

/// <summary>Euler with the noise put back that the step just took out, which is
/// the euler_a that ComfyUI and A1111 default to. Sampling stops being a
/// function of the starting latent alone, so the caller hands over the
/// generator that drew that latent and sampling keeps drawing from it. It has
/// to be the same stream: a generator restarted from the seed would hand the
/// first step the starting latent's own noise back, and instead of cancelling
/// against what the step removed it would pile on top of it.</summary>
public sealed class EulerAncestralScheduler : SigmaScheduler
{
    private readonly GaussianGenerator _noise;

    public EulerAncestralScheduler(GaussianGenerator noise, ScheduleKind schedule, ModelSampling sampling)
        : base(schedule, sampling)
    {
        _noise = noise;
    }

    public override void Step(float[] modelOutput, int step, float[] sample)
    {
        double sigma = Sigmas[step];
        double sigmaNext = Sigmas[step + 1];
        float[] derivative = Derivative(modelOutput, sigma, sample);

        // The variance-preserving split: integrate past sigma_next down to
        // sigma_down, which is sigma_next² / sigma, then add fresh noise of
        // scale sigma_up back.
        double up = sigmaNext * Math.Sqrt(Math.Max(sigma * sigma - sigmaNext * sigmaNext, 0.0)) / sigma;
        double down = Math.Sqrt(Math.Max(sigmaNext * sigmaNext - up * up, 0.0));
        float delta = (float)(down - sigma);

        // How the sample gets back up is the prediction's call. For the
        // Stable Diffusion models that noise is sigma_up itself, and the
        // update is one single-precision expression of its own, so their
        // outputs do not pass through the rescale; a flow model puts the
        // signal back on its line first, and needs less noise for it.
        (double rescale, double noiseScale) = Renoise(down, sigmaNext);
        if (rescale == 1.0)
        {
            float upScale = (float)up;
            for (int i = 0; i < sample.Length; i++)
            {
                sample[i] += derivative[i] * delta + _noise.Next() * upScale;
            }
            return;
        }
        for (int i = 0; i < sample.Length; i++)
        {
            sample[i] = (float)(rescale * (sample[i] + derivative[i] * delta) + _noise.Next() * noiseScale);
        }
    }
}

/// <summary>DPM-Solver++ 2M: the exponential integrator for the semi-linear
/// diffusion ODE, second order by extrapolating over the previous step's
/// prediction rather than by evaluating the model twice. It therefore costs
/// exactly what Euler costs and converges in noticeably fewer steps, which is
/// why dpmpp_2m is what most SDXL workflows are set to.</summary>
public sealed class DpmPlusPlus2MScheduler : SigmaScheduler
{
    private float[]? _lastDenoised;
    private double _lastStepSize;

    public DpmPlusPlus2MScheduler(ScheduleKind schedule, ModelSampling sampling)
        : base(schedule, sampling)
    {
    }

    public override void Step(float[] modelOutput, int step, float[] sample)
    {
        double sigma = Sigmas[step];
        double sigmaNext = Sigmas[step + 1];
        float[] denoised = Denoise(modelOutput, sigma, sample);

        // The solver integrates in half-log-SNR, where the step size is
        // log(sigma / sigma_next) and exp(-step size) is the ratio itself. The
        // last step runs to sigma 0, an infinite step size whose limit is the
        // prediction alone.
        if (sigmaNext <= 0.0)
        {
            Array.Copy(denoised, sample, sample.Length);
            return;
        }
        double stepSize = Math.Log(sigma / sigmaNext);
        double shrink = sigmaNext / sigma;

        // Second order from the second step on: extrapolate the prediction over
        // the two most recent ones. The weight is 1/(2r) for r the previous
        // step's length over this one's — a step that follows a short one
        // leans further out than one that follows a long one.
        float[]? last = SecondOrderAt(step) ? _lastDenoised : null;
        double weight = last is null ? 0.0 : stepSize / (2.0 * _lastStepSize);
        for (int i = 0; i < sample.Length; i++)
        {
            double prediction = last is null
                ? denoised[i]
                : (1.0 + weight) * denoised[i] - weight * last[i];
            sample[i] = (float)(shrink * sample[i] + (1.0 - shrink) * prediction);
        }

        _lastDenoised = denoised;
        _lastStepSize = stepSize;
    }
}

/// <summary>DPM-Solver++ 2M on the stochastic differential equation instead of
/// the ODE, at the full noise level (eta = 1) ComfyUI's dpmpp_2m_sde runs at.
/// The deterministic pull towards the prediction is squared — sigma_next/sigma
/// twice over — and noise of exactly the variance that removes is put back, so
/// the run keeps being handed new detail to resolve instead of only sharpening
/// what the first steps happened to lay down. Like euler a it draws from the
/// starting latent's own stream.</summary>
public sealed class DpmPlusPlus2MSdeScheduler : SigmaScheduler
{
    private readonly GaussianGenerator _noise;
    private float[]? _lastDenoised;
    private double _lastStepSize;

    public DpmPlusPlus2MSdeScheduler(GaussianGenerator noise, ScheduleKind schedule, ModelSampling sampling)
        : base(schedule, sampling)
    {
        _noise = noise;
    }

    public override void Step(float[] modelOutput, int step, float[] sample)
    {
        double sigma = Sigmas[step];
        double sigmaNext = Sigmas[step + 1];
        float[] denoised = Denoise(modelOutput, sigma, sample);

        if (sigmaNext <= 0.0)
        {
            Array.Copy(denoised, sample, sample.Length);
            return;
        }
        double stepSize = Math.Log(sigma / sigmaNext);
        double shrink = sigmaNext / sigma;

        // At eta = 1 every exponential in the update collapses to a power of
        // that ratio: the sample decays by its square, the prediction takes the
        // rest — which is an Euler step down to sigma_next² / sigma, the same
        // sigma_down as euler a's — and the noise carries the variance that
        // leaves, after the prediction's rescale, as in euler a.
        double keep = shrink * shrink;
        double take = 1.0 - keep;
        (double rescale, double noise) = Renoise(sigmaNext * shrink, sigmaNext);
        float noiseScale = (float)noise;

        // The midpoint correction, the same second-order term as the ODE
        // solver, scaled by the drift this step actually applies.
        float[]? last = SecondOrderAt(step) ? _lastDenoised : null;
        double weight = last is null ? 0.0 : 0.5 * take * stepSize / _lastStepSize;
        for (int i = 0; i < sample.Length; i++)
        {
            double next = keep * sample[i] + take * denoised[i];
            if (last is not null)
            {
                next += weight * (denoised[i] - last[i]);
            }
            sample[i] = (float)(rescale * next + _noise.Next() * noiseScale);
        }

        _lastDenoised = denoised;
        _lastStepSize = stepSize;
    }
}
