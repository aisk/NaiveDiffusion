namespace NaiveDiffusion.Sampling;

/// <summary>The noise levels a model was trained at: one sigma per training
/// timestep, and the two lookups between them. This is the part of a
/// scheduler that belongs to the model rather than to the sampler — SDXL's
/// scaled-linear betas, a flow model's straight line — and the spacing
/// (<see cref="ScheduleKind"/>) and the solvers are written against it.</summary>
public abstract class NoiseSchedule
{
    /// <summary>The noise level at each training timestep, rising.</summary>
    protected abstract double[] TrainSigmas { get; }

    public int TrainTimesteps => TrainSigmas.Length;

    public double SigmaMin => TrainSigmas[0];

    public double SigmaMax => TrainSigmas[TrainTimesteps - 1];

    /// <summary>The Align Your Steps table for this model, lowest noise last,
    /// or null where nobody has solved for one.</summary>
    public virtual double[]? AlignedNoiseLevels => null;

    /// <summary>These levels rescaled so that the last training timestep is
    /// pure noise (Lin et al. 2024, "Common Diffusion Noise Schedules and
    /// Sample Steps are Flawed"). Stable Diffusion's betas leave a little
    /// signal at the top of the schedule, and a model trained on them never
    /// sees a start that is noise alone; it learns to expect the average
    /// image to be in there, which is why it cannot make a very dark or a
    /// very bright one. A model finetuned with the fix — a <c>ztsnr</c>
    /// marker in the file — was trained on this table instead, and has to
    /// be sampled off it: the level sitting at a timestep is what the model
    /// was told that timestep means.
    ///
    /// The arithmetic is ComfyUI's, done on sqrt(alpha_cumprod): shift so
    /// the top is zero, scale so the bottom is what it was, and set the top
    /// to a small number instead of zero so that the sigma there is large
    /// rather than infinite. The Align Your Steps table carries over as
    /// it is, which is also what ComfyUI does.</summary>
    public NoiseSchedule WithZeroTerminalSnr() => new ZeroTerminalSnrSchedule(this);

    private sealed class ZeroTerminalSnrSchedule : NoiseSchedule
    {
        private const double TerminalAlphaProduct = 4.8973451890853435e-08;

        private readonly double[] _trainSigmas;
        private readonly double[]? _alignedNoiseLevels;

        public ZeroTerminalSnrSchedule(NoiseSchedule original)
        {
            double[] sigmas = original.TrainSigmas;
            int count = sigmas.Length;
            var alphaRoots = new double[count];
            for (int t = 0; t < count; t++)
            {
                alphaRoots[t] = Math.Sqrt(1.0 / (sigmas[t] * sigmas[t] + 1.0));
            }
            double bottom = alphaRoots[0];
            double top = alphaRoots[count - 1];
            _trainSigmas = new double[count];
            for (int t = 0; t < count; t++)
            {
                double root = (alphaRoots[t] - top) * bottom / (bottom - top);
                double alphaProduct = t == count - 1 ? TerminalAlphaProduct : root * root;
                _trainSigmas[t] = Math.Sqrt((1.0 - alphaProduct) / alphaProduct);
            }
            _alignedNoiseLevels = original.AlignedNoiseLevels;
        }

        protected override double[] TrainSigmas => _trainSigmas;

        public override double[]? AlignedNoiseLevels => _alignedNoiseLevels;
    }

    /// <summary>The noise level at a continuous timestep, linearly interpolated
    /// between the training levels.</summary>
    public double SigmaAt(double timestep)
    {
        double[] sigmas = TrainSigmas;
        int low = Math.Clamp((int)Math.Floor(timestep), 0, TrainTimesteps - 1);
        int high = Math.Min(low + 1, TrainTimesteps - 1);
        double fraction = timestep - low;
        return sigmas[low] + (sigmas[high] - sigmas[low]) * fraction;
    }

    /// <summary>The continuous training timestep a noise level sits at — the
    /// inverse of <see cref="SigmaAt"/>, interpolated in log-sigma space the
    /// way k-diffusion does it. The training sigmas rise monotonically, so a
    /// walk from the low end finds the bracket.</summary>
    public double TimestepAt(double sigma)
    {
        double[] sigmas = TrainSigmas;
        double target = Math.Log(sigma);
        int low = 0;
        while (low < TrainTimesteps - 2 && Math.Log(sigmas[low + 1]) < target)
        {
            low++;
        }
        double from = Math.Log(sigmas[low]);
        double to = Math.Log(sigmas[low + 1]);
        return Math.Clamp(low + (target - from) / (to - from), 0, TrainTimesteps - 1);
    }
}

/// <summary>The variance-preserving schedule under "scaled_linear" betas —
/// linear in sqrt-space — as Stable Diffusion's published configs specify it.
/// The noise level per training timestep comes out of the cumulative alpha
/// product.</summary>
public sealed class VpScaledLinearSchedule : NoiseSchedule
{
    /// <summary>SDXL's published config: a thousand steps from 0.00085 to
    /// 0.012, with the Align Your Steps table NVIDIA solved for on it.</summary>
    public static readonly VpScaledLinearSchedule Sdxl = new(0.00085, 0.012, 1000,
        alignedNoiseLevels: new[]
        {
            14.6146412293, 6.3184485287, 3.7681790315, 2.1811480769, 1.3405244945,
            0.8620721141, 0.5550693289, 0.3798540708, 0.2332364134, 0.1114188177,
            0.0291671582,
        });

    private readonly double[] _trainSigmas;
    private readonly double[]? _alignedNoiseLevels;

    /// <param name="alignedNoiseLevels">The Align Your Steps table: eleven
    /// numbers describing a ten-step run, its ends the training range's ends
    /// and everything between them the part that was optimized.</param>
    public VpScaledLinearSchedule(double betaStart, double betaEnd, int trainTimesteps,
        double[]? alignedNoiseLevels = null)
    {
        _trainSigmas = new double[trainTimesteps];
        double alphaProduct = 1.0;
        for (int t = 0; t < trainTimesteps; t++)
        {
            double sqrtBeta = Math.Sqrt(betaStart) +
                (Math.Sqrt(betaEnd) - Math.Sqrt(betaStart)) * t / (trainTimesteps - 1);
            alphaProduct *= 1.0 - sqrtBeta * sqrtBeta;
            _trainSigmas[t] = Math.Sqrt((1.0 - alphaProduct) / alphaProduct);
        }
        _alignedNoiseLevels = alignedNoiseLevels;
    }

    protected override double[] TrainSigmas => _trainSigmas;

    public override double[]? AlignedNoiseLevels => _alignedNoiseLevels;
}

/// <summary>The straight line of a rectified-flow model, with the timestep
/// shift the larger transformers are trained under: at training time
/// t in (0, 1] the sample is (1 − σ)·x₀ + σ·ε with σ = shift·t / (1 + (shift −
/// 1)·t), so the noise level a timestep stands for is pushed towards the
/// noisy end — shift 3 is what Anima and the Wan family use, shift 1 the
/// plain line. The table is ComfyUI's ModelSamplingDiscreteFlow: a thousand
/// timesteps from 1/1000 to 1, σ_max exactly 1, and the model reads the
/// level itself rather than an index — see <see cref="FlowPrediction"/>.</summary>
public sealed class FlowShiftSchedule : NoiseSchedule
{
    private readonly double[] _trainSigmas;

    public FlowShiftSchedule(double shift, int trainTimesteps = 1000)
    {
        _trainSigmas = new double[trainTimesteps];
        for (int i = 0; i < trainTimesteps; i++)
        {
            double t = (i + 1.0) / trainTimesteps;
            _trainSigmas[i] = shift * t / (1.0 + (shift - 1.0) * t);
        }
    }

    protected override double[] TrainSigmas => _trainSigmas;
}
