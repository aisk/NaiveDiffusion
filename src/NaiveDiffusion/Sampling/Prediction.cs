namespace NaiveDiffusion.Sampling;

/// <summary>What the model's output means, and so how a noisy latent and that
/// output combine into the clean latent the solvers integrate towards. The
/// solvers are written against this and the sigmas alone; whether the model
/// predicts the noise, the velocity or the flow is decided here.</summary>
public abstract class Prediction
{
    /// <summary>The clean latent the model is pointing at from
    /// <paramref name="sample"/> at noise level <paramref name="sigma"/>.</summary>
    public abstract float[] Denoise(float[] modelOutput, double sigma, float[] sample);

    /// <summary>d(sample)/d(sigma) at this level — what an Euler step moves
    /// along. Mathematically (sample − denoised) / sigma; a prediction whose
    /// output is that derivative already returns it as is, without the
    /// rounding a subtraction and a division would add.</summary>
    public abstract float[] Derivative(float[] modelOutput, double sigma, float[] sample);

    /// <summary>What the sample is multiplied by before the model sees it, so
    /// that the input has the variance the model was trained on.</summary>
    public abstract float ModelInputScale(double sigma);

    /// <summary>The standard deviation to draw the starting latent with, given
    /// the schedule's highest level. <paramref name="variancePreserving"/> is
    /// diffusers' convention for its own spacing: fold the signal's unit
    /// variance in as well.</summary>
    public abstract float InitNoiseSigma(float sigmaMax, bool variancePreserving);

    /// <summary>A clean latent taken to noise level <paramref name="sigma"/>
    /// with the given draw of unit noise — where an image-to-image run starts.
    /// The variance-exploding form, reference + σ·noise, is what every
    /// Stable Diffusion means by it; a flow model interpolates instead.</summary>
    public virtual float[] NoiseReference(float[] reference, float[] noise, float sigma)
    {
        var noised = new float[reference.Length];
        for (int i = 0; i < noised.Length; i++)
        {
            noised[i] = reference[i] + noise[i] * sigma;
        }
        return noised;
    }

    /// <summary>How a sample that a stochastic sampler has integrated down to
    /// <paramref name="sigmaDown"/> gets back up to <paramref name="sigmaNext"/>:
    /// it is multiplied by the first value, then given unit noise scaled by the
    /// second. For a model whose sample is x₀ + σ·ε — k-diffusion's split, the
    /// Stable Diffusion models — nothing is rescaled and the noise supplies
    /// the variance between the two levels. A flow model's sample is
    /// (1 − σ)·x₀ + σ·ε, where the signal's weight is a function of the
    /// level too, so it overrides this.</summary>
    public virtual (double Rescale, double NoiseScale) Renoise(double sigmaDown, double sigmaNext) =>
        (1.0, Math.Sqrt(Math.Max(sigmaNext * sigmaNext - sigmaDown * sigmaDown, 0.0)));
}

/// <summary>The model predicts the flow, ε − x₀: the velocity that carries
/// the sample along the straight line x = (1 − σ)·x₀ + σ·ε from the clean
/// latent at σ = 0 to noise at σ = 1 — rectified flow, ComfyUI's CONST,
/// which Anima, Flux and SD3 all put out. The clean latent is x − σ·v, the
/// derivative along σ is v itself, and the model reads the sample as it
/// is: there is no variance-preserving rescale to undo.</summary>
public sealed class FlowPrediction : Prediction
{
    public static readonly FlowPrediction Instance = new();

    private FlowPrediction()
    {
    }

    public override float[] Denoise(float[] modelOutput, double sigma, float[] sample)
    {
        var denoised = new float[sample.Length];
        for (int i = 0; i < sample.Length; i++)
        {
            denoised[i] = (float)(sample[i] - sigma * modelOutput[i]);
        }
        return denoised;
    }

    public override float[] Derivative(float[] modelOutput, double sigma, float[] sample) =>
        modelOutput;

    public override float ModelInputScale(double sigma) => 1.0f;

    /// <summary>σ_max is 1: the starting latent is the unit noise itself.</summary>
    public override float InitNoiseSigma(float sigmaMax, bool variancePreserving) => sigmaMax;

    /// <summary>(1 − σ)·reference + σ·noise: the point on the training line.</summary>
    public override float[] NoiseReference(float[] reference, float[] noise, float sigma)
    {
        var noised = new float[reference.Length];
        float keep = 1.0f - sigma;
        for (int i = 0; i < noised.Length; i++)
        {
            noised[i] = reference[i] * keep + noise[i] * sigma;
        }
        return noised;
    }

    /// <summary>The renoising of ComfyUI's sample_euler_ancestral_RF. Left
    /// at k-diffusion's split, a step lands the signal at (1 − σ_down)·x₀
    /// where the line says (1 − σ_next)·x₀, and σ_down is well below
    /// σ_next: the signal comes out up to twice too strong on the first
    /// step, the next prediction reads that as a brighter image, and twenty
    /// steps of it blow every highlight out to white and tint the whole
    /// image. So the sample is multiplied by (1 − σ_next) / (1 − σ_down) to
    /// put the signal where the line wants it, and the noise then supplies
    /// exactly what brings the noise part back to σ_next.</summary>
    public override (double Rescale, double NoiseScale) Renoise(double sigmaDown, double sigmaNext)
    {
        double rescale = (1.0 - sigmaNext) / (1.0 - sigmaDown);
        double noiseScale = Math.Sqrt(Math.Max(
            sigmaNext * sigmaNext - sigmaDown * sigmaDown * rescale * rescale, 0.0));
        return (rescale, noiseScale);
    }
}

/// <summary>The model predicts the noise that was added — Stable Diffusion's
/// default, "epsilon" in the configs. The clean latent is the sample with that
/// noise, scaled to the level, taken back out, and the derivative is the
/// noise itself.</summary>
public sealed class EpsilonPrediction : Prediction
{
    public static readonly EpsilonPrediction Instance = new();

    private EpsilonPrediction()
    {
    }

    public override float[] Denoise(float[] modelOutput, double sigma, float[] sample)
    {
        var denoised = new float[sample.Length];
        for (int i = 0; i < sample.Length; i++)
        {
            denoised[i] = (float)(sample[i] - sigma * modelOutput[i]);
        }
        return denoised;
    }

    public override float[] Derivative(float[] modelOutput, double sigma, float[] sample) =>
        modelOutput;

    /// <summary>Normalize to the unit-variance input a variance-preserving
    /// model expects.</summary>
    public override float ModelInputScale(double sigma) =>
        1.0f / (float)Math.Sqrt(sigma * sigma + 1.0);

    public override float InitNoiseSigma(float sigmaMax, bool variancePreserving) =>
        variancePreserving
            ? (float)Math.Sqrt(sigmaMax * (double)sigmaMax + 1.0)
            : sigmaMax;
}

/// <summary>The model predicts the velocity, "v" in the configs (Salimans and
/// Ho 2022): the combination of noise and signal, v = sqrt(alpha) * noise −
/// sqrt(1 − alpha) * signal, that stays well-conditioned at both ends of the
/// schedule where the noise alone does not. It is what SD 2.x's 768 model
/// and the v-pred SDXL finetunes — NoobAI, some Illustrious builds — put out,
/// flagged by a <c>v_pred</c> marker tensor in the file.
///
/// In sigma terms, with the variance-preserving scaling sqrt(alpha) =
/// 1 / sqrt(sigma² + 1), the clean latent is sample / (sigma² + 1) −
/// v · sigma / sqrt(sigma² + 1), which is k-diffusion's CompVisVDenoiser
/// and ComfyUI's V_PREDICTION. The model's input is scaled exactly as for
/// epsilon; only what comes back means something else.</summary>
public sealed class VelocityPrediction : Prediction
{
    public static readonly VelocityPrediction Instance = new();

    private VelocityPrediction()
    {
    }

    public override float[] Denoise(float[] modelOutput, double sigma, float[] sample)
    {
        double variance = sigma * sigma + 1.0;
        double skip = 1.0 / variance;
        double output = sigma / Math.Sqrt(variance);
        var denoised = new float[sample.Length];
        for (int i = 0; i < sample.Length; i++)
        {
            denoised[i] = (float)(sample[i] * skip - modelOutput[i] * output);
        }
        return denoised;
    }

    /// <summary>(sample − denoised) / sigma, the definition — nothing here
    /// falls out for free the way the noise does for epsilon.</summary>
    public override float[] Derivative(float[] modelOutput, double sigma, float[] sample)
    {
        float[] denoised = Denoise(modelOutput, sigma, sample);
        var derivative = new float[sample.Length];
        for (int i = 0; i < sample.Length; i++)
        {
            derivative[i] = (float)((sample[i] - denoised[i]) / sigma);
        }
        return derivative;
    }

    public override float ModelInputScale(double sigma) =>
        EpsilonPrediction.Instance.ModelInputScale(sigma);

    public override float InitNoiseSigma(float sigmaMax, bool variancePreserving) =>
        EpsilonPrediction.Instance.InitNoiseSigma(sigmaMax, variancePreserving);
}
