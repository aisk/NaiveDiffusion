namespace NaiveDiffusion.Pipeline;

/// <summary>How the denoiser's runs on each conditioning branch combine into
/// the one prediction the sampler steps on. Classifier-free guidance runs
/// twice and amplifies the difference; a guidance-distilled model runs once
/// and takes the scale as an input instead.</summary>
public abstract class GuidanceStrategy
{
    /// <summary>Denoiser runs per step at this guidance scale, one per
    /// conditioning branch; the conditioner encodes as many, negative first
    /// and the prompt's own last.</summary>
    public abstract int Branches(float guidance);

    /// <summary>The prediction to step on, from the branches' outputs in
    /// order and the guidance scale.</summary>
    public abstract float[] Combine(float[][] predictions, float guidance);
}

/// <summary>Classifier-free guidance: the model runs on the negative prompt
/// and on the prompt, and the difference between the two is what gets
/// amplified. Always both, whatever the scale: at 1 the arithmetic lands
/// on the prompt's branch to within rounding, and SDXL's images are
/// checked bit for bit against that arithmetic.</summary>
public sealed class ClassifierFreeGuidance : GuidanceStrategy
{
    public static readonly ClassifierFreeGuidance Instance = new();

    private ClassifierFreeGuidance()
    {
    }

    public override int Branches(float guidance) => 2;

    public override float[] Combine(float[][] predictions, float guidance)
    {
        float[] uncond = predictions[0];
        float[] cond = predictions[1];
        var prediction = new float[uncond.Length];
        for (int i = 0; i < prediction.Length; i++)
        {
            prediction[i] = uncond[i] + guidance * (cond[i] - uncond[i]);
        }
        return prediction;
    }
}

/// <summary>Classifier-free guidance that is off at a scale of 1: the model
/// runs on the prompt alone and its prediction is stepped on as it is,
/// which is what a model released to run without guidance is meant to do
/// and half the time of running the negative prompt for nothing. Above 1
/// it is <see cref="ClassifierFreeGuidance"/>.</summary>
public sealed class OptionalClassifierFreeGuidance : GuidanceStrategy
{
    public static readonly OptionalClassifierFreeGuidance Instance = new();

    private OptionalClassifierFreeGuidance()
    {
    }

    public override int Branches(float guidance) => guidance == 1f ? 1 : 2;

    public override float[] Combine(float[][] predictions, float guidance) =>
        predictions.Length == 1
            ? predictions[0]
            : ClassifierFreeGuidance.Instance.Combine(predictions, guidance);
}
