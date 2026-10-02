namespace NaiveDiffusion.Sampling;

/// <summary>What a sampler has to know about the model it is driving: the
/// noise levels it was trained at and what its output means. A family
/// decides both per checkpoint — the same SDXL UNet is shipped as epsilon
/// and as v-prediction, with and without the zero-terminal-SNR rescale —
/// and the pipeline hands the pair to <see cref="SigmaScheduler.Create"/>
/// without looking inside.</summary>
public sealed record ModelSampling(NoiseSchedule Levels, Prediction Prediction);
