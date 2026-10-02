using NaiveDiffusion.Tensors;

namespace NaiveDiffusion.Pipeline;

/// <summary>A built diffusion model: a noisy latent, the noise level and one
/// branch of conditioning in, the model's prediction out — noise, velocity or
/// flow, which is the <see cref="Sampling.Prediction"/>'s business, not this
/// one's. The memory figures are what the pipeline reports and what decides
/// whether the model can stay resident through the decode.</summary>
public interface IDenoiser : IDisposable
{
    /// <summary>The level arrives in both of the forms a model is trained to
    /// read it in: <paramref name="timestep"/> is the continuous index into
    /// the training table, which a UNet embeds, and <paramref name="sigma"/>
    /// is the level itself, which a flow model takes as its time.</summary>
    HostTensor Predict(HostTensor latent, double timestep, double sigma,
        IReadOnlyDictionary<string, HostTensor> conditioning);

    /// <summary>Memory the folded weights take, wherever they are.</summary>
    ulong PersistentBytes { get; }

    /// <summary>Of those, the bytes in system memory, read across the bus every step.</summary>
    ulong StreamedBytes { get; }

    /// <summary>And the bytes in video memory.</summary>
    ulong ResidentBytes { get; }

    /// <summary>Scratch one step needs, on top of the resident weights.</summary>
    ulong TemporaryBytes { get; }

    int GraphCount { get; }
}
