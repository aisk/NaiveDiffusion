namespace NaiveDiffusion.Vae;

/// <summary>The shape of one AutoencoderKL: diffusers' config, reduced to
/// what the graph builder reads.</summary>
/// <param name="BlockOutChannels">Width per resolution level, highest
/// resolution first.</param>
/// <param name="LayersPerBlock">Resnets per level in the encoder; the decoder
/// has one more.</param>
/// <param name="Latent">The latent the encoder produces and the decoder
/// consumes.</param>
public sealed record VaeConfig(int[] BlockOutChannels, int LayersPerBlock, int NormGroups,
    float NormEpsilon, LatentSpace Latent);
