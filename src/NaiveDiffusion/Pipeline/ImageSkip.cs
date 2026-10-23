namespace NaiveDiffusion.Pipeline;

/// <summary>Drops the image being sampled and lets the run go on with the
/// next seed. Where cancelling stops the whole run, this gives up on one
/// image that is visibly going wrong. A request made while no image is
/// sampling (encoding, building the model, decoding) is ignored, and none
/// outlives the run it was made in.</summary>
public sealed class ImageSkip
{
    private int _requested;

    /// <summary>Skip the image being sampled now. Safe from any thread.</summary>
    public void Request() => Interlocked.Exchange(ref _requested, 1);

    /// <summary>Whether a request came in since the last call, clearing it.</summary>
    internal bool Take() => Interlocked.Exchange(ref _requested, 0) == 1;

    internal void Clear() => Interlocked.Exchange(ref _requested, 0);
}
