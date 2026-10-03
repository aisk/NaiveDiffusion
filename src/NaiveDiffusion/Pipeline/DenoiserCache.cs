using NaiveDiffusion.Dml;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Pipeline;

/// <summary>A built diffusion model, held between generations.
///
/// Building the model costs a good part of a run — SDXL's UNet is 4.78 GiB
/// to read, convert and upload and a dozen graphs to compile, about fifteen
/// seconds of a fifty-eight second run — and none of it depends on the
/// prompt, the seed, the sampler or the step count. A second run that
/// changes only those can reuse the graphs as they stand, which is what most
/// runs are: the same idea, a different seed.
///
/// What it costs is the model's video memory — 5.5 GiB for that UNet — held
/// while nothing is generating, so this is opt-in and the holder can hand it
/// back at any time.
///
/// The pipeline borrows and returns rather than looking up and keeping, so a
/// run that fails partway leaves nothing half-built behind: while a run holds
/// the model the cache is empty, and the run decides on the way out whether it
/// goes back in.</summary>
public sealed class DenoiserCache : IDisposable
{
    /// <summary>Everything a compiled denoiser is specialized for. The prompt,
    /// the seed, the sampler, the schedule, the step count and the guidance
    /// are all bound at dispatch and are deliberately absent — changing them
    /// is the case this cache exists for.
    ///
    /// <paramref name="Specialization"/> is the family's own, from its
    /// <see cref="Conditioning"/>: for SDXL the context width, so a prompt
    /// growing past 75 tokens adds a chunk and invalidates the graphs. The
    /// checkpoint is identified by its size and write time as well as its
    /// path, because a path can be overwritten in place with a different
    /// model. The LoRAs are in the same terms, with their weights: they are
    /// folded into the weights the graphs hold, so adding one, dropping one
    /// or moving its slider is a different model. The compute precision
    /// builds different graphs outright, so it is part of the key too.</summary>
    public readonly record struct Key(string Family, string CheckpointPath,
        long CheckpointLength, long CheckpointStamp, int Height, int Width,
        string Specialization, int Batch, WeightStorage Weights, ComputePrecision Compute, string Loras)
    {
        public static Key For(string family, GenerationOptions options,
            Conditioning conditioning, int batch = 1)
        {
            var file = new FileInfo(options.CheckpointPath);
            return new Key(family, Path.GetFullPath(options.CheckpointPath), file.Length,
                file.LastWriteTimeUtc.Ticks, options.Height, options.Width,
                conditioning.Specialization, batch, options.DenoiserWeights,
                options.DenoiserCompute, LoraSpec.Stamp(options.Loras));
        }
    }

    private readonly object _lock = new();
    private DmlDevice? _device;
    private Key _key;
    private IDenoiser? _denoiser;
    private bool _enabled;

    /// <summary>Whether a returned model is kept. Turning it off releases what
    /// is held right now, and a run already in flight releases its model on the
    /// way out instead of putting it back — the answer that counts is the one
    /// standing when the run ends, not the one it started under.</summary>
    public bool Enabled
    {
        get { lock (_lock) { return _enabled; } }
        set
        {
            lock (_lock)
            {
                _enabled = value;
                if (!value)
                {
                    Release();
                }
            }
        }
    }

    /// <summary>Stop keeping, but leave what is held for the thread that runs
    /// the device to release: a run in flight does it when it next borrows or
    /// returns, and its caller turns <see cref="Enabled"/> off once the run
    /// is over for what that did not reach. Releasing is a dispose on the
    /// device, which is not to happen from a second thread while the first
    /// is dispatching on it.</summary>
    public void StopKeeping()
    {
        lock (_lock)
        {
            _enabled = false;
        }
    }

    /// <summary>Whether a model is being held onto right now — false while a
    /// run has one borrowed.</summary>
    public bool IsLoaded
    {
        get { lock (_lock) { return _denoiser is not null; } }
    }

    /// <summary>The cached model if it was built for this device and this key,
    /// otherwise null. Either way the cache is empty afterwards: a model that
    /// does not match is released here, before the caller builds its
    /// replacement, because the two do not fit at once.</summary>
    public IDenoiser? Borrow(DmlDevice device, Key key)
    {
        lock (_lock)
        {
            if (_denoiser is not null && ReferenceEquals(_device, device) && _key == key && _enabled)
            {
                IDenoiser denoiser = _denoiser;
                _denoiser = null;
                _device = null;
                return denoiser;
            }
            Release();
            return null;
        }
    }

    /// <summary>Hand a borrowed model back, and say whether it was kept. It is
    /// released instead of stored when caching has been turned off in the
    /// meantime — the caller reports what actually happened to it.</summary>
    public bool Return(DmlDevice device, Key key, IDenoiser denoiser)
    {
        lock (_lock)
        {
            if (!_enabled)
            {
                denoiser.Dispose();
                return false;
            }
            Release();
            _device = device;
            _key = key;
            _denoiser = denoiser;
            return true;
        }
    }

    /// <summary>Release whatever is held. The device's resources belong to the
    /// device, so this has to run before that device is disposed — and after a
    /// device is lost, before the reference to it is dropped.</summary>
    public void Clear()
    {
        lock (_lock)
        {
            Release();
        }
    }

    private void Release()
    {
        _denoiser?.Dispose();
        _denoiser = null;
        _device = null;
    }

    public void Dispose() => Clear();
}
