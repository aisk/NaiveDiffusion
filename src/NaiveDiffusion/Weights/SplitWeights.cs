namespace NaiveDiffusion.Weights;

/// <summary>Weights split over three files — a diffusion model's
/// checkpoint, a text encoder and a VAE — each mapped whole the first time
/// something is read from it, so a run that only wants the VAE never
/// touches the transformer's gigabytes. The family's own class says how
/// its checkpoint is recognized and what each file's tensors become; this
/// owns the mappings, and as with <see cref="Models.Sdxl.SdxlWeights"/> the
/// tensors handed out are windows onto them, so it is disposed after they
/// have been uploaded or converted and never before.</summary>
public abstract class SplitWeights : IDisposable
{
    private readonly string _checkpointPath;
    private readonly string? _textEncoderPath;
    private readonly string? _vaePath;
    private SafetensorsFile? _checkpoint;
    private string? _prefix;
    private SafetensorsFile? _textEncoder;
    private SafetensorsFile? _vae;

    /// <param name="checkpointPath">The diffusion model's file, under any of the family's prefixes.</param>
    /// <param name="textEncoderPath">The text encoder's file, when it will be read.</param>
    /// <param name="vaePath">The VAE's file, when it will be read.</param>
    protected SplitWeights(string checkpointPath, string? textEncoderPath, string? vaePath)
    {
        _checkpointPath = checkpointPath;
        _textEncoderPath = textEncoderPath;
        _vaePath = vaePath;
    }

    /// <summary>The prefix the diffusion model sits under in this file, or
    /// null when the file is not this family's checkpoint.</summary>
    protected abstract string? CheckpointPrefix(SafetensorsFile file);

    /// <summary>What the checkpoint should have been, for the refusal:
    /// "an Anima checkpoint".</summary>
    protected abstract string CheckpointKind { get; }

    protected string CheckpointPath => _checkpointPath;

    protected string? TextEncoderPath => _textEncoderPath;

    protected string? VaePath => _vaePath;

    /// <summary>The checkpoint, mapped on first use and refused at once
    /// when it is not this family's.</summary>
    protected SafetensorsFile Checkpoint
    {
        get
        {
            if (_checkpoint is null)
            {
                var file = new SafetensorsFile(_checkpointPath);
                _prefix = CheckpointPrefix(file);
                if (_prefix is null)
                {
                    file.Dispose();
                    throw new InvalidDataException($"{_checkpointPath}: not {CheckpointKind}");
                }
                _checkpoint = file;
            }
            return _checkpoint;
        }
    }

    /// <summary>The prefix the diffusion model's tensors sit under in the
    /// checkpoint; maps it if it has not been.</summary>
    protected string Prefix
    {
        get
        {
            _ = Checkpoint;
            return _prefix!;
        }
    }

    /// <summary>The text encoder's file, mapped on first use.</summary>
    protected SafetensorsFile TextEncoderFile =>
        _textEncoder ??= new SafetensorsFile(_textEncoderPath
            ?? throw new InvalidOperationException("no text encoder file was given"));

    /// <summary>The VAE's file, mapped on first use.</summary>
    protected SafetensorsFile VaeFile =>
        _vae ??= new SafetensorsFile(_vaePath
            ?? throw new InvalidOperationException("no VAE file was given"));

    public void Dispose()
    {
        _vae?.Dispose();
        _textEncoder?.Dispose();
        _checkpoint?.Dispose();
    }
}
