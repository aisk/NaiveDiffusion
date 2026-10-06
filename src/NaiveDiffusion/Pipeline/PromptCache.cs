using System.Globalization;

namespace NaiveDiffusion.Pipeline;

/// <summary>What the text encoders made of the prompts of earlier runs, held
/// between generations.
///
/// A text encoder's output depends on the text and on the encoder — its
/// files, the LoRAs folded into it, the layer it is read at — and on nothing
/// else a run sets: not the seed, the size, the sampler, the step count or
/// the guidance. A run that changes only those finds its prompts here and
/// never opens the encoder, which for Qwen-Image is sixteen gigabytes of
/// weights read off the disk for every prompt. Each prompt is kept on its
/// own, so a run that rewrites the prompt and leaves the negative alone
/// encodes one text rather than two.
///
/// What it costs is host memory, a few megabytes a prompt, bounded by
/// <see cref="CapacityBytes"/>; the prompts used longest ago go first.</summary>
public sealed class PromptCache
{
    public const long DefaultCapacityBytes = 64L << 20;

    /// <summary>One text through one encoder. <paramref name="Encoder"/> is
    /// <see cref="EncoderStamp"/>; the text and the length it was encoded to
    /// are the family's <see cref="PromptRequest"/>.</summary>
    public readonly record struct Key(string Encoder, string Text, int Length)
    {
        public Key(string encoder, PromptRequest request) : this(encoder, request.Text, request.Length)
        {
        }
    }

    private readonly object _lock = new();
    private readonly LinkedList<(Key Key, float[][] Encoded, long Bytes)> _recent = new();
    private readonly Dictionary<Key, LinkedListNode<(Key Key, float[][] Encoded, long Bytes)>> _entries = new();
    private long _capacity;
    private long _held;

    public PromptCache(long capacityBytes = DefaultCapacityBytes)
    {
        CapacityBytes = capacityBytes;
    }

    /// <summary>How many bytes of encoded prompts are kept at most. Lowering
    /// it drops the oldest down to the new figure, and 0 keeps nothing.</summary>
    public long CapacityBytes
    {
        get { lock (_lock) { return _capacity; } }
        set
        {
            if (value < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(value), value, "a capacity of 0 or more bytes");
            }
            lock (_lock)
            {
                _capacity = value;
                Trim();
            }
        }
    }

    public long HeldBytes
    {
        get { lock (_lock) { return _held; } }
    }

    public int Count
    {
        get { lock (_lock) { return _entries.Count; } }
    }

    /// <summary>What was kept under the key, or null. The arrays are the
    /// cache's own and are handed to every run that asks: read, never
    /// written.</summary>
    public float[][]? Find(Key key)
    {
        lock (_lock)
        {
            if (!_entries.TryGetValue(key, out var node))
            {
                return null;
            }
            _recent.Remove(node);
            _recent.AddFirst(node);
            return node.Value.Encoded;
        }
    }

    /// <summary>Keep an encoder's output, in place of anything already under
    /// the key. One larger than the whole capacity is not kept.</summary>
    public void Add(Key key, float[][] encoded)
    {
        long bytes = encoded.Sum(values => (long)values.Length * sizeof(float));
        lock (_lock)
        {
            if (_entries.Remove(key, out var old))
            {
                _recent.Remove(old);
                _held -= old.Value.Bytes;
            }
            if (bytes > _capacity)
            {
                return;
            }
            _entries[key] = _recent.AddFirst((key, encoded, bytes));
            _held += bytes;
            Trim();
        }
    }

    public void Clear()
    {
        lock (_lock)
        {
            _recent.Clear();
            _entries.Clear();
            _held = 0;
        }
    }

    private void Trim()
    {
        while (_held > _capacity)
        {
            var last = _recent.Last!;
            _recent.RemoveLast();
            _entries.Remove(last.Value.Key);
            _held -= last.Value.Bytes;
        }
    }

    /// <summary>Everything a family's text encoder is besides the text it is
    /// given: the checkpoint and the files of the parts the conditioner
    /// reads, each by its size and write time as well as its path, since a
    /// path can be overwritten in place; the LoRAs in the same terms with
    /// their weights, whether or not one of them reaches the encoder; and
    /// the clip skip.</summary>
    public static string EncoderStamp(IModelFamily family, GenerationOptions options)
    {
        var parts = new List<string> { family.Name, FileStamp(options.CheckpointPath) };
        foreach (ModelComponent component in family.Components)
        {
            if ((component.Part & PipelineParts.Conditioner) != 0
                && options.ComponentPath(component.Id) is string path)
            {
                parts.Add($"{component.Id}={FileStamp(path)}");
            }
        }
        parts.Add(options.ClipSkip.ToString(CultureInfo.InvariantCulture));
        parts.Add(Weights.LoraSpec.Stamp(options.Loras));
        return string.Join("\n", parts);
    }

    private static string FileStamp(string path)
    {
        var file = new FileInfo(path);
        return $"{Path.GetFullPath(path)}|{file.Length}|{file.LastWriteTimeUtc.Ticks}";
    }
}
