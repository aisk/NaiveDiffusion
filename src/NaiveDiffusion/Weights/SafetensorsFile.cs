using System.Buffers;
using System.IO.MemoryMappedFiles;
using System.Text.Json;
using NaiveDiffusion.Tensors;

namespace NaiveDiffusion.Weights;

/// <summary>Reads the safetensors format: an 8-byte header length, a JSON table
/// of dtype/shape/offsets per tensor, then the raw data.
///
/// The file is memory-mapped and a tensor is a window onto it — no bytes are
/// read into the heap. That matters because most of a checkpoint is on its way
/// somewhere that does not want a copy: the UNet is stored at the half precision
/// it runs at, so it goes from the mapping straight into DirectML's staging
/// buffer, and the text towers are read a row at a time by the CPU. Mapped pages
/// are file-backed, so they cost no commit and the OS can drop them under
/// pressure and fault them back from the page cache.
///
/// <b>Every tensor it hands out is only valid until this is disposed.</b> The
/// caller owns that lifetime: hold the file until the weights have been uploaded
/// or converted, and no longer.</summary>
public sealed unsafe class SafetensorsFile : IDisposable
{
    private sealed record Entry(string DataType, int[] Shape, long Begin, long End);

    private readonly MemoryMappedFile _map;
    private readonly MemoryMappedViewAccessor _view;
    private readonly byte* _base;
    private readonly Dictionary<string, Entry> _entries = new();
    private readonly Dictionary<string, string> _metadata = new();
    private readonly long _dataStart;
    private bool _disposed;

    public SafetensorsFile(string path)
    {
        // Every offset below is checked against this before anything reads at
        // it. Nothing in the format is self-describing enough to notice a
        // truncated download on its own, and a read past the end of the mapping
        // raises an AccessViolationException, which no caller can catch — so a
        // file that does not describe itself consistently is turned away here,
        // where it is still an exception someone can show.
        long length = new FileInfo(path).Length;
        if (length < 8)
        {
            throw new InvalidDataException("the file is too short to be a safetensors file");
        }

        try
        {
            _map = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0,
                MemoryMappedFileAccess.Read);
            _view = _map.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);

            byte* pointer = null;
            _view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
            _base = pointer + _view.PointerOffset;
            _dataStart = ReadHeader(length);
        }
        catch
        {
            // A refused file must not stay mapped. The user is about to
            // replace it — finish the download, pick the right one — and a
            // mapping left open, pointer acquired and with no finalizer to
            // let go of it, would make the overwrite fail until the process
            // exits.
            Dispose();
            throw;
        }
    }

    /// <summary>Parse the JSON table into <see cref="_entries"/>, returning
    /// where the tensor data starts.</summary>
    private long ReadHeader(long length)
    {
        long headerLength = BitConverter.ToInt64(new ReadOnlySpan<byte>(_base, 8));
        if (headerLength <= 0 || headerLength > length - 8 || headerLength > int.MaxValue)
        {
            throw new InvalidDataException(
                "the safetensors header does not fit the file: it is not a safetensors " +
                "file, or the download did not finish");
        }
        long dataStart = 8 + headerLength;
        long dataLength = length - dataStart;

        JsonDocument header;
        try
        {
            header = JsonDocument.Parse(
                new ReadOnlyMemory<byte>(new ReadOnlySpan<byte>(_base + 8,
                    checked((int)headerLength)).ToArray()));
        }
        catch (JsonException exception)
        {
            // One kind of "not a safetensors file" for every caller.
            throw new InvalidDataException(
                $"the safetensors header is not valid JSON: {exception.Message}", exception);
        }
        using var _ = header;
        try
        {
            foreach (JsonProperty property in header.RootElement.EnumerateObject())
            {
                if (property.Name == "__metadata__")
                {
                    // What the trainer wrote about the model. Only the string-valued
                    // entries: the convention allows nothing else, but a file is
                    // free to be wrong about that and it is not worth failing over.
                    foreach (JsonProperty note in property.Value.EnumerateObject())
                    {
                        if (note.Value.ValueKind == JsonValueKind.String)
                        {
                            _metadata[note.Name] = note.Value.GetString()!;
                        }
                    }
                    continue;
                }
                JsonElement value = property.Value;
                int[] shape = value.GetProperty("shape").EnumerateArray()
                    .Select(extent => extent.GetInt32()).ToArray();
                JsonElement offsets = value.GetProperty("data_offsets");
                long begin = offsets[0].GetInt64(), end = offsets[1].GetInt64();
                if (begin < 0 || end < begin || end > dataLength)
                {
                    throw new InvalidDataException(
                        $"the safetensors entry {property.Name} points outside the file: " +
                        "the file is damaged or the download did not finish");
                }
                // The dtype is kept as written and only translated in Read. A
                // checkpoint quantized to a width this library cannot run is still one
                // whose header is worth reading — CheckpointInspector names the
                // width in its verdict, which it could not do if this threw here.
                _entries[property.Name] = new Entry(
                    value.GetProperty("dtype").GetString()
                        ?? throw new InvalidDataException(
                            $"the safetensors entry {property.Name} has no dtype"),
                    shape, begin, end);
            }
        }
        catch (Exception exception) when (exception is InvalidOperationException
            or KeyNotFoundException or IndexOutOfRangeException or FormatException)
        {
            // JSON, but not the table: the root or an entry is not an object,
            // an entry lacks its shape, dtype or offsets, a number is not one.
            // The same kind of "not a safetensors file" as JSON that does not
            // parse, so that whoever opened it has one thing to catch.
            throw new InvalidDataException(
                $"the safetensors header is not a tensor table: {exception.Message}", exception);
        }
        return dataStart;
    }

    public IEnumerable<string> Keys => _entries.Keys;

    /// <summary>What the file says about itself: the __metadata__ the trainer
    /// wrote, which is where a model card records things the tensors do not.</summary>
    public IReadOnlyDictionary<string, string> Metadata => _metadata;

    /// <summary>One tensor as the header describes it, without mapping it. The
    /// dtype is the string the file carries, which may be one this library cannot
    /// read.</summary>
    public bool TryGetInfo(string name, out string dataType, out int[] shape)
    {
        if (_entries.TryGetValue(name, out Entry? entry))
        {
            (dataType, shape) = (entry.DataType, entry.Shape);
            return true;
        }
        (dataType, shape) = ("", Array.Empty<int>());
        return false;
    }

    /// <summary>One tensor, as a window onto the mapping.</summary>
    public HostTensor Read(string name)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Entry entry = _entries[name];
        int length = checked((int)(entry.End - entry.Begin));
        var window = new MappedWindow(_base + _dataStart + entry.Begin, length);
        return new HostTensor(ParseDataType(entry.DataType), entry.Shape, window.Memory);
    }

    /// <summary>Every tensor whose name starts with <paramref name="prefix"/>,
    /// keyed by the name with the prefix stripped off.</summary>
    public Dictionary<string, HostTensor> ReadPrefix(string prefix)
    {
        var tensors = new Dictionary<string, HostTensor>();
        foreach (string name in _entries.Keys)
        {
            if (name.StartsWith(prefix, StringComparison.Ordinal))
            {
                tensors[name[prefix.Length..]] = Read(name);
            }
        }
        return tensors;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        // Only as far as the constructor got, when it is the one calling.
        if (_base != null)
        {
            _view.SafeMemoryMappedViewHandle.ReleasePointer();
        }
        _view?.Dispose();
        _map?.Dispose();
    }

    /// <summary>Whether a tensor stored at this width can be read at all.</summary>
    public static bool IsSupportedDataType(string dtype) => dtype switch
    {
        "F32" or "F16" or "BF16" or "U32" or "I32" or "U16" or "I64" => true,
        _ => false,
    };

    private static HostDataType ParseDataType(string dtype) => dtype switch
    {
        "F32" => HostDataType.Float32,
        "F16" => HostDataType.Float16,
        "BF16" => HostDataType.BFloat16,
        "U32" => HostDataType.UInt32,
        "I32" => HostDataType.Int32,
        "U16" => HostDataType.UInt16,
        "I64" => HostDataType.Int64,
        _ => throw new NotSupportedException($"safetensors dtype {dtype} is not supported"),
    };

    /// <summary>Presents a range of the mapping as <see cref="ReadOnlyMemory{T}"/>.
    /// It owns nothing — the mapping outlives every window by construction,
    /// because the file is what gets disposed.</summary>
    private sealed class MappedWindow : MemoryManager<byte>
    {
        private readonly byte* _pointer;
        private readonly int _length;

        public MappedWindow(byte* pointer, int length)
        {
            _pointer = pointer;
            _length = length;
        }

        public override Span<byte> GetSpan() => new(_pointer, _length);

        public override MemoryHandle Pin(int elementIndex = 0) =>
            new(_pointer + elementIndex);

        public override void Unpin()
        {
        }

        protected override void Dispose(bool disposing)
        {
        }
    }
}
