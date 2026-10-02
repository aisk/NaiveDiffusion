using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace NaiveDiffusion.Dml;

/// <summary>Video memory readings, taken before every submission and at the
/// points where one stage hands over to the next.
///
/// Two numbers describe the same thing from opposite sides and neither is
/// enough alone. The <em>usage</em> figures come from DXGI and are the truth —
/// this process's real footprint, driver and DirectML allocations included —
/// but they are a single total with no attribution. The <em>tracked</em>
/// figures are this code adding up the buffers it created itself, so they say
/// which allocation grew, but they miss everything allocated on our behalf.
/// The gap between the two is the driver's own overhead, and watching it is how
/// a surprise like a repacked second copy of a gemm's weights gets noticed.
///
/// Local memory is the card's own; non-local is system memory the GPU reaches
/// over PCIe. Non-local rising during a run is the signal that the working set
/// no longer fits and the weights are being read across the bus — the state
/// that turns a two second dispatch into a TDR.</summary>
public sealed class MemoryLog
{
    /// <summary>One reading. <paramref name="Event"/> says what was about to
    /// happen (or had just finished); <paramref name="Stage"/> is whatever the
    /// caller had set as the current stage, so a peak can be attributed to the
    /// part of the pipeline that caused it.
    ///
    /// <b>The DXGI figures are this process's, not the system's.</b>
    /// <paramref name="LocalUsage"/> is what this process holds in video memory
    /// and <paramref name="LocalBudget"/> is what the OS is currently willing to
    /// let it hold — neither is the card's total, and the budget moves as other
    /// processes take and release memory. That is what makes it the useful
    /// number: usage crossing budget is the moment the working set starts
    /// spilling to system memory.</summary>
    public readonly record struct Sample(TimeSpan Elapsed, string Stage, string Event,
        ulong LocalUsage, ulong LocalBudget, ulong NonLocalUsage,
        ulong Persistent, ulong Temporary, ulong Inputs, ulong Outputs, ulong TrackedShared,
        ulong Private, ulong WorkingSet, ulong Managed)
    {
        /// <summary>What this code knows it allocated in video memory. The
        /// inputs buffer is not in it: that is the staging copy, which lives
        /// in system memory and is counted under <see cref="TrackedShared"/>.</summary>
        public ulong TrackedLocal => Persistent + Temporary + Outputs;

        /// <summary>Committed system memory this code cannot account for: the
        /// private bytes less the managed heap and less the CPU-visible buffers
        /// it allocated itself. What is left was allocated by DirectML or by the
        /// driver, and nothing here can attribute it any further — but nothing
        /// else would notice it growing either.</summary>
        public ulong UnattributedPrivate =>
            Private > Managed + TrackedShared ? Private - Managed - TrackedShared : 0;
    }

    private readonly Stopwatch _clock = Stopwatch.StartNew();
    private readonly List<Sample> _samples = new();

    // Boxed and volatile so a UI thread can read the latest reading while the
    // pipeline's worker thread is still appending: a reference assignment is
    // atomic where a struct field's would tear.
    private volatile object? _last;

    /// <summary>The most recent reading, for a caller that wants to show where
    /// memory stands right now rather than read the whole history. Safe to read
    /// from another thread.</summary>
    public Sample? Last => _last is Sample sample ? sample : null;

    internal void Add(string stage, string what, ulong localUsage, ulong localBudget,
        ulong nonLocalUsage, ulong persistent, ulong temporary, ulong inputs, ulong outputs,
        ulong trackedShared)
    {
        HostMemory.Reading host = HostMemory.Read();
        var sample = new Sample(_clock.Elapsed, stage, what, localUsage, localBudget,
            nonLocalUsage, persistent, temporary, inputs, outputs, trackedShared,
            host.Private, host.WorkingSet, (ulong)GC.GetTotalMemory(false));
        lock (_samples)
        {
            _samples.Add(sample);
        }
        _last = sample;
    }

    /// <summary>The history as it stands. The worker thread appends while a
    /// UI thread may be reading or clearing, so every touch of the list is
    /// under its lock and the readers walk a copy.</summary>
    private Sample[] Snapshot()
    {
        lock (_samples)
        {
            return _samples.ToArray();
        }
    }

    /// <summary>Drop the history. A long-lived device generating image after
    /// image would otherwise accumulate one reading per dispatch forever.</summary>
    public void Clear()
    {
        lock (_samples)
        {
            _samples.Clear();
        }
        _last = null;
    }

    /// <summary>Every reading, one row each, for plotting or for diffing two
    /// runs against each other. Bytes stay bytes here — this is for a program to
    /// read, and rounding them to gigabytes throws away the resolution that
    /// makes two runs comparable.</summary>
    public void WriteCsv(string path)
    {
        var text = new StringBuilder(
            "seconds,stage,event,local_usage,local_budget,nonlocal_usage,tracked_local," +
            "persistent,temporary,inputs,outputs,tracked_shared," +
            "private,working_set,managed\n");
        foreach (Sample sample in Snapshot())
        {
            // Invariant throughout: a machine with a comma for a decimal
            // separator would otherwise write a file with the wrong number of
            // columns.
            text.Append(sample.Elapsed.TotalSeconds.ToString("0.000",
                CultureInfo.InvariantCulture));
            text.Append(",\"").Append(sample.Stage).Append("\",\"").Append(sample.Event);
            text.Append("\",").Append(sample.LocalUsage);
            text.Append(',').Append(sample.LocalBudget);
            text.Append(',').Append(sample.NonLocalUsage);
            text.Append(',').Append(sample.TrackedLocal);
            text.Append(',').Append(sample.Persistent);
            text.Append(',').Append(sample.Temporary);
            text.Append(',').Append(sample.Inputs);
            text.Append(',').Append(sample.Outputs);
            text.Append(',').Append(sample.TrackedShared);
            text.Append(',').Append(sample.Private);
            text.Append(',').Append(sample.WorkingSet);
            text.Append(',').Append(sample.Managed);
            text.Append('\n');
        }
        File.WriteAllText(path, text.ToString());
    }

    /// <summary>The peak of each stage, in the order the stages ran — the shape
    /// of a run at a glance, and the answer to "which stage is the one that will
    /// not fit". Empty when nothing was ever sampled.
    ///
    /// Two tables, because the two memories peak at different moments: video
    /// memory is highest while a graph is dispatching and system memory while
    /// the previous one was being staged into it, so a single row picked by one
    /// of them would report the other at whatever it happened to be.</summary>
    public string Summarize()
    {
        Sample[] samples = Snapshot();
        if (samples.Length == 0)
        {
            return "";
        }

        var order = new List<string>();
        var video = new Dictionary<string, Sample>(StringComparer.Ordinal);
        var host = new Dictionary<string, Sample>(StringComparer.Ordinal);
        foreach (Sample sample in samples)
        {
            if (!video.TryGetValue(sample.Stage, out Sample peak))
            {
                order.Add(sample.Stage);
                video[sample.Stage] = sample;
                host[sample.Stage] = sample;
                continue;
            }
            if (sample.LocalUsage > peak.LocalUsage)
            {
                video[sample.Stage] = sample;
            }
            if (sample.Private > host[sample.Stage].Private)
            {
                host[sample.Stage] = sample;
            }
        }

        int width = order.Max(stage => Label(stage).Length);
        var text = new StringBuilder("video memory peaks (MiB):");
        text.AppendLine();
        text.Append($"  {"stage".PadRight(width)}  {"usage",6} {"budget",7} {"shared",7}  " +
                    $"{"weights",7} {"scratch",7} {"staging",7}  at");
        foreach (string stage in order)
        {
            Sample peak = video[stage];
            text.AppendLine();
            text.Append($"  {Label(stage).PadRight(width)}  " +
                        $"{Mib(peak.LocalUsage),6} {Mib(peak.LocalBudget),7} " +
                        $"{Mib(peak.NonLocalUsage),7}  " +
                        $"{Mib(peak.Persistent),7} {Mib(peak.Temporary),7} " +
                        $"{Mib(peak.Inputs),7}  {peak.Event}");
        }

        text.AppendLine();
        text.AppendLine();
        // Private bytes are the number that matters: the working set counts the
        // pages of the mapped checkpoint that happen to be resident, which the
        // OS can drop at any time, and the difference between the columns is
        // roughly how much of the run is file-backed rather than committed.
        text.Append("host memory peaks (MiB):");
        text.AppendLine();
        text.Append($"  {"stage".PadRight(width)}  {"private",7} {"working",7}  " +
                    $"{"managed",7} {"staging",7} {"other",7}  at");
        foreach (string stage in order)
        {
            Sample peak = host[stage];
            text.AppendLine();
            text.Append($"  {Label(stage).PadRight(width)}  " +
                        $"{Mib(peak.Private),7} {Mib(peak.WorkingSet),7}  " +
                        $"{Mib(peak.Managed),7} {Mib(peak.TrackedShared),7} " +
                        $"{Mib(peak.UnattributedPrivate),7}  {peak.Event}");
        }
        return text.ToString();
    }

    private static string Label(string stage) => stage.Length == 0 ? "(no stage)" : stage;

    /// <summary>Whole mebibytes. Gigabytes would read better for the stage this
    /// exists to size — a UNet is gigabytes — but they round the small graphs to
    /// a column of zeroes, which looks like the readings are broken.</summary>
    private static string Mib(ulong bytes) =>
        (bytes >> 20).ToString(CultureInfo.InvariantCulture);
}
