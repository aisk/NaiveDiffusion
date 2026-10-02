using System.Runtime.InteropServices;

namespace NaiveDiffusion.Dml;

/// <summary>This process's system memory footprint, for <see cref="MemoryLog"/>
/// to sample alongside the video memory figures.
///
/// Two numbers, because with the checkpoint memory-mapped the two say
/// different things. The working set counts file-backed pages, so it includes
/// however much of the mapping is currently resident; the private bytes do not,
/// so they are what the process actually committed and cannot give back. A
/// change that moves bytes out of the heap and onto a mapping shows up only in
/// the second one.
///
/// <see cref="System.Diagnostics.Process"/> would report the same figures, but
/// it refreshes by enumerating every process on the machine, which is far too
/// much work to do before each of a hundred dispatches.</summary>
internal static partial class HostMemory
{
    public readonly record struct Reading(ulong Private, ulong WorkingSet);

    public static Reading Read()
    {
        var counters = new ProcessMemoryCounters { Size = (uint)Marshal.SizeOf<ProcessMemoryCounters>() };
        return GetProcessMemoryInfo(-1, ref counters, counters.Size)
            ? new Reading(counters.PrivateUsage, counters.WorkingSetSize)
            : default;
    }

    /// <summary>PROCESS_MEMORY_COUNTERS_EX. The K32-prefixed entry point lives in
    /// kernel32 itself, so nothing has to load psapi.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct ProcessMemoryCounters
    {
        public uint Size;
        public uint PageFaultCount;
        public nuint PeakWorkingSetSize;
        public nuint WorkingSetSize;
        public nuint QuotaPeakPagedPoolUsage;
        public nuint QuotaPagedPoolUsage;
        public nuint QuotaPeakNonPagedPoolUsage;
        public nuint QuotaNonPagedPoolUsage;
        public nuint PagefileUsage;
        public nuint PeakPagefileUsage;
        public nuint PrivateUsage;
    }

    [LibraryImport("kernel32.dll", EntryPoint = "K32GetProcessMemoryInfo")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool GetProcessMemoryInfo(nint process,
        ref ProcessMemoryCounters counters, uint size);
}
