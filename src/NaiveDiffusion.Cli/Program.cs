using NaiveDiffusion.Cli;
using NaiveDiffusion.Dml;

if (args.Length == 0)
{
    Console.Error.WriteLine(Commands.Usage);
    return 2;
}
Command? command = Commands.Find(args[0]);
if (command is null)
{
    Console.Error.WriteLine($"unknown command: {args[0]}");
    Console.Error.WriteLine(Commands.Usage);
    return 2;
}

// Anything on the command line that cannot be run as asked — an unknown
// option, a missing value, a file that is not what it was given as — ends
// here with its sentence and exit code 2, wherever in the command it was
// noticed: nothing is run with a default in place of what was asked for.
try
{
    CommandLine line = CommandLine.Parse(args, Commands.ValueOptionsOf(command),
        command.FlagOptions);
    if (!command.OpensDevice)
    {
        return command.Run(null, line);
    }

    // Every command that opens a device can pick the adapter it opens, and
    // can ask for the video memory readings taken along the way.
    // An adapter that is not there is a bad option like any other; left out,
    // the first hardware adapter is opened.
    int adapter = line.Int("--adapter", -1);
    if (adapter < 0 && line.Value("--adapter") is not null)
    {
        CommandLine.Fail($"--adapter {adapter}: an index that devices lists");
    }
    int hogMib = line.Int("--hog", 0);
    if (hogMib < 0)
    {
        CommandLine.Fail($"--hog {hogMib}: MiB, 0 or above");
    }
    using DmlDevice device = Open(adapter);
    Console.WriteLine($"DirectML on {device.AdapterName}, " +
                      $"feature level {device.Device.HighestFeatureLevel}");

    // --hog N takes N MiB of video memory off the table before the command
    // runs, which is how a smaller card is stood in for on this one: the OS
    // demotes our allocations before it touches this one, so the run sees a
    // card that much smaller. Test tooling; a real user has other processes
    // for this.
    using IDisposable? hog = hogMib > 0
        ? device.HoldVideoMemory((ulong)hogMib << 20)
        : null;

    int status;
    try
    {
        status = command.Run(device, line);
    }
    catch (Exception exception) when (exception is not UsageException)
    {
        // A run that died of memory is the one whose readings matter most, so
        // they print on the way out too. A lost device or an exhausted card is
        // the machine's state, not a fault in the code: its sentence is the
        // whole story and the exit code is 1. Anything else goes on to the
        // runtime's own report, stack and all.
        Console.Error.WriteLine($"failed: {exception.Message}");
        Report(device, line);
        if (exception is DeviceRemovedException or OutOfVideoMemoryException)
        {
            return 1;
        }
        throw;
    }
    Report(device, line);
    return status;
}
catch (UsageException exception)
{
    Console.Error.WriteLine(exception.Message);
    return 2;
}

// The device constructor's own refusals — no adapter at that index, one that
// cannot create a Direct3D 12 device, no hardware adapter at all — are
// sentences about the command line or the machine, not failures of a run.
static DmlDevice Open(int adapter)
{
    try
    {
        return new DmlDevice(adapterIndex: adapter);
    }
    catch (Exception exception) when (exception is ArgumentException or InvalidOperationException)
    {
        return CommandLine.Fail<DmlDevice>(exception.Message);
    }
}

// The per-stage peaks always print: they cost nothing and they are the answer
// to "would this have fit on a smaller card". The full history goes to a file
// only when asked for — one row per submission, for plotting a run or for
// diffing two of them.
static void Report(DmlDevice device, CommandLine line)
{
    string summary = device.Memory.Summarize();
    if (summary.Length > 0)
    {
        Console.WriteLine(summary);
    }
    if (line.Value("--memory-log") is string logPath)
    {
        device.Memory.WriteCsv(logPath);
        Console.WriteLine($"Wrote {logPath}");
    }
}
