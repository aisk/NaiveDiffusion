using NaiveDiffusion.Dml;

namespace NaiveDiffusion.Cli;

/// <summary>What one command accepts and what it does. <paramref name="Usage"/>
/// is its line of the usage text; <paramref name="OpensDevice"/> says
/// whether it takes the adapter, memory-log and hog options and runs with a
/// device — the text-only commands do not.</summary>
internal sealed record Command(string Name, string Usage, string[] ValueOptions,
    string[] FlagOptions, bool OpensDevice, Func<DmlDevice?, CommandLine, int> Run);

/// <summary>Every command, with its usage line beside the options it
/// parses so the two cannot drift apart unnoticed: the tests check that
/// each option appears in its line and nothing in the line is unknown.</summary>
internal static class Commands
{
    /// <summary>The options every device-opening command takes on top of its own.</summary>
    public static readonly string[] DeviceOptions = { "--adapter", "--memory-log", "--hog" };

    public static readonly IReadOnlyList<Command> All = new[]
    {
        new Command("devices", "devices", Array.Empty<string>(), Array.Empty<string>(), false,
            (_, _) => DevicesCommand.Run()),
        new Command("tags", "tags <prompt> [--set name=text ...] [--snippet-dir folder ...]",
            TagsCommand.ValueOptions, Array.Empty<string>(), false,
            (_, line) => TagsCommand.Run(line)),
        new Command("smoke", "smoke", Array.Empty<string>(), Array.Empty<string>(), true,
            (device, _) => SmokeTest.Run(device!)),
        new Command("quantprobe", "quantprobe", Array.Empty<string>(), Array.Empty<string>(), true,
            (device, _) => QuantProbe.Run(device!)),
        new Command("attnprobe", "attnprobe", Array.Empty<string>(), Array.Empty<string>(), true,
            (device, _) => AttentionProbe.Run(device!)),
        // The text towers run on the CPU, so encode has no device to open.
        new Command("encode",
            "encode <prompt> [--dump file] [--dump-hidden file] [--clip-skip n] " +
            CommandLine.ComponentUsage() + " --checkpoint file",
            EncodeCommand.ValueOptions, Array.Empty<string>(), false,
            (_, line) => EncodeCommand.Run(line)),
        // Reads the weights off the mapping on the CPU; nothing is built.
        new Command("inspect", "inspect [--int8] --checkpoint file",
            InspectCommand.ValueOptions, InspectCommand.FlagOptions, false,
            (_, line) => InspectCommand.Run(line)),
        new Command("roundtrip",
            "roundtrip <image> [--size n] [--tile-encode] [--tile-decode] [--dump-latent file] [--dump-decoded file] " +
            CommandLine.ComponentUsage() + " --checkpoint file",
            RoundtripCommand.ValueOptions, RoundtripCommand.FlagOptions, true,
            (device, line) => RoundtripCommand.Run(device!, line)),
        new Command("denoise",
            "denoise <prompt> [--size n | --width n --height n] [--sigma s] [--latent file] " +
            "[--seed n] [--dump file] [--context file] [--lora file[:weight] ...] [--unet-vram MiB] " +
            "[--int8] [--fp16-compute] " +
            CommandLine.ComponentUsage() + " --checkpoint file",
            DenoiseCommand.ValueOptions, DenoiseCommand.FlagOptions, true,
            (device, line) => DenoiseCommand.Run(device!, line)),
        new Command("generate",
            "generate <prompt> [--negative text] [--size n | --width n --height n] " +
            "[--steps n] [--seed n] [--count n] " +
            "[--guidance g] [--sampler euler|euler-a|dpmpp-2m|dpmpp-2m-sde] " +
            "[--schedule leading|linspace|karras|exponential|ays] [--clip-skip n] [--out file] " +
            CommandLine.ComponentUsage() + " [--unet-vram MiB|auto] [--int8] [--fp16-compute] [--whole-vae] " +
            "[--image file --strength 0..1] [--lora file[:weight] ...] [--lora-dir folder ...] " +
            "[--set name=text ...] [--snippet-dir folder ...] [--step text ...] " +
            "[--step-seeds same|continue] --checkpoint file",
            GenerateCommand.ValueOptions, GenerateCommand.FlagOptions, true,
            (device, line) => GenerateCommand.Run(device!, line)),
    };

    public static Command? Find(string name) =>
        All.FirstOrDefault(command => command.Name == name);

    /// <summary>The whole usage text, one command per line.</summary>
    public static string Usage =>
        "commands (those that open a device accept --adapter n (see devices), " +
        "--memory-log file.csv and --hog mib):\n" +
        string.Join("\n", All.Select(command => "  " + command.Usage));

    /// <summary>The options a command's line is parsed against: its own plus
    /// the device ones where it opens a device.</summary>
    public static string[] ValueOptionsOf(Command command) => command.OpensDevice
        ? command.ValueOptions.Concat(DeviceOptions).ToArray()
        : command.ValueOptions;
}

/// <summary>Every adapter DirectML could run on, with its index for --adapter.</summary>
internal static class DevicesCommand
{
    public static int Run()
    {
        foreach (DmlDevice.AdapterInfo adapter in DmlDevice.EnumerateAdapters())
        {
            Console.WriteLine($"{adapter.Index}: {adapter.Name} " + (adapter.Software
                ? "(software: smoke and VAE only, too small and too slow to generate)"
                : $"({adapter.DedicatedMemory / (double)(1UL << 30):0.0} GiB)"));
        }
        return 0;
    }
}
