namespace NaiveDiffusion.Models;

/// <summary>The chosen file is not a checkpoint this library can run, and
/// <see cref="Report"/> says why. It carries the verdict rather than only a
/// sentence because callers word it differently: a front end may have the
/// explanation in several languages, and should not find the command line's
/// English baked into the library. The message is the CLI's wording and the
/// fallback for anywhere that only has an exception to show.</summary>
public sealed class CheckpointNotSupportedException : InvalidOperationException
{
    public CheckpointNotSupportedException(
        CheckpointInspector.CheckpointReport report, string message)
        : base(message)
    {
        Report = report;
    }

    public CheckpointInspector.CheckpointReport Report { get; }
}
