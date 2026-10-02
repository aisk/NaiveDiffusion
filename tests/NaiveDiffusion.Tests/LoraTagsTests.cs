using NaiveDiffusion.Text;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Tests;

/// <summary>The A1111 '&lt;lora:name:weight&gt;' tag: how it is read out of a
/// prompt, what the prompt looks like without it, and how the name turns
/// into a file.</summary>
public class LoraTagsTests
{
    [TestCase("1girl, <lora:x:0.8>, smile", "1girl, smile")]
    [TestCase("<lora:x:0.8>, 1girl", "1girl")]
    [TestCase("1girl, <lora:x:0.8>", "1girl")]
    [TestCase("1girl <lora:x:0.8> smile", "1girl smile")]
    [TestCase("1girl<lora:x:0.8>smile", "1girl smile")]
    [TestCase("1girl <lora:x:0.8>, smile", "1girl, smile")]
    [TestCase("1girl, <lora:x:0.8> smile", "1girl, smile")]
    [TestCase("1girl, <lora:a>, <lora:b>, smile", "1girl, smile")]
    [TestCase("1girl,\n<lora:x:0.8>,\nsmile", "1girl,\nsmile")]
    [TestCase("masterpiece\n<lora:x>\n1girl", "masterpiece\n1girl")]
    [TestCase("<lora:x>", "")]
    [TestCase("  <lora:x>, 1girl  ", "1girl")]
    public void ExtractTidiesTheSeam(string prompt, string stripped)
    {
        Assert.That(LoraTags.Extract(prompt, out IReadOnlyList<LoraTag> loras), Is.EqualTo(stripped));
        Assert.That(loras, Is.Not.Empty);
    }

    [Test]
    public void ExtractReadsNamesWeightsAndTheLycoSpelling()
    {
        string stripped = LoraTags.Extract(
            "a, <lora:watercolor_style:0.8>, <LoRA: sub/name >, <lyco:foo:0.5:0.7>, <lora:bad:x>, b",
            out IReadOnlyList<LoraTag> loras);
        Assert.That(stripped, Is.EqualTo("a, b"));
        Assert.That(loras.Select(lora => lora.Name),
            Is.EqualTo(new[] { "watercolor_style", "sub/name", "foo", "bad" }));
        Assert.That(loras[0].Weight, Is.EqualTo(0.8f));
        Assert.That(loras[1].Weight, Is.EqualTo(1f));
        // Only the first number: A1111's separate text and UNet weights are
        // not a thing here.
        Assert.That(loras[2].Weight, Is.EqualTo(0.5f));
        Assert.That(float.IsNaN(loras[3].Weight));
    }

    [Test]
    public void APromptWithoutTagsComesBackUntouched()
    {
        const string prompt = "1girl, <not a tag>, a < b, c > d";
        Assert.That(LoraTags.Contains(prompt), Is.False);
        Assert.That(LoraTags.Extract(prompt, out IReadOnlyList<LoraTag> loras), Is.SameAs(prompt));
        Assert.That(loras, Is.Empty);
    }

    [Test]
    public void FormatRoundTrips()
    {
        string tag = LoraTags.Format("watercolor_style", 0.75f);
        Assert.That(tag, Is.EqualTo("<lora:watercolor_style:0.75>"));
        LoraTags.Extract(tag, out IReadOnlyList<LoraTag> loras);
        Assert.That(loras, Is.EqualTo(new[] { new LoraTag("watercolor_style", 0.75f) }));
    }

    [Test]
    public void CombineLetsTheLaterEntryForAFileWin()
    {
        IReadOnlyList<LoraSpec> combined = LoraSpec.Combine(
            new[] { new LoraSpec(@"C:\loras\a.safetensors", 1f), new LoraSpec(@"C:\loras\b.safetensors", 0.5f) },
            new[] { new LoraSpec(@"C:\LORAS\A.safetensors", 0.3f) });
        Assert.That(combined, Is.EqualTo(new[]
        {
            new LoraSpec(@"C:\LORAS\A.safetensors", 0.3f),
            new LoraSpec(@"C:\loras\b.safetensors", 0.5f),
        }));
    }

    /// <summary>The ComfyUI layout: models/checkpoints beside models/loras,
    /// with a LoRA filed under a subfolder as A1111 allows.</summary>
    private sealed class Layout : IDisposable
    {
        public string Root { get; } = Path.Combine(Path.GetTempPath(),
            "NaiveDiffusion.LoraTagsTests", Guid.NewGuid().ToString("N"));
        public string Checkpoint => Path.Combine(Root, "models", "checkpoints", "model.safetensors");
        public string Loras => Path.Combine(Root, "models", "loras");
        public string Bar => Path.Combine(Loras, "bar.safetensors");
        public string Foo => Path.Combine(Loras, "sub", "Foo.safetensors");

        public Layout()
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Checkpoint)!);
            Directory.CreateDirectory(Path.GetDirectoryName(Foo)!);
            File.WriteAllBytes(Checkpoint, Array.Empty<byte>());
            File.WriteAllBytes(Bar, Array.Empty<byte>());
            File.WriteAllBytes(Foo, Array.Empty<byte>());
        }

        public void Dispose() => Directory.Delete(Root, recursive: true);
    }

    [Test]
    public void DefaultFoldersAreTheCheckpointsAndTheLorasBesideThem()
    {
        using var layout = new Layout();
        Assert.That(LoraLibrary.DefaultFolders(layout.Checkpoint),
            Is.EqualTo(new[] { Path.GetDirectoryName(layout.Checkpoint), layout.Loras }));
        Assert.That(LoraLibrary.DefaultFolders(Path.Combine(layout.Root, "nowhere", "x.safetensors")),
            Is.Empty);
    }

    [Test]
    public void FindTriesThePathThenTheNameAnywhereBelow()
    {
        using var layout = new Layout();
        string[] folders = { layout.Loras };
        Assert.That(LoraLibrary.Find("bar", folders), Is.EqualTo(layout.Bar));
        Assert.That(LoraLibrary.Find("BAR", folders), Is.EqualTo(layout.Bar).IgnoreCase);
        Assert.That(LoraLibrary.Find("sub/foo", folders), Is.EqualTo(layout.Foo).IgnoreCase);
        Assert.That(LoraLibrary.Find("foo", folders), Is.EqualTo(layout.Foo));
        Assert.That(LoraLibrary.Find("nope", folders), Is.Null);
        Assert.That(LoraLibrary.Find("../checkpoints/model", folders), Is.Null);
        Assert.That(LoraLibrary.Find("", folders), Is.Null);
        Assert.That(LoraLibrary.Find("bar", new[] { Path.Combine(layout.Root, "missing") }), Is.Null);
    }

    [Test]
    public void ResolveSeparatesTheFoundFromTheMissing()
    {
        using var layout = new Layout();
        string prompt = LoraLibrary.Resolve("1girl, <lora:bar:0.5>, <lora:nope>, <lora:foo:x>",
            LoraLibrary.DefaultFolders(layout.Checkpoint),
            out IReadOnlyList<LoraSpec> found, out IReadOnlyList<LoraTag> missing);
        Assert.That(prompt, Is.EqualTo("1girl"));
        Assert.That(found, Is.EqualTo(new[] { new LoraSpec(layout.Bar, 0.5f) }));
        // A weight that is not a number is a miss too, not a file at some
        // weight it did not ask for.
        Assert.That(missing.Select(tag => tag.Name), Is.EqualTo(new[] { "nope", "foo" }));
    }
}
