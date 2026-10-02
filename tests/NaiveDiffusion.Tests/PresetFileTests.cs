using System.Globalization;
using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Sampling;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Tests;

/// <summary>Presets as files: what goes in comes back out, whatever the
/// culture; names become file names safely; a folder's worth loads with the
/// bad ones passed over; saving replaces and deleting removes by name.</summary>
public class PresetFileTests
{
    private string _folder = "";

    [SetUp]
    public void MakeFolder()
    {
        _folder = Path.Combine(Path.GetTempPath(), "NaiveDiffusion.PresetFileTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_folder);
    }

    [TearDown]
    public void RemoveFolder()
    {
        try
        {
            Directory.Delete(_folder, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    private static Preset Sample() => new()
    {
        Name = "Portrait / soft light",
        Prompt = "1girl, smile, soft light",
        Negative = "lowres",
        Checkpoint = @"C:\models\illustriousXL_v1.safetensors",
        Components = new Dictionary<string, string> { ["vae"] = @"C:\models\vae\fix.safetensors" },
        Loras = new[] { new LoraSpec(@"C:\loras\watercolor_style.safetensors", 0.8f) },
        Width = 832,
        Height = 1216,
        Steps = 28,
        Guidance = 5.5f,
        Sampler = SamplerKind.DpmPlusPlus2MSde,
        Schedule = ScheduleKind.Karras,
        Seed = 12345,
        Strength = 0.4f,
        ClipSkip = 1,
        Sequence = new[] { "sitting at desk", "standing up", "walking out" },
        SequenceSeeds = SequenceSeeds.Continue,
    };

    /// <summary>Field by field: a record holding a list compares the list by
    /// reference, and a list read back is never the array written.</summary>
    private static void AssertSame(Preset? actual, Preset expected)
    {
        Assert.That(actual, Is.Not.Null);
        Assert.That(actual!.Loras, Is.EqualTo(expected.Loras));
        Assert.That(actual.Components, Is.EquivalentTo(expected.Components));
        Assert.That(actual.Sequence, Is.EqualTo(expected.Sequence));
        Assert.That(actual with { Loras = expected.Loras, Components = expected.Components, Sequence = expected.Sequence },
            Is.EqualTo(expected));
    }

    [Test]
    [SetCulture("de-DE")]
    public void RoundTripsWhateverTheCulture()
    {
        string json = PresetFile.Serialize(Sample());
        Assert.That(json, Does.Contain("\"Sampler\": \"DpmPlusPlus2MSde\""),
            "enums go by name so a reordered enum does not change what a file means");
        Assert.That(json, Does.Contain("5.5"), "numbers are invariant");
        AssertSame(PresetFile.Parse(json), Sample());
    }

    [Test]
    public void MissingFieldsTakeDefaults()
    {
        Preset? preset = PresetFile.Parse("{ \"Prompt\": \"cat\" }", fallbackName: "from file");
        Assert.That(preset, Is.Not.Null);
        Assert.That(preset!.Name, Is.EqualTo("from file"));
        Assert.That(preset.Prompt, Is.EqualTo("cat"));
        Assert.That(preset.Steps, Is.EqualTo(20));
        Assert.That(preset.Sampler, Is.EqualTo(SamplerKind.Euler));
        Assert.That(preset.Loras, Is.Empty);
        Assert.That(preset.Components, Is.Empty);
        Assert.That(preset.ClipSkip, Is.EqualTo(2), "SDXL's own layer when the field is missing");
        Assert.That(preset.Sequence, Is.Empty);
        Assert.That(preset.SequenceSeeds, Is.EqualTo(SequenceSeeds.Same));
    }

    [Test]
    public void ParseRefusesWhatIsNotAPreset()
    {
        Assert.That(PresetFile.Parse("not json"), Is.Null);
        Assert.That(PresetFile.Parse("null"), Is.Null);
    }

    [Test]
    public void NullsInADocumentReadAsEmpty()
    {
        Preset? preset = PresetFile.Parse(
            "{ \"Name\": null, \"Prompt\": null, \"Loras\": [ {}, { \"Path\": \"a.safetensors\", \"Weight\": 1 } ], " +
            "\"Components\": { \"vae\": \"\", \"\": \"x\", \"text_encoder\": \"t.safetensors\" } }",
            fallbackName: "file");
        Assert.That(preset, Is.Not.Null);
        Assert.That(preset!.Name, Is.EqualTo("file"));
        Assert.That(preset.Prompt, Is.EqualTo(""));
        Assert.That(preset.Negative, Is.EqualTo(""));
        Assert.That(preset.Loras.Select(lora => lora.Path), Is.EqualTo(new[] { "a.safetensors" }));
        // A part without a file, or a file without a part, is nothing to look for.
        Assert.That(preset.Components, Is.EquivalentTo(
            new Dictionary<string, string> { ["text_encoder"] = "t.safetensors" }));
        Assert.That(PresetFile.Parse("{ \"Components\": null }")!.Components, Is.Empty);
        // A blank step is no step.
        Assert.That(PresetFile.Parse("{ \"Sequence\": [ \"a\", \"\", \"  \", \"b\" ] }")!.Sequence,
            Is.EqualTo(new[] { "a", "b" }));
        Assert.That(PresetFile.Parse("{ \"Sequence\": null }")!.Sequence, Is.Empty);
    }

    [TestCase("Portrait / soft light", "Portrait _ soft light.json")]
    [TestCase("  padded  ", "padded.json")]
    [TestCase("dots...", "dots.json")]
    [TestCase("???", "___.json")]
    [TestCase("", "preset.json")]
    public void FileNamesAreSafe(string name, string expected)
    {
        Assert.That(PresetFile.FileName(name), Is.EqualTo(expected));
    }

    [Test]
    public void LongNamesAreCutToFit()
    {
        string file = PresetFile.FileName(new string('a', 300));
        Assert.That(file.Length, Is.LessThanOrEqualTo(100 + PresetFile.Extension.Length));
        Assert.That(file, Does.EndWith(PresetFile.Extension));
    }

    [Test]
    public void SavesLoadsAndDeletesByName()
    {
        Preset first = Sample();
        Preset second = Sample() with { Name = "zebra", Prompt = "zebra" };
        Preset third = Sample() with { Name = "apple", Prompt = "apple" };
        PresetFile.Save(_folder, first);
        PresetFile.Save(_folder, second);
        PresetFile.Save(_folder, third);

        IReadOnlyList<Preset> loaded = PresetFile.LoadAll(_folder);
        Assert.That(loaded.Select(preset => preset.Name),
            Is.EqualTo(new[] { "apple", "Portrait / soft light", "zebra" }), "sorted by name");
        AssertSame(loaded[1], first);

        Assert.That(PresetFile.Delete(_folder, "ZEBRA"), Is.True, "names match without regard to case");
        Assert.That(PresetFile.Delete(_folder, "zebra"), Is.False);
        Assert.That(PresetFile.LoadAll(_folder).Select(preset => preset.Name),
            Is.EqualTo(new[] { "apple", "Portrait / soft light" }));
    }

    [Test]
    public void SavingUnderTheSameNameReplaces()
    {
        PresetFile.Save(_folder, Sample());
        PresetFile.Save(_folder, Sample() with { Name = "portrait / SOFT light", Prompt = "changed" });

        IReadOnlyList<Preset> loaded = PresetFile.LoadAll(_folder);
        Assert.That(loaded, Has.Count.EqualTo(1));
        Assert.That(loaded[0].Prompt, Is.EqualTo("changed"));
        Assert.That(Directory.GetFiles(_folder), Has.Length.EqualTo(1), "the old spelling's file is gone");
    }

    [Test]
    public void TwoNamesThatMapToOneFileNameKeepAFileEach()
    {
        PresetFile.Save(_folder, Sample() with { Name = "a:b", Prompt = "colon" });
        PresetFile.Save(_folder, Sample() with { Name = "a_b", Prompt = "underscore" });
        // Saved again, each goes back to its own file rather than to a third.
        PresetFile.Save(_folder, Sample() with { Name = "a_b", Prompt = "underscore, again" });
        PresetFile.Save(_folder, Sample() with { Name = "a:b", Prompt = "colon, again" });

        IReadOnlyList<Preset> loaded = PresetFile.LoadAll(_folder);
        Assert.That(loaded.Select(preset => (preset.Name, preset.Prompt)), Is.EquivalentTo(new[]
        {
            ("a:b", "colon, again"), ("a_b", "underscore, again"),
        }));
        Assert.That(Directory.GetFiles(_folder), Has.Length.EqualTo(2), "and nothing half-written is left");

        Assert.That(PresetFile.Delete(_folder, "a_b"), Is.True);
        Assert.That(PresetFile.LoadAll(_folder).Single().Name, Is.EqualTo("a:b"),
            "deleting one does not take the file its name shares with the other");
    }

    [Test]
    public void ARenamedFileStillLoadsAsWhatItSays()
    {
        string path = PresetFile.Save(_folder, Sample());
        string renamed = Path.Combine(_folder, "renamed by hand.json");
        File.Move(path, renamed);

        Assert.That(PresetFile.LoadAll(_folder).Single().Name, Is.EqualTo("Portrait / soft light"));
        Assert.That(PresetFile.Delete(_folder, "Portrait / soft light"), Is.True);
        Assert.That(File.Exists(renamed), Is.False);
    }

    [Test]
    public void AFileThatIsNotAPresetIsPassedOver()
    {
        PresetFile.Save(_folder, Sample());
        File.WriteAllText(Path.Combine(_folder, "broken.json"), "{ not json");
        File.WriteAllText(Path.Combine(_folder, "notes.txt"), "not a preset");

        Assert.That(PresetFile.LoadAll(_folder), Has.Count.EqualTo(1));
    }

    [Test]
    public void AFolderThatIsNotThereHoldsNone()
    {
        string missing = Path.Combine(_folder, "missing");
        Assert.That(PresetFile.LoadAll(missing), Is.Empty);
        Assert.That(PresetFile.Delete(missing, "anything"), Is.False);
    }
}
