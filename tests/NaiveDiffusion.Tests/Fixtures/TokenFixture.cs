using System.Text.Json;

namespace NaiveDiffusion.Tests.Fixtures;

/// <summary>The tokenization fixtures ComfyUI's tokenizers wrote: one JSON
/// array, one object per prompt, read here so that each family's test only
/// says what its objects hold.</summary>
public static class TokenFixture
{
    public static IEnumerable<JsonElement> Entries(string fileName)
    {
        string path = Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", fileName);
        using JsonDocument json = JsonDocument.Parse(File.ReadAllText(path));
        foreach (JsonElement entry in json.RootElement.EnumerateArray())
        {
            yield return entry;
        }
    }

    /// <summary>One test case per sample, named by its prompt, so a failing
    /// prompt is named and does not hide the ones after it.</summary>
    public static IEnumerable<TestCaseData> Cases<T>(IEnumerable<T> samples, Func<T, string> prompt) =>
        samples.Select(sample => new TestCaseData(sample).SetName($"{{m}}({prompt(sample)})"));
}
