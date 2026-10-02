using NaiveDiffusion.Pipeline;
using NaiveDiffusion.Tests.Fixtures;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Tests;

/// <summary>A typed prompt into what the pipeline runs, the same way for
/// a front end and the command line: snippets expanded, the sequence's text
/// in place, the LoRA tags resolved to files, and every step seeded.</summary>
public class RunPlannerTests
{
    private static readonly Dictionary<string, string> Snippets = new(StringComparer.OrdinalIgnoreCase)
    {
        ["miku"] = "hatsune miku, twintails <lora:miku_v2:0.7>",
        ["loop"] = "{loop}",
        ["quality"] = "masterpiece",
    };

    private static string? Resolve(string name) => Snippets.GetValueOrDefault(name);

    private static RunPlan? Plan(string prompt, IReadOnlyList<string>? steps = null,
        string negative = "", IReadOnlyList<string>? folders = null, IReadOnlyList<LoraSpec>? listed = null,
        int seed = 1, int count = 1, bool continueSeeds = false)
        => Plan(prompt, out _, steps, negative, folders, listed, seed, count, continueSeeds);

    private static RunPlan? Plan(string prompt, out PlanProblem? problem, IReadOnlyList<string>? steps = null,
        string negative = "", IReadOnlyList<string>? folders = null, IReadOnlyList<LoraSpec>? listed = null,
        int seed = 1, int count = 1, bool continueSeeds = false)
        => RunPlanner.Plan(prompt, negative, steps ?? Array.Empty<string>(), Resolve,
            listed ?? Array.Empty<LoraSpec>(), folders ?? Array.Empty<string>(), seed, count,
            continueSeeds, out problem);

    [Test]
    public void APlainPromptIsOneStepWithItsSeeds()
    {
        RunPlan plan = Plan("1girl, {quality}", negative: "lowres, {quality}", seed: 5, count: 3)!;
        Assert.That(plan.Negative, Is.EqualTo("lowres, masterpiece"));
        Assert.That(plan.Steps, Has.Count.EqualTo(1));
        StepRun step = plan.Steps[0];
        Assert.That(step.Prompt, Is.EqualTo("1girl, masterpiece"));
        Assert.That(step.Seeds, Is.EqualTo(new[] { 5, 6, 7 }));
        Assert.That(step.InSequence, Is.False);
        Assert.That(step.FilePrefix, Is.Empty);
    }

    [Test]
    public void ASequenceFillsTheStepAndRepeatsOrContinuesTheSeeds()
    {
        string[] steps = { "sitting", "standing" };
        RunPlan same = Plan("{quality}, {step}", steps, seed: 10, count: 2)!;
        Assert.That(same.Steps.Select(step => step.Prompt),
            Is.EqualTo(new[] { "masterpiece, sitting", "masterpiece, standing" }));
        Assert.That(same.Steps.Select(step => step.Seeds), Is.EqualTo(new[] { new[] { 10, 11 }, new[] { 10, 11 } }));
        Assert.That(same.Steps.Select(step => step.FilePrefix), Is.EqualTo(new[] { "s01-", "s02-" }));

        RunPlan continued = Plan("1girl", steps, seed: 10, count: 2, continueSeeds: true)!;
        Assert.That(continued.Steps.Select(step => step.Seeds), Is.EqualTo(new[] { new[] { 10, 11 }, new[] { 12, 13 } }));
        // No place for the step: it goes on the end.
        Assert.That(continued.Steps[1].Prompt, Does.EndWith("standing"));
    }

    [Test]
    public void SeedsStopAtTheLargestInt()
    {
        Assert.That(RunPlanner.Seeds(int.MaxValue - 1, 3, 0, false),
            Is.EqualTo(new[] { int.MaxValue - 1, int.MaxValue, int.MaxValue }));
        Assert.That(RunPlanner.Seeds(0, 2, 3, true), Is.EqualTo(new[] { 6, 7 }));
    }

    [Test]
    public void AStepWithoutASequenceIsAProblem()
    {
        Assert.That(Plan("1girl, {step}", out PlanProblem? problem), Is.Null);
        Assert.That(problem!.Kind, Is.EqualTo(PlanProblemKind.StepWithoutSteps));
    }

    [Test]
    public void AMissingOrCyclicSnippetIsNamedWithWhereItWas()
    {
        Assert.That(Plan("1girl, {nobody}", out PlanProblem? missing), Is.Null);
        Assert.That(missing, Is.EqualTo(new PlanProblem(PlanProblemKind.MissingSnippet, "nobody", "prompt")));

        Assert.That(Plan("1girl", out PlanProblem? inStep, steps: new[] { "{nobody}" }), Is.Null);
        Assert.That(inStep!.Where, Is.EqualTo("step 1"));

        Assert.That(Plan("{loop}", out PlanProblem? cyclic), Is.Null);
        Assert.That(cyclic!.Kind, Is.EqualTo(PlanProblemKind.CyclicSnippet));
    }

    [Test]
    public void ASnippetsLoraTagIsResolvedForThatRunAndCombinedWithTheList()
    {
        using var folder = new TempFolder();
        string file = folder.File("miku_v2.safetensors");
        File.WriteAllBytes(file, Array.Empty<byte>());
        var listed = new[] { new LoraSpec(folder.File("style.safetensors"), 1f) };

        RunPlan plan = Plan("{miku}, smile", folders: new[] { folder.Path }, listed: listed)!;
        StepRun step = plan.Steps[0];
        Assert.That(step.Prompt, Does.Not.Contain("<lora:"));
        Assert.That(step.PromptLoras, Is.EqualTo(new[] { new LoraSpec(file, 0.7f) }));
        Assert.That(step.Loras, Is.EqualTo(new[] { listed[0], new LoraSpec(file, 0.7f) }));
    }

    [Test]
    public void ALoraTagNothingHasStopsThePlan()
    {
        Assert.That(Plan("{miku}", out PlanProblem? problem), Is.Null);
        Assert.That(problem, Is.EqualTo(new PlanProblem(PlanProblemKind.MissingLora, "miku_v2", "prompt")));

        Assert.That(Plan("1girl <lora:x:abc>", out PlanProblem? weight), Is.Null);
        Assert.That(weight!.Kind, Is.EqualTo(PlanProblemKind.LoraWeightNotANumber));
    }
}
