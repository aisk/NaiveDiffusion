using NaiveDiffusion.Text;
using NaiveDiffusion.Weights;

namespace NaiveDiffusion.Pipeline;

/// <summary>What stopped a run from being planned: a name in the prompt
/// that nothing answers to, or a file a tag names that is nowhere to be
/// found. <paramref name="Name"/> is the snippet's or the LoRA's;
/// <paramref name="Where"/> is the text it was found in — "prompt",
/// "negative prompt", "step 2" — for the caller's sentence.</summary>
public sealed record PlanProblem(PlanProblemKind Kind, string Name, string Where);

public enum PlanProblemKind
{
    /// <summary>A '{name}' the resolver has no text for.</summary>
    MissingSnippet,

    /// <summary>A snippet that refers to itself, through however many others.</summary>
    CyclicSnippet,

    /// <summary>A '{step}' in a prompt run without a sequence.</summary>
    StepWithoutSteps,

    /// <summary>A '&lt;lora:name&gt;' no folder has a file for.</summary>
    MissingLora,

    /// <summary>A '&lt;lora:name:weight&gt;' whose weight is not a number.</summary>
    LoraWeightNotANumber,
}

/// <summary>One step of a planned run: the prompt as the pipeline gets it —
/// snippets expanded, the step's text in place, LoRA tags taken out — the
/// LoRAs to fold in, and the seeds its images sample with.</summary>
/// <param name="Index">Which step this is, from 0.</param>
/// <param name="Count">How many steps the run has; 1 for a run without a sequence.</param>
/// <param name="Loras">Every LoRA the step folds in: the caller's list with
/// the prompt's tags combined on top.</param>
/// <param name="PromptLoras">Of those, the ones the prompt's tags named —
/// the caller may want to say so, and they stay off any list it keeps.</param>
public sealed record StepRun(int Index, int Count, string Prompt, IReadOnlyList<LoraSpec> Loras,
    IReadOnlyList<LoraSpec> PromptLoras, int[] Seeds)
{
    /// <summary>What goes in a saved file's name to tell the steps apart:
    /// nothing for a run without a sequence.</summary>
    public string FilePrefix => Count > 1 ? $"s{Index + 1:00}-" : "";

    /// <summary>Whether this is one step of several.</summary>
    public bool InSequence => Count > 1;
}

/// <summary>A run as planned: the negative prompt, expanded once, and the
/// steps — one for a run without a sequence.</summary>
public sealed record RunPlan(string Negative, IReadOnlyList<StepRun> Steps);

/// <summary>Turns what was typed into what the pipeline runs, the same way
/// from a front end and from the command line: snippets are expanded first,
/// the sequence's text is put where '{step}' is, then the LoRA tags the
/// prompt carries — its own or a snippet's — are resolved to files, and
/// every step gets its seeds. The pipeline's own checks refuse a prompt
/// that still has a reference or a tag in it, so this is the one place
/// they are taken out.</summary>
public static class RunPlanner
{
    /// <summary>The plan, or null with <paramref name="problem"/> saying
    /// what could not be resolved. The caller words the problem.</summary>
    /// <param name="steps">The sequence's texts, or none.</param>
    /// <param name="resolve">A snippet's text by name, or null for a name
    /// nothing has.</param>
    /// <param name="listed">The LoRAs the caller applies to every step.</param>
    /// <param name="loraFolders">Where a tag's file is looked for.</param>
    /// <param name="continueSeeds">Whether a step's seeds carry on from the
    /// step before, rather than every step sampling with the same ones.</param>
    public static RunPlan? Plan(string prompt, string negative, IReadOnlyList<string> steps,
        Func<string, string?> resolve, IReadOnlyList<LoraSpec> listed,
        IReadOnlyList<string> loraFolders, int seed, int count, bool continueSeeds,
        out PlanProblem? problem)
    {
        // The step's name is left in the prompt for the sequence to fill; with
        // no sequence it is a name nothing answers to.
        string? keep = steps.Count > 0 ? PromptTemplate.StepName : null;
        if (!Expand(prompt, resolve, keep, "prompt", out string expandedPrompt, out problem)
            || !Expand(negative, resolve, null, "negative prompt", out string expandedNegative,
                out problem))
        {
            return null;
        }
        if (steps.Count == 0 && PromptTemplate.HasStep(expandedPrompt))
        {
            problem = new PlanProblem(PlanProblemKind.StepWithoutSteps, PromptTemplate.StepName,
                "prompt");
            return null;
        }

        var prompts = new List<string>();
        if (steps.Count == 0)
        {
            prompts.Add(expandedPrompt);
        }
        for (int i = 0; i < steps.Count; i++)
        {
            if (!Expand(steps[i], resolve, null, $"step {i + 1}", out string text, out problem))
            {
                return null;
            }
            prompts.Add(PromptTemplate.WithStep(expandedPrompt, text));
        }

        var runs = new List<StepRun>(prompts.Count);
        for (int i = 0; i < prompts.Count; i++)
        {
            string text = prompts[i];
            IReadOnlyList<LoraSpec> found = Array.Empty<LoraSpec>();
            if (LoraTags.Contains(text))
            {
                text = LoraLibrary.Resolve(text, loraFolders, out found,
                    out IReadOnlyList<LoraTag> missing);
                if (missing.Count > 0)
                {
                    LoraTag tag = missing[0];
                    problem = new PlanProblem(float.IsFinite(tag.Weight)
                        ? PlanProblemKind.MissingLora
                        : PlanProblemKind.LoraWeightNotANumber, tag.Name,
                        prompts.Count > 1 ? $"step {i + 1}" : "prompt");
                    return null;
                }
            }
            runs.Add(new StepRun(i, prompts.Count, text, LoraSpec.Combine(listed, found), found,
                Seeds(seed, count, i, continueSeeds)));
        }
        problem = null;
        return new RunPlan(expandedNegative, runs);
    }

    /// <summary>Image i of a step gets the seed + i, so a run is reproducible
    /// from one number; a sequence repeats the same seeds every step, so
    /// what moves between two steps is what the step text moved — or counts
    /// on, when asked to, so no two of its images share one.</summary>
    public static int[] Seeds(int seed, int count, int step, bool continueSeeds)
    {
        var seeds = new int[count];
        for (int i = 0; i < count; i++)
        {
            seeds[i] = (int)Math.Min((long)seed + (continueSeeds ? (long)step * count : 0) + i,
                int.MaxValue);
        }
        return seeds;
    }

    private static bool Expand(string text, Func<string, string?> resolve, string? keep,
        string where, out string expanded, out PlanProblem? problem)
    {
        Expansion expansion = PromptTemplate.Expand(text, resolve, keep);
        expanded = expansion.Text;
        if (expansion.Missing.Count > 0)
        {
            string name = expansion.Missing[0];
            problem = new PlanProblem(
                PromptTemplate.NameComparer.Equals(name, PromptTemplate.StepName)
                    ? PlanProblemKind.StepWithoutSteps
                    : PlanProblemKind.MissingSnippet, name, where);
            return false;
        }
        if (expansion.Cyclic.Count > 0)
        {
            problem = new PlanProblem(PlanProblemKind.CyclicSnippet, expansion.Cyclic[0], where);
            return false;
        }
        problem = null;
        return true;
    }
}
