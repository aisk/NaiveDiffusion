using NaiveDiffusion.Text;

namespace NaiveDiffusion.Cli;

/// <summary>A prompt cut into tags, with whatever the parser finds wrong with
/// each one, from a place where it can be checked against a prompt in a
/// file.</summary>
internal static class TagsCommand
{
    public static readonly string[] ValueOptions = SnippetOptions.ValueOptions;

    /// <summary>Text only: no model, no device, nothing to open.</summary>
    public static int Run(CommandLine line)
    {
        // Snippets go in first and LoRA tags come out next, as they do on a
        // run: what is left is what the parser and the tokenizer see.
        string typed = line.Argument ?? "astronaut, riding, unicorn, space";
        string expanded = SnippetOptions.Expand(typed, SnippetOptions.Resolver(line), "prompt");
        if (expanded != typed)
        {
            Console.WriteLine($"expanded: {expanded}");
        }
        string prompt = LoraTags.Extract(expanded, out IReadOnlyList<LoraTag> loras);
        IReadOnlyList<PromptTag> tags = PromptTags.Parse(prompt);

        int flagged = 0;
        foreach (PromptTag tag in tags)
        {
            Console.Write($"  {tag.Content}");
            if (tag.Weight != 1f)
            {
                Console.Write($"    x{tag.Weight:0.##}");
            }
            if (tag.Issues.Count > 0)
            {
                flagged++;
                Console.Write($"    <- {string.Join(", ", tag.Issues)}");
            }
            Console.WriteLine();
        }

        foreach (LoraTag lora in loras)
        {
            Console.WriteLine($"  <lora:{lora.Name}>    x{lora.Weight:0.##}    <- applied as a LoRA, not sent to CLIP");
        }

        Console.WriteLine($"{tags.Count} tags, {flagged} with something to fix" +
                          (loras.Count > 0 ? $", {loras.Count} LoRA tags" : ""));
        // What the tokenizer is actually handed, weight syntax taken off and
        // escapes resolved. A bracket meant literally has to survive this line.
        Console.WriteLine("to clip: " +
            string.Concat(PromptTags.Segments(prompt).Select(segment => segment.Text)));
        if (PromptTags.NeedsRepair(prompt))
        {
            Console.WriteLine($"repaired: {PromptTags.Repair(prompt)}");
        }
        // A prompt nothing is wrong with still exits 0; the count is the answer.
        return 0;
    }
}
