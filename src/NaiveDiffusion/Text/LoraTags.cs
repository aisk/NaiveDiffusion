using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace NaiveDiffusion.Text;

/// <summary>A LoRA named inside a prompt, A1111's way: '&lt;lora:name:0.8&gt;'.
/// <paramref name="Name"/> is the file's name without its extension, as the
/// tag wrote it; <paramref name="Weight"/> is 1 for a tag that gave none,
/// and NaN for one that gave something that is not a number.</summary>
public sealed record LoraTag(string Name, float Weight);

/// <summary>The '&lt;lora:name:weight&gt;' tags A1111 reads a LoRA out of a
/// prompt with, which is how every prompt shared on Civitai says which LoRAs
/// it was written for. CLIP has no idea what one means, so a prompt pasted in
/// with the tags left in would spend tokens spelling out angle brackets and a
/// file name; this takes them out and hands them to whoever can open a file.
///
/// A1111 also spells them '&lt;lyco:…&gt;', from before LyCORIS files were
/// loaded the same way; those read the same. A tag may carry more numbers
/// after the weight (A1111 lets the text and UNet halves be weighted apart);
/// only the first is read, as the one weight applied here.</summary>
public static partial class LoraTags
{
    [GeneratedRegex(@"<(?:lora|lyco):\s*([^:<>]+?)\s*(?::([^<>]*))?>",
        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex Tag();

    /// <summary>Whether the text holds a LoRA tag at all.</summary>
    public static bool Contains(string text) => Tag().IsMatch(text);

    /// <summary>The prompt with every LoRA tag taken out, and the tags that
    /// were in it, in order. The seams are tidied so that no empty tag is
    /// left behind: '1girl, &lt;lora:x&gt;, smile' comes back as '1girl,
    /// smile', a tag at either end takes its comma with it, and a tag between
    /// two words leaves a space between them. The rest of the prompt — its
    /// line breaks and spacing — is left as typed.</summary>
    public static string Extract(string prompt, out IReadOnlyList<LoraTag> loras)
    {
        var found = new List<LoraTag>();
        var kept = new StringBuilder(prompt.Length);
        int copied = 0;
        foreach (Match match in Tag().Matches(prompt))
        {
            found.Add(new LoraTag(match.Groups[1].Value, Weight(match.Groups[2])));
            kept.Append(prompt, copied, match.Index - copied);
            copied = match.Index + match.Length;

            // What the tag sat between: the kept text's last character that
            // is not whitespace, and the first such one after the tag.
            int last = kept.Length;
            while (last > 0 && char.IsWhiteSpace(kept[last - 1]))
            {
                last--;
            }
            int next = copied;
            while (next < prompt.Length && char.IsWhiteSpace(prompt[next]))
            {
                next++;
            }
            bool commaBefore = last > 0 && kept[last - 1] == ',';
            bool commaAfter = next < prompt.Length && prompt[next] == ',';

            if (last == 0 || (commaBefore && commaAfter))
            {
                // Nothing kept yet, or a separator on each side: the one after
                // the tag goes, with the spacing that followed it.
                if (commaAfter)
                {
                    next++;
                    while (next < prompt.Length && char.IsWhiteSpace(prompt[next]))
                    {
                        next++;
                    }
                }
                copied = next;
            }
            else if (commaBefore && next == prompt.Length)
            {
                // The tag was the last piece: the comma that led to it goes.
                kept.Length = last - 1;
                copied = next;
            }
            else if (commaAfter)
            {
                // A word before, a comma after: the word gets the comma.
                kept.Length = last;
                copied = next;
            }
            else if (last == kept.Length && next == copied)
            {
                // No gap on either side: one is needed, or the neighbours
                // run together.
                kept.Append(' ');
            }
            else if (last < kept.Length)
            {
                // A gap on both sides, or only before: the one before stays.
                copied = next;
            }
            // Only a gap after the tag: it stays, as it was.
        }
        kept.Append(prompt, copied, prompt.Length - copied);
        loras = found;
        return found.Count == 0 ? prompt : kept.ToString().Trim();
    }

    /// <summary>The number after the name, or 1 when there is none.</summary>
    private static float Weight(Group group)
    {
        if (!group.Success)
        {
            return 1f;
        }
        string text = group.Value;
        int colon = text.IndexOf(':');
        if (colon >= 0)
        {
            text = text[..colon];
        }
        text = text.Trim();
        if (text.Length == 0)
        {
            return 1f;
        }
        return float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture,
            out float weight) ? weight : float.NaN;
    }

    /// <summary>A tag written back the way A1111 writes one, which is what the
    /// PNG metadata carries so a saved image's prompt pastes back in.</summary>
    public static string Format(string name, float weight) =>
        string.Create(CultureInfo.InvariantCulture, $"<lora:{name}:{weight:0.##}>");
}
