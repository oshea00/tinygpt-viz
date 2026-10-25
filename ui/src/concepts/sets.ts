// Prompt sets for the concept explorer, as editable text:
//
//   @animal: dog cat bird            a category and its words
//   Lily saw a {animal} in the park.  a template with one slot; every word of the category fills it
//
// Lines starting with # are comments. The word filling the slot is the target that gets captured.

export interface PromptTarget { start: number; end: number; word: string; category: string; }
export interface Prompt { text: string; targets: PromptTarget[]; }

export const DEFAULT_SET = `# Words a 3- to 4-year-old knows, in TinyStories-style sentences.
@animal: dog cat bird fish rabbit lion bear frog duck horse
@colour: red blue green yellow pink purple orange brown
@emotion: happy sad angry scared excited tired
@family: mom dad sister brother grandma baby
@place: park house garden school forest beach

One day, a little {animal} went to the park.
Lily saw a {animal} in the garden.
The {animal} was very hungry.
Tom wanted to play with the {animal}.
She had a {colour} ball.
The {colour} flower was very pretty.
He wore a {colour} hat to the park.
Mom gave her a big {colour} box.
Lily felt very {emotion} that day.
The boy was {emotion} because his toy broke.
Tim was so {emotion} when he saw the dog.
Lily went to the park with her {family}.
Her {family} gave her a hug.
Tom asked his {family} for help.
They went to the {place} to play.
The cat ran to the {place}.
`;

export function parseSet(text: string): { prompts: Prompt[]; categories: Record<string, string[]>; errors: string[] } {
  const categories: Record<string, string[]> = {};
  const templates: string[] = [];
  const errors: string[] = [];
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const cat = /^@(\w+):\s*(.*)$/.exec(line);
    if (cat) categories[cat[1]] = cat[2].split(/\s+/).filter(Boolean);
    else templates.push(line);
  }
  const prompts: Prompt[] = [];
  for (const t of templates) {
    const slots = [...t.matchAll(/\{(\w+)\}/g)];
    if (slots.length !== 1) {
      errors.push(`“${t}” needs exactly one {category} slot`);
      continue;
    }
    const category = slots[0][1];
    const words = categories[category];
    if (!words) {
      errors.push(`no @${category}: line for “${t}”`);
      continue;
    }
    const start = slots[0].index!;
    for (const word of words) {
      prompts.push({ text: t.replace(slots[0][0], word), targets: [{ start, end: start + word.length, word, category }] });
    }
  }
  return { prompts, categories, errors };
}
