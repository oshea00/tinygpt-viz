# Prompt sets for the concept explorer

The [concept explorer](../guide/10-concepts.md) compares the model's internal vectors for words, grouped into categories you define. The **prompt set** says which words, in which categories, and in what sentences. Edit it in the app: **concepts → Prompt set → edit**.

## Format

```text
# Lines starting with # are comments.
@animal: dog cat bird fish rabbit lion
@colour: red blue green yellow pink

Lily saw a {animal} in the garden.
She had a {colour} ball.
```

- `@name: word word …` defines a category and its words. The name is only a label: the model never sees it. It is used to colour, group and label the results.
- Every other line is a template with exactly one `{name}` slot. It is filled with every word of that category: one prompt per word.
- The filled-in word is the **target**: the only position recorded from each prompt.

The panel reports the number of prompts and categories, and any template it cannot use.

## Writing a good set

**Use words that are single tokens.** On `bpe-small` most common words are, including the leading space: ` dog` is one token. Rare or long words are split; the explorer then reads the last piece (or the average, with **mean of sub-tokens**), which is less clean. To check a word, type it after a space into the prompt box and count the chips.

**Share templates across categories.** If animals only ever appear in animal sentences and colours in colour sentences, the sentence itself gives the category away. A probe can then score well without the model knowing anything about the words: it is reading the template. The untrained-model baseline exposes this: when it scores well too, suspect the templates. Templates that fit every category avoid the problem:

```text
@animal: dog cat bird fish
@food: apple cake bread soup
@toy: ball doll kite train
Lily looked at the {animal} for a long time.
Lily looked at the {food} for a long time.
Lily looked at the {toy} for a long time.
Tom said the word {animal} again and again.
Tom said the word {food} again and again.
Tom said the word {toy} again and again.
```

The default set in the app does *not* do this, so that it gives natural sentences, and the probes panel points out the confound. It makes a useful first lesson in reading results carefully.

**Have enough words per category.** Probes are tested on held-out words: every fourth word of each category (sorted alphabetically) is never trained on. With fewer than about 6 words per category, very little is left to test on.

**Put the target early, and keep prompts short.** Every position after the target is ignored, and prompts are cut at 100 tokens.

**Use several templates per category.** Then a word appears in several contexts, which [context drift](../guide/10-concepts.md#context-drift) needs, and the results depend less on one particular sentence.

## Guarding against false findings

- Always keep **also the untrained model** ticked, and compare. Structure that the untrained model also shows was not learned.
- Look at the **shuffled labels** bars in the probes panel: they show what chance looks like with this set.
- A probe shows a concept is readable; **steering** tests whether the model uses it. Pair them: find a direction, then check that adding it changes the text the way you expect, and subtracting it the other way.
- Capture again after training or loading a checkpoint: captured vectors do not update themselves.
