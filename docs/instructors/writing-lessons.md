# Writing lessons

A lesson is a JSON file in `ui/src/lessons/`. It needs no code: the app picks up every `*.json` there. After adding or changing one, rebuild the UI (`cd ui && npm run build`) and reload the page. While you work, `npm run dev` in `ui/` reloads on every save; open <http://localhost:5173> with the server running.

## A small lesson

```json
{
  "id": "my-heads",
  "order": 11,
  "title": "Heads that look back",
  "summary": "Find a head that attends to the previous token.",
  "steps": [
    {
      "text": "Every head has its own habit. The **head gallery** shows them all for this prompt.",
      "prompt": "Once upon a time, there was a little girl named Lily. She",
      "tab": "attention"
    },
    {
      "text": "Some heads look one token back, whatever it is.",
      "tab": "attention",
      "ask": {
        "question": "Click the head that puts most weight on the **previous** position.",
        "check": { "kind": "headBest", "stat": "prev" },
        "hint": "Look for a badge that says “previous”.",
        "correct": "This head copies information forward from the previous position."
      }
    }
  ]
}
```

## The lesson

| Field | |
| --- | --- |
| `id` | Unique, short. |
| `order` | Position in the lessons menu (1, 2, 3, …). |
| `title`, `summary` | Shown in the menu. |
| `requiresTraining` | Optional: `true` for a lesson that trains the model (the training step tab, replay). Without `--train` on the server it is listed but disabled. |
| `steps` | The steps, in order. |

## A step

Every field except `text` is optional. A step only changes what it names; everything else stays as the learner left it.

| Field | Sets |
| --- | --- |
| `text` | The narration. See [Formatting](#formatting). |
| `prompt` | The prompt (and clears generated text). |
| `tab` | `inspector`, `attention`, `residual`, `lens`, `atlas`, `interventions`, `concepts` or `onestep` (the training step tab, which needs `requiresTraining`). |
| `stage` | The selected stage, by id: `embed.sum`, `L1.mlp.gelu`, `L0.attn.H2.weights`… See the [list of stage ids](../guide/02-pipeline-and-stepper.md#stage-names). |
| `phase` | `forward` (values) or `backward` (gradients). |
| `position` | The selected position: a number, or `"last"`. |
| `dim` | The selected column. |
| `head` | `[layer, head]` for the attention views. |
| `atlasMode` | `values`, `grads`, `flow`, `diff` or `recent`. |
| `lensMode` | `confidence` or `actual`. |
| `residualMode` | `snapshots` or `writes`. |
| `conceptPanel` | `similarity`, `map`, `neighbours`, `drift`, `probes`, `steer`, `attribution`, `neurons` or `patching`. |
| `interventions` | Replaces the active interventions; `[]` clears them. Same form as the server's, e.g. `{"kind": "head", "layer": 1, "head": 2, "mode": "zero"}` or `{"kind": "shuffle_pos", "seed": 1}`. |
| `sampling` | `{"temperature": 0.8, "topK": 5, "topP": null}` for the next-token panel. |
| `kvCache` | `true` or `false`: KV cache mode. |
| `ask` | A question; see below. |
| `variants` | Changes for one kind of model; see [Lessons that adapt to the model](#lessons-that-adapt-to-the-model). |

The learner's **reset this step** applies the step again.

Lessons start by clearing interventions, so each one starts from the plain model.

## Questions

`ask` turns a step into a question that is checked against what the learner clicks:

```json
"ask": {
  "question": "Click a cell in the row for the second **␣a** (position 7).",
  "check": { "kind": "position", "positions": [7] },
  "hint": "Count from 0.",
  "correct": "Compare its colours with row 2: the same vector."
}
```

`hint` and `correct` are optional. The check is one of:

| `check` | Correct when the learner has… |
| --- | --- |
| `{"kind": "stage", "ids": ["L1.mlp.gelu"]}` | selected one of these stages |
| `{"kind": "position", "positions": [3]}` | selected one of these positions (clicking any cell in that row counts) |
| `{"kind": "tab", "tab": "atlas"}` | opened this tab |
| `{"kind": "headBest", "stat": "prev"}` | selected the head that scores highest on `prev`, `self`, `first`, `peak` (sharpness) or `induction` (needs a prompt in which a token repeats) |
| `{"kind": "attnTop"}` | clicked the key the selected head weights most for the current query |
| `{"kind": "bright", "ids": ["L1.mlp.gelu"]}` | clicked a cell of one of these stages whose value is at least half the stage's largest |
| `{"kind": "lensEarly"}` | clicked a logit-lens cell in the top half of the rows whose guess is correct |
| `{"kind": "captured"}` | pressed **capture** in the concepts tab (with the untrained model too, if ticked) |
| `{"kind": "trained"}` | taken a step in the training step tab |
| `{"kind": "patchHandoff"}` | applied the residual-stream patching cell in the last column at the earliest snapshot that recovers at least half the answer |
| `{"kind": "patchTop"}` | applied the head patching cell that recovers most |
| `{"kind": "reveal"}` | pressed **show the answer**: for a question to think about, with the answer in `correct` |

A question must not be answered by its own step: a `tab` check on a step that opens that tab, or a `stage` check on a step that selects that stage, shows ✓ before the learner does anything. The lesson tests catch this. **Reset this step** clears the selection, the clicked lens row, a shown answer and, on a `captured` or `trained` step, the capture or training step.

`headBest`, `attnTop`, `bright`, `lensEarly`, `patchHandoff` and `patchTop` are worked out from the live model. Use them for anything that depends on what the model learned (which head does what): a hard-coded answer would be wrong at another checkpoint or on another model.

## Formatting

Narration, questions, hints and answers are plain text with:

- `**bold**` and `` `code` ``,
- links to the glossary: a `` `code` `` span that names a weight or stage (`` `wte` ``, `` `W_Q` ``, `` `L2.mlp.up` ``; any layer or head number works) links to its entry automatically, and `[[term]]` links plain words to a concept, such as `[[residual stream]]` or `[[the maps|PCA]]` to show other text. A term with no entry is shown as plain text, so a typo does no harm; the glossary's entries are in `ui/src/help/glossary.ts`. Where a stage and a concept share a name (`logits`, `loss`), the stage wins: link the concept by another of its names, such as `[[cross-entropy]]`.
- placeholders filled in from the running model: `{{d}}` (the vector width, 256 on `bpe-small`) and `{{V}}` (the vocabulary size).

Use the placeholders rather than writing *256*, so a lesson stays correct if the model changes.

## Positions and tokens

To find positions for a `position` check, type the prompt into the app and count the chips under the prompt box, from 0. BPE tokens usually include their leading space: `␣upon`, not `upon`.

Lessons are checked against a 2-layer, 2-head test model, so `head` and stage ids must stay within `L0`–`L1` and `H0`–`H1`.

## Checking a lesson

```sh
cd ui && npm test
```

The lesson tests check every lesson against a small test model (2 layers, 2 heads). They check that stage ids exist, heads are in range, tabs are real, placeholders are known, and order numbers run 1, 2, 3… without gaps. Then walk through the lesson in the app.
