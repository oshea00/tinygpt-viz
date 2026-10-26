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
      "text": "Some heads look one {{unit}} back, whatever it is.",
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
| `requires` | Optional: `"char"` or `"bpe"`, for a lesson that only works with that kind of model. On the other kind it is listed but disabled, with the config to start. |
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
| `conceptPanel` | `similarity`, `map`, `neighbours`, `drift`, `probes`, `steer`, `attribution` or `neurons`. |
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
| `{"kind": "headBest", "stat": "prev"}` | selected the head that scores highest on `prev`, `self`, `first` or `peak` (sharpness) |
| `{"kind": "attnTop"}` | clicked the key the selected head weights most for the current query |

`headBest` and `attnTop` are worked out from the live model. Use them for anything that depends on what the model learned (which head does what): a hard-coded answer would be wrong at another checkpoint or on another model.

## Formatting

Narration, questions, hints and answers are plain text with:

- `**bold**` and `` `code` ``,
- placeholders filled in from the running model: `{{unit}}` (*character* or *token*), `{{units}}`, `{{d}}` (the vector width: 64 on `char-small`, 256 on `bpe-small`) and `{{V}}` (the vocabulary size).

Use the placeholders rather than writing *character* or *64*, so a lesson reads correctly on every model.

## Lessons that adapt to the model

Character models and the BPE model read text differently, so a step about positions or particular tokens often needs different content for each. `variants` holds the fields that change, by kind of model (`"char"` or `"bpe"`):

```json
{
  "text": "Each id picks one row of the token embedding table.",
  "prompt": "Once upon a time",
  "ask": { "question": "Click the row for the first **e** (position 3).", "check": { "kind": "position", "positions": [3] } },
  "variants": {
    "bpe": {
      "prompt": "Once upon a time, there was a little girl.",
      "ask": { "question": "Click the row for **␣upon** (position 1).", "check": { "kind": "position", "positions": [1] } }
    }
  }
}
```

A variant's fields replace the step's own (`ask` is replaced as a whole). A variant of `null` leaves the step out for that kind of model:

```json
{ "text": "Zoom in on the striped band at the top of the token table…", "tab": "atlas", "variants": { "char": null } }
```

To find BPE positions, type the prompt into the app running `bpe-small` and count the chips under the prompt box, from 0. BPE tokens usually include their leading space: `␣upon`, not `upon`.

## Which model to write for

- **Mechanics** (attention, the MLP, the residual stream, training): write for `char-small`, and add `bpe` variants where positions or tokens matter. Every number is small enough to read, and the next-token distribution has only 64 bars.
- **Words and meanings**: write for `bpe-small` and set `"requires": "bpe"`.
- `char-tiny` is useful for a first look, where learners should read individual numbers; its 2 layers and 2 heads mean `head` and stage ids must stay within `L0`–`L1` and `H0`–`H1`.

## Checking a lesson

```sh
cd ui && npm test
```

The lesson tests check every lesson as both kinds of model see it, against the smallest model (2 layers, 2 heads). They check that stage ids exist, heads are in range, tabs are real, placeholders are known, and order numbers run 1, 2, 3… without gaps. Then walk through the lesson in the app on each model it supports.
