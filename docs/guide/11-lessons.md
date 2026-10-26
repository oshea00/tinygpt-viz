# Lessons

Lessons are guided tours of the app. Each step sets up the views (a prompt, a tab, a stage, a head) and explains what to look at. Some steps ask you to find something.

![The lessons menu](../images/lessons-menu.png)

Open **lessons** in the header and pick one. A lesson that can't run on this server is greyed out, with what to start the server with: a different `--config`, or `--train` for the lesson that trains the model.

![A lesson step with a question](../images/lesson-player.png)

- **next** and **back** move between steps; **reset this step** sets the views up again if you have wandered off.
- A **?** question checks what you click and turns into a **✓** with an explanation when you find it. **hint** helps; **skip** moves on without answering.
- Close a lesson with ×. Everything it changed stays as it was, so you can keep exploring from there.

## The lessons

| # | Lesson | What it shows |
| --- | --- | --- |
| 1 | Text to numbers | Tokens, and the embedding table. On `bpe-small`, a step on tokens the model never saw |
| 2 | Where am I? Position embeddings | Why order needs its own table, and what happens without it |
| 3 | Attention as search | One head, one query: the dot products behind an attention pattern |
| 4 | Many heads | The head gallery, naming heads, and ablating one |
| 5 | The MLP | Neurons that fire on specific patterns |
| 6 | The residual stream and the logit lens | A shared workspace, and when the prediction forms |
| 7 | From logits to text | Temperature, top-k and top-p |
| 8 | Why only 100 positions? | The context window, and the KV cache |
| 9 | How it learns | The loss, gradients, one training step, and replaying a whole run (needs `--train`) |
| 10 | Where concepts live | Clusters, a probe and a steering test (needs `bpe-small`) |

Lessons adapt to the model you run: on `bpe-small` the questions are about whole-word tokens, the numbers match that model, and some steps differ. Lessons 1–9 work on any model (lesson 9 needs a server started with `--train`); lesson 10 needs the BPE model.

Model-dependent answers, such as "click the sharpest head", are worked out from the live model, so lessons work at any training checkpoint, even partway through training.
