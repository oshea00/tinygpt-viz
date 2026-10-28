# Interventions

To learn what part of the network does, change it and see what breaks. The interventions tab changes one thing inside the forward pass and compares the result with the original run.

![Ablating head 2 of layer 1](../images/interventions.png)

## What you can change

- **Ablate a head**: **zero** its output, or replace it with its **mean** over the prompt's positions (keeping its average contribution but removing what is specific to each position).
- **Ablate an MLP neuron**: zero or mean. Select a neuron in the inspector (click a cell of an `mlp.gelu` stage), then use the link to fill it in.
- **Switch off a sublayer**: a block's attention or MLP adds nothing, so the stream passes straight through.
- **Force attention**: a head attends only to one position. Positions before it can't see it, so they attend to themselves.
- **Edit a value**: set any cell selected in the inspector to a number you choose.
- **Shuffle position embeddings**: every position is told the wrong position.

Shortcuts: the **ablate** buttons in the [head gallery](04-attention.md#the-head-gallery), and **add as intervention** in the [concept explorer's steering](10-concepts.md#directions-and-steering).

You can combine several. The header shows how many are active; remove them with × or **clear all**.

## Reading the effect

Every view then shows the changed model. The interventions tab compares it with the original:

- **loss on this prompt**, before and after, and the **first stage changed**: where the effect enters.
- **biggest change, at one position**: the prediction the intervention changed most, with that position's change in loss and the probability of the actual next token before and after. The loss on the prompt averages over every position, so an intervention that matters a lot in a few places can look small there; click the change to select that position.
- **Next token**: the distribution before and after, for the selected position.
- **Where the change goes**: every changed stage in the order they run, with how much it changed relative to its size. Click one to open it in the inspector, which then shows the **change from intervention**.
- **p(actual next token) at each position**, before and after: where the change matters for the predictions.

The pipeline map outlines changed stages in orange.

Two experiments that show **causality**:

- **Edit a value** at position 30 and watch the logits: only positions from 30 onward change, because no position can see a later one.
- **Shuffle the position embeddings**: the predictions fall apart, because the model has lost track of order.

Interventions apply only to the traces you look at. Training never sees them.

Further reading: causal tracing, activation patching and self-repair are in the [bibliography](../bibliography.md#interventions-and-activation-patching).
