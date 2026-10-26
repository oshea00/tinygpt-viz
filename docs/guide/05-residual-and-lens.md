# Residual stream and logit lens

## The residual stream

Each position carries one vector through the whole network: the **residual stream**. The embeddings start it; then every attention and MLP sublayer reads it, computes something, and **adds** its result. Nothing overwrites the stream. It is a shared workspace that every layer writes into.

![The residual stream tab](../images/residual.png)

The residual stream tab follows the selected position (click a token chip to change it):

- **Position through the stream**: one row per snapshot, after the embedding and after each attention and MLP. Switch to **what each sublayer writes** to see the additions instead; those rows add up to the final vector. Magnitudes grow with depth, so **symmetric log** helps.
- **Write size**: how big each sublayer's addition is at this position (thick bar), next to the size of the stream after it (thin bar).
- **Write size at every position**: the same for every position at once. Click a column to select that position.

### The gradient highway

On the way back, the loss's gradient reaches each residual add and splits: one part **passes straight through** to the layer below, and another flows **back through the sublayer**. The chart shows the size of each part, layer by layer, from the output down, for the whole sequence or one position.

Because the straight-through part never shrinks, even the first layers get a strong learning signal. This is why deep networks with residual connections train at all.

## The logit lens

The model makes its prediction from the last snapshot only: the final layer norm, then the unembedding. The logit lens applies the same readout to **every** snapshot, showing what the stream would predict if the network stopped there.

![The logit lens](../images/logit-lens.png)

- One row per snapshot, one column per position. Each cell is the top guess for the next token; it is **underlined** when that is the token that actually comes next.
- **shade by** the confidence in the top guess, or by the probability of the actual next token.
- The right column is each row's accuracy over the prompt.
- Click a cell: its top guesses, and a chart of the actual next token's probability through the layers.

Reading down a column shows the guess forming: often wrong in the early rows, then right in the last ones. Some predictions are easy enough to read straight from the embeddings; others only appear in the final layers.
