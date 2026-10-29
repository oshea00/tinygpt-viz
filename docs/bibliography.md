# Bibliography

Papers behind the views in this app, grouped by the view they relate to. Each note says what the work is and where it shows up. For the terms, see the glossary in the app's **help** panel.

## The model and its data

- **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. & Polosukhin, I. (2017).** Attention Is All You Need. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762). The transformer architecture: attention heads, queries, keys and values, the residual stream. See [Pipeline and stepper](guide/02-pipeline-and-stepper.md) and [Attention](guide/04-attention.md).
- **Eldan, R. & Li, Y. (2023).** TinyStories: How Small Can Language Models Be and Still Speak Coherent English? [arXiv:2305.07759](https://arxiv.org/abs/2305.07759). The dataset `bpe-small` is trained on, and the source of the validation stories that the beliefs tab reads. See [Getting started](guide/01-getting-started.md).

## Reading the residual stream

- **nostalgebraist (2020).** [interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens). LessWrong, 31 August 2020. Applying the unembedding to every layer to see when a prediction forms. See [Residual stream and logit lens](guide/05-residual-and-lens.md).
- **Elhage, N., Nanda, N., Olsson, C., Henighan, T., et al. (2021).** [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html). Transformer Circuits Thread. Reading the network as sublayers that read from and write to a shared residual stream. See [Residual stream and logit lens](guide/05-residual-and-lens.md).

## Attention and induction heads

- **Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., et al. (2022).** In-context Learning and Induction Heads. [arXiv:2209.11895](https://arxiv.org/abs/2209.11895). Heads that complete "A B … A → B", and their previous-token partners. See [Attention](guide/04-attention.md) and lesson 11.

## Interventions and activation patching

- **Meng, K., Bau, D., Andonian, A. & Belinkov, Y. (2022).** Locating and Editing Factual Associations in GPT. NeurIPS 2022. [arXiv:2202.05262](https://arxiv.org/abs/2202.05262). Causal tracing: restoring clean activations in a corrupted run to find where information is stored. See [Interventions](guide/09-interventions.md).
- **Wang, K., Variengien, A., Conmy, A., Shlegeris, B. & Steinhardt, J. (2022).** Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small. [arXiv:2211.00593](https://arxiv.org/abs/2211.00593). The "who gave the ball to whom" task behind the patching example and lesson 12. See [Concepts: activation patching](guide/10-concepts.md#activation-patching).
- **Heimersheim, S. & Nanda, N. (2024).** How to use and interpret activation patching. [arXiv:2404.15255](https://arxiv.org/abs/2404.15255). A practical guide to the patching metrics, including the logit difference and recovery used in the patching panel.
- **McGrath, T., Rahtz, M., Kramár, J., Mikulik, V. & Legg, S. (2023).** The Hydra Effect: Emergent Self-repair in Language Model Computations. [arXiv:2307.15771](https://arxiv.org/abs/2307.15771). Later layers compensating for an ablated one, which is why a single-layer intervention can understate what a component does.

## Probes

- **Alain, G. & Bengio, Y. (2016).** Understanding intermediate layers using linear classifier probes. [arXiv:1610.01644](https://arxiv.org/abs/1610.01644). Linear probes at every layer, as in the concept explorer's probes panel. See [Concepts: probes](guide/10-concepts.md#probes).
- **Hewitt, J. & Liang, P. (2019).** Designing and Interpreting Probes with Control Tasks. EMNLP 2019. [arXiv:1909.03368](https://arxiv.org/abs/1909.03368). Why a probe needs a baseline it should fail on, like the shuffled-label and untrained-model bars.

## Steering

- **Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U. & MacDiarmid, M. (2023).** Steering Language Models With Activation Engineering. [arXiv:2308.10248](https://arxiv.org/abs/2308.10248). Activation addition: adding a vector to the residual stream to change behaviour. See [Concepts: directions and steering](guide/10-concepts.md#directions-and-steering).
- **Panickssery, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E. & Turner, A. M. (2023).** Steering Llama 2 via Contrastive Activation Addition. [arXiv:2312.06681](https://arxiv.org/abs/2312.06681). Steering vectors taken as a difference of means between contrasting examples.

## Attribution

- **Abnar, S. & Zuidema, W. (2020).** Quantifying Attention Flow in Transformers. [arXiv:2005.00928](https://arxiv.org/abs/2005.00928). Attention rollout. See [Concepts: attribution](guide/10-concepts.md#attribution).
- **Shrikumar, A., Greenside, P., Shcherbina, A. & Kundaje, A. (2016).** Not Just a Black Box: Learning Important Features Through Propagating Activation Differences. [arXiv:1605.01713](https://arxiv.org/abs/1605.01713). Includes gradient × input as an attribution baseline.
- **Zeiler, M. D. & Fergus, R. (2013).** Visualizing and Understanding Convolutional Networks. [arXiv:1311.2901](https://arxiv.org/abs/1311.2901). Occlusion: hide part of the input and measure the drop.

## Belief trajectories

- **Bigelow, E., Sarfati, R., Wurgaft, D., Lewis, O., McGrath, T., Merullo, J., Geiger, A. & Lubana, E. S. (2026).** Stories in Space: In-Context Learning Trajectories in Conceptual Belief Space. [arXiv:2605.12412](https://arxiv.org/abs/2605.12412). The paper the beliefs tab repeats: beliefs as paths through a low-dimensional space, probes, shared geometry, steering and entanglement. See [Belief trajectories](guide/13-beliefs.md) and lesson 13.
- **Xie, S. M., Raghunathan, A., Liang, P. & Ma, T. (2021).** An Explanation of In-context Learning as Implicit Bayesian Inference. ICLR 2022. [arXiv:2111.02080](https://arxiv.org/abs/2111.02080). The Bayesian view of in-context learning that the paper builds on.
- **Finke, L., Sreedhara, C., Dooms, T., Allen, M., Zhang, E., Rodriguez, J. D., Nabeshima, N., Marshall, T. & Braun, D. (2025).** Parameterized Synthetic Text Generation with SimpleStories. [arXiv:2504.09184](https://arxiv.org/abs/2504.09184). The story dataset the paper uses; this app uses TinyStories instead.
- **Kriegeskorte, N., Mur, M. & Bandettini, P. (2008).** Representational similarity analysis – connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*, 2. [doi:10.3389/neuro.06.004.2008](https://doi.org/10.3389/neuro.06.004.2008). Comparing two spaces through their distance matrices, as the geometry section does.
- **McInnes, L., Healy, J. & Melville, J. (2018).** UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. [arXiv:1802.03426](https://arxiv.org/abs/1802.03426). The non-linear map the paper uses beside PCA; the app uses PCA only.
- **Russell, J. A. (1980).** A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161–1178. [doi:10.1037/h0077714](https://doi.org/10.1037/h0077714). Emotions arranged by valence and arousal; the paper compares its emotion map with these ratings.
- **Mehrabian, A. (1996).** Pleasure-arousal-dominance: A general framework for describing and measuring individual differences in temperament. *Current Psychology*, 14(4), 261–292. [doi:10.1007/BF02686918](https://doi.org/10.1007/BF02686918). The pleasure–arousal–dominance (PAD) model of emotion.

## Fine-tuning and preference tuning

- **Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L. & Chen, W. (2021).** LoRA: Low-Rank Adaptation of Large Language Models. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685). Training two thin matrices beside a frozen weight instead of the weight itself. See [Fine-tuning](guide/14-fine-tuning.md) and lesson 14.
- **Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., et al. (2022).** Training language models to follow instructions with human feedback. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155). InstructGPT: RLHF with a reward model fitted to human comparisons, then PPO. DPO is the alternative the refusal adapter uses. See [Fine-tuning: preference tuning](guide/14-fine-tuning.md#preference-tuning-with-dpo-teaching-a-refusal).
- **Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D. & Finn, C. (2023).** Direct Preference Optimization: Your Language Model is Secretly a Reward Model. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290). The preference loss the refusal adapter is trained with, and the implicit reward the pairs table shows. See lesson 15.
- **Pang, R. Y., Yuan, W., Cho, K., He, H., Sukhbaatar, S. & Weston, J. (2024).** Iterative Reasoning Preference Optimization. [arXiv:2404.19733](https://arxiv.org/abs/2404.19733). Adds the preferred answer's log-likelihood to the DPO loss, as `core/dpo.py` does: DPO alone can widen the gap only by pushing the rejected answer down.
- **Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W. & Nanda, N. (2024).** Refusal in Language Models Is Mediated by a Single Direction. [arXiv:2406.11717](https://arxiv.org/abs/2406.11717). A difference-of-means direction whose removal stops refusals and whose addition causes them, repeated in the preference-tuning section on this model.
