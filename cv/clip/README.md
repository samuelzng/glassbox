# cv/clip — Contrastive Language–Image Pre-training

CLIP learns a shared image–text space from nothing but pairing. Two encoders — an image tower (a
[ViT](../vit)) and a text tower (a Transformer) — each project their input into one common space and
L2-normalize it; for a batch of B matched (image, caption) pairs, the B×B matrix of cosine
similarities is scored so the diagonal (the true pairs) is high and every off-diagonal cell low.
That is the *entire* supervision: no class labels, only *which caption goes with which image*, with
each image's B−1 non-partners serving as free negatives — so a larger batch is a harder, stronger
contrast (CLIP trains at a 32k batch). The defining payoff is what alignment buys. A normal
classifier hard-codes its label set in the output layer; CLIP turns the classes into a *runtime*
input — embed whatever prompts you like and rank — converting vision from "recognize these N things"
into open-vocabulary, promptable recognition. Zero-shot classification falls straight out of the
shared space, and that same space underwrites image–text retrieval, dataset filtering (LAION), the
text encoder of Stable Diffusion, and the vision tower of essentially every VLM (LLaVA-style models
consume a CLIP-pretrained ViT).

> **References** — Radford et al., *Learning Transferable Visual Models From Natural Language
> Supervision*, ICML 2021 ([arXiv:2103.00020](https://arxiv.org/abs/2103.00020)) · official
> [openai/CLIP](https://github.com/openai/CLIP) · community
> [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip).

```
images [B,3,32,32]                    texts [B, L] word-ids
   │  ViT encoder (head dropped)         │  Embedding + Transformer + pool
   ▼                                     ▼
[B, d_v] ─ Linear(d_v→d) ─┐     ┌─ Linear(d_t→d) ─ [B, d_t]      both bias=False
                          ▼     ▼
                 zi, zt  [B, d]  ── L2-normalize ──▶  unit sphere
                          │
   logits = exp(logit_scale) · zi @ ztᵀ              [B, B]   (cosine × temperature)
   loss   = ½·( CE(logits, arange B) + CE(logitsᵀ, arange B) )    diagonal = the positives
```

Each projection is a *single* bias-free linear layer, no nonlinearity: the towers already did all
the nonlinear feature extraction, so the projection only has to rotate each modality into a shared
coordinate frame where a dot product means alignment. L2-normalization lands every vector on the
unit sphere, which is what makes `zi @ ztᵀ` read as cosine similarity — directions, not magnitudes,
so the only remaining scale knob is the temperature. That temperature is stored as a learnable
`log(1/τ)` (init `log(1/0.07) ≈ 2.66`) and read back with `.exp()`: a scale must stay positive and
is inherently multiplicative, and `exp` is the bridge from the additive world gradient descent steps
in to the multiplicative world a scale lives in — it is clamped at `ln 100` so the loss can't sharpen
it to infinity. The objective is *symmetric*: `arange(B)` cross-entropy down the rows (image→text)
plus the same over `logits.t()` (text→image), the transpose reusing one matrix for both directions.
This is the *contrastive* route to image–text — [`../mae`](../mae) is pure-vision self-supervision
(reconstruct masked pixels, no text), and [`../vl`](../vl) keeps the image as a *sequence* of tokens
fed into an LLM rather than pooling it to one aligned vector. CLIP commits the whole image to a
single point on the sphere: its power (one directly comparable vector) and its ceiling (fine
compositional / counting / attribute-binding detail is averaged away).

## Experiment (`python clip.py`)

Four assertions on one CPU in under a minute, on an 18-class toy (6 colors × 3 shapes) that reuses
[`../vit`](../vit) as the image tower (head dropped) and its Transformer block as the text tower —
each batch draws *distinct* classes so the diagonal `arange(B)` relies on is exact:

```
contract zi,zt unit on R^128 sphere, shape (18, 128)   init diag-acc 6% (chance 6%)
         init loss  scaled 2.96  |  unscaled 2.89  (ln B = 2.89)   <- no signal at init
learns   loss 2.96 -> 0.01   diagonal-match accuracy 96%   logit_scale 37.1
zeroshot fresh images vs 18 class captions, NO classifier head:  top-1 98%
batch    B=4 (3 neg) 29%   B=9 (8 neg) 44%   B=18 (17 neg) 98%
```

Each line pins one claim; together they say *the two towers align into one space and zero-shot
classification falls out of it*:

- **contract** — both towers emit unit vectors on a shared R¹²⁸ sphere, and at random init the
  diagonal is not special: top-1 sits at chance (6% ≈ 1/18) and the *unscaled* cross-entropy lands on
  `ln 18 = 2.89` to the digit. The *scaled* loss is 2.96, higher by exactly what `logit_scale =
  1/0.07 ≈ 14.3` does to random cosine noise — the temperature is legible before a single step.
- **alignment** — symmetric InfoNCE collapses the loss 2.96 → 0.01 and lifts diagonal-match accuracy
  6% → 96%. `logit_scale` climbs to 37.1 on its own (the loss always prefers a sharper softmax), held
  off the rails by `clamp(max = ln 100)` — the one CLIP-specific line in the loop.
- **zero-shot** — encode the 18 class captions once, then classify *freshly rendered* images by
  nearest caption: 98% top-1, from a classifier that was never trained. Alignment to a shared space
  *is* an open-vocabulary classifier. Qualifier: all 18 classes were seen in training, so this is
  zero-shot in the "new image, no head" sense, not "unseen class."
- **batch size** — at equal steps and learning rate, zero-shot accuracy rises monotonically with B —
  29% (3 negatives) → 44% (8) → 98% (17) — drawn straight from the toy, the reason CLIP trains at a
  32k batch. Caveat: equal *steps* is not equal *images seen*, so the curve conflates negative count
  with data throughput.

This is the **correctness-and-mechanism** sanity, not a performance result: the toy is built to be
separable, so a clean 96 / 98% means the wiring is right, not that the model is strong. The next
rungs — unseen-*class* (compositional) zero-shot by holding out a (color, shape) pair, a real
tokenizer in place of word-level lookup, and a batch ablation controlled for image budget.

> Next: [`../vl`](../vl) — the other image–text route, feeding visual tokens straight into an LLM
> sequence instead of aligning them in a shared space.
