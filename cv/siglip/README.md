# cv/siglip — Sigmoid Loss for Language–Image Pre-training

SigLIP is CLIP with exactly one thing changed: the loss. CLIP scores a batch of `B` image–text
pairs into a `B×B` similarity matrix and normalizes each row and column with a softmax — the diagonal
(the matched pair) must *win* its row. That softmax couples the whole row through its denominator: to
normalize you need every text in the batch, which is why CLIP leans on very large batches (up to 32k)
and, in distributed training, an all-gather of every feature to form the denominator. SigLIP replaces
the softmax with a **sigmoid applied to every cell independently**: each `(image i, text j)` pair is
its own binary "matched? yes/no." There is no denominator and no competition, so the batch drops from
*part of the loss* back to *just the source of samples* — the matrix can be scored in chunks and the
loss needs no global normalization. The surprising part is that nothing is lost: the working
ingredient of contrastive learning was never the in-batch softmax competition, it was the
matched-vs-mismatched discrimination itself. (Same family of result as RMSNorm dropping centering — a
part assumed mandatory turns out to be detachable.)

The catch is class imbalance. A batch has `B` positive pairs and `B²−B` negatives; at any real batch
size the negatives swamp the positives, and at init the summed gradient is dominated by "push
everything apart," drowning the few "pull matched pairs together" signals. The fix is a learnable
**bias `b`, initialized to −10** (with the temperature `t = exp(t')` initialized to 10): the model
starts predicting "not a match" for *everything*, so the flood of negatives is already satisfied and
only the `B` positives carry gradient from step one. This is the same trick as the foreground-bias
init in focal loss — set the bias to the prior log-odds of the rare class. The temperature is the same
learnable confidence dial as CLIP's `logit_scale`: cosine similarity is bounded in `[−1, 1]`, so
without a scale the logits can never be sharp; `t` scales them up, and because its gradient is `∝
similarity` it climbs during training only as fast as the embeddings actually separate.

> **References** — Zhai et al., *Sigmoid Loss for Language Image Pre-Training*, ICCV 2023
> ([arXiv:2303.15343](https://arxiv.org/abs/2303.15343)) · *SigLIP 2*, 2025
> ([arXiv:2502.14786](https://arxiv.org/abs/2502.14786)) · official (JAX)
> [google-research/big_vision](https://github.com/google-research/big_vision) · cleanest PyTorch
> reference: `SigLipLoss` in `src/open_clip/loss.py` of
> [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip). Prerequisite:
> [`../clip`](../clip) (the two towers + InfoNCE this builds on).

```
images [B,3,32,32]              texts [B,L]
   │ image encoder                 │ text encoder
   ▼                               ▼
 image features [B,d]          text features [B,d]
   │                               │
   └────── L2 normalize ───────────┘
                  │
        logits = t · image @ text.T + b        [B,B]      (CLIP: no + b)
        labels = 2·I − 1        (+1 on the diagonal, −1 off-diagonal)
        loss   = −Σ logσ(labels ⊙ logits) / B             (CLIP: CE(·,arange) twice)
```

Only the loss changes, so the two towers, the projections, the L2 normalization, the toy world and
the training loop are reused verbatim from [`../clip`](../clip): `siglip.py` imports the same
hand-written ViT image tower and small Transformer text tower. The diff against `clip.py` is exactly
three edits — a `+ bias` term in the logits, `labels = 2·eye − 1` instead of `arange`, and a single
`logsigmoid` sum in place of the two cross-entropies. The loss is normalized by `B`, **not `B²`**:
that keeps the per-anchor scale comparable to CLIP's row-wise cross-entropy and preserves the positive
signal (dividing by `B²` would shrink each positive's weight to `1/B`). The positive/negative
imbalance is handled by the *bias*, not by the normalization constant — two separate knobs: `/B`
controls loss scale across batch sizes, `b` controls the imbalance. The temperature keeps CLIP's one
runaway clamp at 100.

A common myth worth correcting: *"every modern VLM vision tower is SigLIP."* The lineage is mixed —
PaliGemma / Idefics2 / LLaVA-OneVision use SigLIP(-SO400M), the InternVL line uses its own InternViT,
Qwen2-VL trains its own dynamic-resolution ViT with 2-D RoPE. The 2024–25 frontier differences live
in resolution handling, token merging, the projector, and positional encoding — not in whether the
tower's contrastive loss is softmax or sigmoid. That is also why this component is a *nice-to-have* on
the CV sidetrack rather than core.

## Experiment (`python siglip.py`)

The same toy world as [`../clip`](../clip) — 18 classes (6 colors × 3 shapes), one short caption each.
Four blocks, each pinning one property of the sigmoid objective:

```
bias     init mean P(match) 0.0001  = sigmoid(b), ~0 -> 'nothing matches' at init
         init loss  b=-10  9.30 (~|b|, only 18 positives speak)   |   b=0 19.47 (>= ln2*B = 12.5 floor, all 306 negatives flood in)
learns   loss 9.30 -> 0.15   diagonal-match 100%   temp 14.6   bias -10.4
zeroshot fresh images vs 18 captions, NO classifier head:  top-1 98%
temp     step ->  temperature  (flat at init, climbs as the towers separate)
              0        10.0      diag 6%
              5        10.0      diag 6%
             50         9.9      diag 11%
            300        10.1      diag 58%
           1199        14.6      diag 99%
drown    frozen bias, B=6, equal steps & lr -- the negative flood made visible:
           b=  -10 (frozen)   diagonal-match 63%
           b=    0 (frozen)   diagonal-match 17%
```

What each block means:

- **bias** — the load-bearing init. At step 0 the model predicts `P(match) ≈ σ(b) ≈ 0` for every
  pair: "nothing matches." With `b=−10` the init loss `≈ |b|` because only the `B` positives
  contribute; with `b=0` it jumps to ~19 — above the `ln2·B` coin-flip floor — as all `B²−B` negatives
  flood in. That gap is exactly what the bias exists to prevent.
- **learns / zeroshot** — the per-pair sigmoid aligns the towers (loss `9.3 → 0.15`, diagonal 100%,
  zero-shot 98%) with *no* softmax denominator. Note the bias drifts to `−10.4`: it is learnable but
  stays negative, parking the decision boundary `sim = −b/t` in the gap between matched and mismatched
  similarities.
- **temp** — temperature sits flat (~10) while the embeddings are still random (diag 6–11%), then
  climbs to 14.6 once they separate (diag → 99%). Its gradient is `∝ similarity`, so it earns
  confidence only as fast as the representation does — it cannot be cranked up for free, because a high
  `t` amplifies mistakes just as much as correct margins.
- **drown** — the imbalance, isolated. With the bias *frozen* so it cannot self-correct, `b=−10`
  reaches 63% while `b=0` stalls at 17% (= chance at `B=6`): the negative flood, made visible. A
  *learnable* bias mostly closes this gap on a toy (it learns to go negative on its own), which is why
  the effect shows up as a head-start that widens at the 32k batches SigLIP actually uses.

This is a **mechanism** check, not a benchmark — the toy is clean, so the high numbers only confirm the
wiring is correct. As in [`../clip`](../clip), the "zero-shot" test is not true unseen-class: all 18
captions appear during training, so it shows classification *without a trained head*, not
generalization to a held-out color–shape pair.

> Next: back to [`../`](../) for the visual-token-budget sidetrack —
> [`../tokbudget`](../tokbudget) / [`../framesample`](../framesample) /
> [`../tokmerge`](../tokmerge) / [`../mrope`](../mrope) — where the system bottlenecks live. SigLIP is
> the pre-training-objective detour off that line, not part of its trunk.
