# cv/vit — Vision Transformer

ViT treats an image patch as a word. Cut the image into a grid of fixed `p×p` patches, flatten
each and project it through one linear layer, and the image becomes a sequence of tokens that a
standard Transformer encoder consumes exactly like text. The defining move is everything it
*drops*: the two inductive biases that make a CNN a CNN — locality and translation equivariance —
are gone, leaving a single image-specific assumption, "cut into 2-D patches." Self-attention has a
global receptive field from the first layer. The trade is blunt — with little data ViT trails CNNs
(no priors to lean on), with enough data its ceiling is higher: inductive bias is a shortcut on
small data and a ceiling on large data. A learnable `[CLS]` token aggregates the sequence into one
vector for the classifier; learnable position embeddings supply the order attention is otherwise
blind to. It is the backbone under CLIP, SigLIP, MAE, and the vision tower of essentially every VLM.

> **References** — Dosovitskiy et al., *An Image is Worth 16×16 Words: Transformers for Image
> Recognition at Scale*, ICLR 2021 ([arXiv:2010.11929](https://arxiv.org/abs/2010.11929)) · official
> [google-research/vision_transformer](https://github.com/google-research/vision_transformer) ·
> reference impl: `vision_transformer.py` in
> [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models).

```
image   [B, 3, 32, 32]
   │  patchify (unfold → flatten)      patch 4 → 8×8 = 64 patches, each 3·4·4 = 48
   ▼
patches [B, 64, 48]
   │  Linear(48 → D)  ·  prepend [CLS]  ·  + learnable pos
   ▼
tokens  [B, 65, D]
   │  Transformer encoder × L     (pre-norm · no causal mask · scale 1/√d_head)
   ▼
        [B, 65, D]
   │  take token 0 ([CLS])  →  Linear(D → n_classes)
   ▼
logits  [B, n_classes]
```

Patchify is `unfold → flatten → Linear`, which is exactly a stride-`p` convolution with a `p×p`
kernel — written here as `view`+`permute`+`reshape`, no einops. The encoder block is the *classic*
Transformer, not the modern LLM variant in [`../../llm`](../../llm): `LayerNorm` (not RMSNorm), a
GELU FFN (not gated SwiGLU), plain multi-head attention (not GQA), and — the one that matters —
**no causal mask**, so every patch attends to every other. Scores scale by `1/√d_head` (per-head
dim, not `√D`). The readout is the `[CLS]` output `x[:,0]`; global average pooling over the patch
tokens works about as well and is what SimpleViT uses. Set `num_classes=0` and the head is dropped —
`forward` returns the full `[B, N, D]` token sequence, i.e. ViT as a pure encoder / vision tower,
the interface MAE, CLIP and VLM connectors consume.

## Experiment (`python vit.py`)

Four non-training assertions — forward plus one backward, no dataset — on a CIFAR-sized config
(32×32, patch 4 → 64 patches), so it runs instantly:

```
tokens   64 patches + 1 cls = 65
shape    (2, 3, 32, 32) --ViT--> (2, 10)    params 545,002
encoder  (2, 3, 32, 32) --ViT(classes=0)--> (2, 65, 128)
grad     56/56 weights receive gradient
pos-emb  zeroing positions changes logits  ->  order matters
```

Each line pins one property; together they say *patchify, CLS, positions, encoder and head are all
correct and connected*:

- **contract** — an image `(B,3,32,32)` becomes class logits `(B,10)`. The basic shape contract holds.
- **duality** — with `num_classes=0` the same model returns the token sequence `(B,65,128)` instead
  of logits: drop the head and ViT *is* a vision tower, the encoder interface that feeds
  [`../mae`](../mae) / [`../clip`](../clip) / a VLM connector.
- **wiring** — one backward reaches **56/56** weights. The strongest "nothing is mis-wired" probe,
  and it needs no training: any parameter built but forgotten in `forward` (the CLS token, the
  position table, a block) surfaces immediately as a zero gradient.
- **signature** — zeroing `pos_embedding` *must* change the logits, proving the position embedding
  does real work; without it ViT would treat the patches as an unordered set, since attention alone
  is permutation-equivariant. At init the table is random noise, so this proves position is *wired
  in* — that it encodes real 2-D geometry is what training adds.

This is the **correctness** sanity, not performance. The performance sanity — a few hundred steps on
a CIFAR-10 subset, asserting train accuracy climbs (mechanism works, not chasing SOTA) — is the next
rung, and the baseline the token-budget ablation builds on.

> Next: [`../mae`](../mae) — pretrain this encoder with no labels by masking patches and
> reconstructing; the visual-token-budget ablation that rides on it is [`../tokbudget`](../tokbudget).
