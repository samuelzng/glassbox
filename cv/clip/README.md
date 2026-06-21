# cv/clip — Contrastive Language–Image Pre-training

CLIP learns to put images and text into the same embedding space.

Given a batch of matched image–caption pairs, the image encoder maps each image to a vector and the text encoder maps each caption to a vector. After L2 normalization, their dot product is cosine similarity. For a batch of `B` pairs, we get a `B × B` similarity matrix:

- diagonal cells are the real image–caption pairs
- off-diagonal cells are treated as negatives

The loss pushes the diagonal up and the off-diagonal down. There are no class labels. The only supervision is which caption came with which image.

This is what makes CLIP useful: at test time, classes can be written as text prompts. Instead of training a fixed classifier head, we encode prompts like `"a red circle"` or `"a blue square"` and rank them by similarity to the image. So classification becomes nearest-text matching in the shared space.

    images [B,3,32,32]              texts [B,L]
       │ image encoder                 │ text encoder
       ▼                               ▼
     image features [B,d]          text features [B,d]
       │                               │
       └────── L2 normalize ───────────┘
                      │
            logits = scale · image @ text.T     [B,B]
            loss = CE(logits, arange(B)) + CE(logits.T, arange(B))

The projection layers are just linear maps. The encoders do the nonlinear feature extraction; the projections only move both modalities into the same coordinate system. L2 normalization puts all vectors on the unit sphere, so similarity is about direction, not magnitude. The learnable temperature controls how sharp the similarity scores are.

This implementation is small on purpose. It uses the local ViT as the image tower and a small Transformer as the text tower.

## Experiment

Run:

    python clip.py

The toy dataset has 18 classes: 6 colors × 3 shapes. Each class has a simple text caption. The experiment checks four things:

    contract zi,zt unit on R^128 sphere, shape (18, 128)   init diag-acc 6% (chance 6%)
             init loss  scaled 2.96  |  unscaled 2.89  (ln B = 2.89)
    learns   loss 2.96 -> 0.01   diagonal-match accuracy 96%   logit_scale 37.1
    prompt   fresh images vs 18 class captions, no classifier head: top-1 98%
    batch    B=4 (3 neg) 29%   B=9 (8 neg) 44%   B=18 (17 neg) 98%

What each line means:

- `contract`: both towers output unit vectors in the same dimension. At initialization, the diagonal is not special.
- `learns`: after training, matched image–caption pairs have much higher similarity than mismatched pairs.
- `prompt`: fresh images can be classified by comparing them with text captions, without a trained classifier head.
- `batch`: larger batches give more in-batch negatives, which makes the contrastive task stronger.

This is a mechanism check, not a benchmark. The toy data is simple, so high accuracy only means the CLIP wiring is correct.

One caveat: the prompt test is not true unseen-class zero-shot. All 18 class captions appear during training. It only shows that classification can be done without a classifier head. A stronger next test would hold out a color–shape pair during training and test whether the model can classify that unseen combination.

