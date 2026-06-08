# llm/swiglu — SwiGLU

SwiGLU replaces the FFN's single activated path with two parallel projections multiplied
together: one branch passes through SiLU and acts as a per-element **gate**, while the other
carries the **content**. Their element-wise product is then projected back down. This uses
three matrices instead of two, so to keep the parameter budget **nearly identical**, the
hidden width shrinks to `⅔·4d`. Llama / PaLM / Qwen all use it.

> **References** — Shazeer, *GLU Variants Improve Transformer*, 2020
> ([arXiv:2002.05202](https://arxiv.org/abs/2002.05202)) · reference impl: `FeedForward` in
> [meta-llama/llama](https://github.com/meta-llama/llama).

```
classic FFN:  out = W2( GELU(W1 x) )                    # 2 matrices, one path
SwiGLU:       out = W_down( SiLU(W_gate x) ⊙ (W_up x) )  # 3 matrices, gate ⊙ content
```

## Experiment (`python swiglu.py`)

A `d=512` SwiGLU against a classic `d→4d→d` FFN:

```
              hidden      params
classic FFN     2048   2,097,152
SwiGLU          1365   2,096,640
```

The classic FFN spends `2·(d·4d) = 8d²` parameters on two matrices. SwiGLU has three
matrices, so it shrinks the hidden width from `4d` to `⅔·4d ≈ 1365`; this gives
`3·(d·⅔·4d) = 8d²` again, up to rounding. The point of the design is to spend the same
parameter budget on **a gate instead of pure width**.

```
shape          (2, 5, 512) -> (2, 5, 512)
```

Shape in equals shape out — SwiGLU is a drop-in replacement for the FFN, so it slots
straight into the [`../nano`](../nano) block. The whole mechanism is the `⊙`: instead of
merely activating a single projection, SwiGLU learns a second projection that decides which
content features should be amplified or suppressed. That input-dependent gating is what
makes it more than a narrower plain FFN.

> Next: [`../gqa`](../gqa); assembled in [`../nano`](../nano).
