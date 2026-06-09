# llm/rope — RoPE

RoPE replaces the *added* position vector with a *rotation*. Instead of adding a position
embedding to the input once, it rotates each query and key by an angle proportional to its
position — inside every attention layer, on Q and K only (V untouched). The payoff: for fixed
`q` and `k`, the *positional* part of the `q·k` dot product depends only on the relative offset
between their positions, not on the absolute positions themselves. Zero learnable parameters,
and it extrapolates beyond the trained length more naturally than learned absolute embeddings
(though very long contexts still need scaling tricks like NTK / YaRN). Llama / Qwen / Mistral
all use it.

> **References** — Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*,
> 2021 ([arXiv:2104.09864](https://arxiv.org/abs/2104.09864)) · reference impl:
> `apply_rotary_emb` in [meta-llama/llama](https://github.com/meta-llama/llama).

```
classic:  x₀ = embed(tok) + PE(pos)                          # absolute vector, added once before layer 0
RoPE:     q, k = Wq·x, Wk·x   then   q ← R(m)·q,  k ← R(n)·k  # rotate by position, every layer, Q/K only
          (R(m)q)ᵀ(R(n)k) = qᵀ R(m)ᵀR(n) k = qᵀ R(n−m) k     # R(m)ᵀR(n) = R(n−m): position enters only via the offset
```

The `d_k` features are split into `d_k/2` pairs, each a 2-D vector rotated by `m·θ_i` with its
own frequency `θ_i = base^(−2i/d_k)` — high-frequency pairs change rapidly and distinguish short
offsets, low-frequency pairs change slowly and stay informative over long ranges. This
implementation uses the **split-half** layout, pairing `x_i` with `x_{i+d/2}` (Llama-style), so
`rotate_half(x) = cat(−x[d/2:], x[:d/2])` supplies the 90° term of each pair. `build_rope_cache`
precomputes the `cos`/`sin` of every angle; `apply_rope` is then one line,
`x·cos + rotate_half(x)·sin`.

## Experiment (`python rope.py`)

Rotation preserves shape — RoPE changes *where* a vector points, not its dimensionality:

```
shape  (2, 4, 16, 32) --apply_rope--> (2, 4, 16, 32)
```

Then take one fixed `(q, k)` pair and score `q·k` after placing them at positions
`(offset + s, s)` — here `offset = query_pos − key_pos` (= `m − n`, so the formula's rotation is
`R(−offset)`), held fixed while both slide by `s`:

```
  offset      s=0      s=3      s=6
       1   1.7848   1.7848   1.7848
       4   7.1485   7.1485   7.1485
       8   7.4501   7.4501   7.4501
```

Read it two ways. **Across a row** the score is constant: sliding both tokens by the same `s`
leaves their absolute positions different but the offset unchanged — RoPE doesn't notice.
**Down a column** the score moves with the offset. That single fact — absolute rotation,
relative attention — is the whole mechanism. The `(q, k)` pair is fixed here to isolate the
positional part; in a real model `q` and `k` also depend on token content, so the score is never
distance-only — RoPE only makes its *positional* component depend on the relative offset.

> Next: [`../gqa`](../gqa), then assemble in [`../nano`](../nano).
