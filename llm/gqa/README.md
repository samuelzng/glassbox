# llm/gqa — Grouped-Query Attention

Grouped-Query Attention keeps all `H` query heads but projects K and V to fewer heads,
`H_kv < H`, with each K/V head shared by a group of `H/H_kv` query heads. The counter-intuitive
part: this saves almost no training compute. Every query head still forms its own `[L, L]` score
table, so the attention matmuls are unchanged — the only thing that shrinks is the K/V
*projection*, a rounding error next to the FFN. What GQA actually buys is **inference**: the
KV-cache stores `H_kv` heads instead of `H`, so it is `H/H_kv×` smaller, and autoregressive
decode — which is bound by reading that cache out of HBM, not by FLOPs — has that much less to
move each step. MHA is the `H_kv = H` end of the spectrum, MQA the `H_kv = 1` end; GQA sits in
between. Llama 2 70B / Mistral / Qwen all use it.

> **References** — Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from
> Multi-Head Checkpoints*, EMNLP 2023 ([arXiv:2305.13245](https://arxiv.org/abs/2305.13245)) ·
> predecessor MQA: Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, 2019
> ([arXiv:1911.02150](https://arxiv.org/abs/1911.02150)) · reference impl: `repeat_kv` in
> [huggingface/transformers](https://github.com/huggingface/transformers).

```
MHA:  wq, wk, wv : d → H·d_k       H query heads ↔ H kv heads
GQA:  wq         : d → H·d_k       H query heads
      wk, wv     : d → H_kv·d_k    H_kv kv heads, each shared by H/H_kv queries   ← the only change
      scores : still H separate [L,L] tables  → no FLOP saving
      cache  : only H_kv heads                → H/H_kv× smaller                   ← the whole point
```

The `H` query heads split into `H_kv` groups of `H/H_kv`; group `g` (the consecutive heads
`g·n_rep … g·n_rep+n_rep−1`) all attend through K/V head `g`. There are two honest ways to run the
attention given that head-count mismatch. **C — broadcast** (`forward_grouped`): reshape the query
heads into their groups (`q.unflatten(1, (H_kv, n_rep))`) and give K/V a size-1 axis
(`k.unsqueeze(2)`) so the shared head broadcasts across the group — no copy, but it still
materializes the full `[B, H, L, L]` scores. **D — fused** (`forward`):
`F.scaled_dot_product_attention(..., enable_gqa=True)` groups inside the kernel and tiles, so the
`L×L` scores never reach HBM; it is the only path that removes the `O(L²)` term, and it falls back
to C on torch < 2.5. (The textbook `repeat_kv` is just C with the broadcast copied out — same
result, wasted memory — so it is not kept here.)

## Experiment (`python gqa.py`)

The one asymmetry, in the shapes — `q` carries `H` heads, `k,v` stay at `H_kv`:

```
q    (2, 8, 16, 64)   H=8 heads
k,v  (2, 2, 16, 64)   H_kv=2 heads
out  (2, 16, 512)
```

The broadcast path and the fused kernel agree to the bit, and at `H_kv = H` GQA collapses back to
plain MHA:

```
forward (SDPA enable_gqa) == forward_grouped (broadcast)
H_kv = H degenerates to plain MHA
```

Then price the KV-cache at 7B scale (32 layers, `L=8192`, `d_k=128`, bf16):

```
        H_kv       cache    vs MHA
MHA       32     4.00 GiB       1x
GQA        8     1.00 GiB       4x
MQA        1     0.12 GiB      32x
```

The cache shrinks by exactly `H/H_kv` — 4 GiB → 1 GiB moving from 32 KV heads to 8 — because it
only ever stores `H_kv` heads. That number is GQA's entire reason to exist: not fewer FLOPs, just
less to read back each decode step. The quality cost is small (the paper places GQA close to MHA
and well above MQA), which is why 8 KV heads is a common production pick.

> Next: assemble with rope + rmsnorm + swiglu in [`../nano`](../nano); the inference payoff is
> realized in [`../kvcache`](../kvcache).
