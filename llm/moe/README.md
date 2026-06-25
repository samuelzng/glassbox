# llm/moe — Mixture of Experts

MoE replaces one dense FFN with many parallel FFNs plus a router. For each token, the router
scores all experts, keeps only the top-k, and the token is evaluated by those experts only; the
outputs are mixed by the router weights. The expert itself is not new machinery here: each one is
the same [`../swiglu`](../swiglu) FFN, instantiated with its own parameters. The trick is that
capacity and compute split apart. Total parameters grow with the number of experts `E`, while the
per-token FFN work grows only with `k`. Mixtral-style top-2 routing is the standard example:
many experts live in memory, but each token activates only two.

> **References** — Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated
> Mixture-of-Experts Layer*, 2017 ([arXiv:1701.06538](https://arxiv.org/abs/1701.06538)) ·
> Fedus et al., *Switch Transformers*, 2021 ([arXiv:2101.03961](https://arxiv.org/abs/2101.03961)) ·
> Jiang et al., *Mixtral of Experts*, 2024 ([arXiv:2401.04088](https://arxiv.org/abs/2401.04088)).

```
dense FFN:  x ─────────────────────► SwiGLU ─────────────► out

MoE FFN:    x ─► router ─► top-k ─┬► expert 3 (SwiGLU) ─┐
                                  └► expert 7 (SwiGLU) ─┴► weighted sum ─► out
                         all other experts are skipped for this token
```

The router is a tiny linear layer `d → E`, followed by `softmax` and `topk`. With `top_k > 1`, the
kept weights are renormalized inside the top-k set, so the selected expert outputs form a convex
mixture. The implementation is deliberately the slow, readable dispatch: flatten `[B, L, d]` into
tokens, loop over experts, gather the tokens assigned to that expert, run one batched SwiGLU call,
and scatter-add the weighted result. It is the same computation a fused dispatch kernel would do,
just written so the routing is visible.

The failure mode is router collapse. If the main task loss alone makes one expert slightly better,
the router sends more tokens there, that expert trains faster, and the loop reinforces itself while
other experts starve. The Switch-style auxiliary loss pushes router probability toward actual
usage balance:

```
f_i   = fraction of top-k slots assigned to expert i      # hard count, no gradient
P_i   = mean router probability for expert i              # differentiable
L_aux = E · Σ_i f_i P_i
```

Uniform routing gives `L_aux ≈ 1`; collapse toward one expert raises it toward `E`. The gradient
flows through `P_i`, while `f_i` tells the loss which experts are overloaded.

## Experiment (`python moe.py`)

A `d=64`, `E=8`, `top_k=2` MoE preserves the token shape and routes every token to exactly two
distinct experts:

```
shape  (4, 16, 64) --MoE--> (4, 16, 64)
```

Against one dense SwiGLU expert, the parameter/FLOP split is visible:

```
                  params  FFNs/token
dense SwiGLU      32,640           1
MoE E=8 k=2      261,632           2     # params x8.0, FLOPs x2
```

The router is negligible next to the experts, so total parameters are almost exactly `E×` the
dense FFN. But each token evaluates only `k` experts, so FFN compute is `k×`, independent of `E`.
That is the whole MoE bargain: more capacity in memory, sparse activation per token.

The script then trains the same toy regression task twice, once with no auxiliary loss and once
with `0.01 · L_aux`. Inputs share a common offset, which makes collapse easy to trigger:

```
expert usage after 300 steps (uniform = 12.5%)
                expert      0      1      2      3      4      5      6      7
           no aux loss   0.0%  50.0%   0.0%   0.0%  50.0%   0.0%   0.0%   0.0%
        aux loss x0.01  11.3%  13.3%  14.9%  11.7%   9.6%  17.1%  11.6%  10.5%
```

Without the auxiliary loss, six experts receive no traffic. With it, usage is not perfectly flat,
but every expert stays alive. The point of the experiment is not quality; it isolates the routing
mechanism and the load-balance pressure that keeps the extra parameters trainable.
