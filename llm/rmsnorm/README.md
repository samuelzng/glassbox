# llm/rmsnorm — RMSNorm

RMSNorm is LayerNorm with the centering removed: no mean subtraction, no bias. It keeps
only the scaling — divide each token vector by its root-mean-square. One statistic instead
of two, one parameter group instead of two, and deeper Transformers train more stably.
Llama / T5 / Qwen all use it.

> **References** — Zhang & Sennrich, *Root Mean Square Layer Normalization*, NeurIPS 2019
> ([arXiv:1910.07467](https://arxiv.org/abs/1910.07467)) · reference impl: `RMSNorm` in
> [meta-llama/llama](https://github.com/meta-llama/llama).

```
LayerNorm(x) = (x − mean(x)) / sqrt(var(x) + ε) · γ + β     # center, then scale, + bias
RMSNorm(x)   =  x            / sqrt(mean(x²) + ε) · g        # scale only, no bias
```

## Experiment (`python rmsnorm.py`)

Same input `x` of shape `[4, 8]` through both norms:

```
            row mean   row rms
LayerNorm      0.000     1.000
RMSNorm        0.332     1.000
```

Both pin every row to **rms 1** — the scaling they share. Only LayerNorm also forces
**mean 0** (centering); RMSNorm leaves the mean alone, so its outputs stay off-center.

Then shift every input by a constant, `x → x + 3`:

```
shift +3      max|Δ|
LayerNorm      0.000
RMSNorm        2.368
```

LayerNorm is **shift-invariant** — subtracting the mean cancels the constant. RMSNorm is
not: the constant enters `mean(x²)` and changes every output. That single asymmetry is the
whole difference between the two.

> Next: [`../swiglu`](../swiglu) / [`../gqa`](../gqa).
