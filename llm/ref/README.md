# llm/ref — reference implementations

Working single-file references for the components still in spec form. The spec READMEs
([`../nano`](../nano), [`../moe`](../moe)) stay the source of truth — attempt your own
reproduction from the spec first, then diff against these. Each file runs standalone
(`python nano.py`) and asserts the mechanism's signature property.

| file | spec | the run proves |
|---|---|---|
| [`nano.py`](nano.py) | [`../nano`](../nano) | rope + rmsnorm + swiglu + gqa assemble into a mini-Llama that trains and generates |
| [`moe.py`](moe.py) | [`../moe`](../moe) | top-k routing decouples params (×E) from FLOPs (×k); without the aux loss the router collapses |

## nano.py

Inlines condensed copies of all four components so the whole assembly is visible in one
file: `ModernBlock` (two pre-norm residuals), `NanoLLM` (embed → N blocks → norm → tied
lm_head). Trains char-level for 300 steps on a description of its own architecture
(loss 3.28 → 0.05) and greedy-decodes it back:

```
greedy from 'rope rot':
rope rotates the queries and keys. rmsnorm rescales the stream. swiglu gates ...
```

## moe.py

8 SwiGLU experts behind a top-2 router: ×8.0 params, ×2 FLOPs per token. Then the
collapse experiment — same toy task, same init, with and without the load-balance loss:

```
expert usage after 300 steps (uniform = 12.5%)
                expert      0      1      2      3      4      5      6      7
           no aux loss   0.0%  50.0%   0.0%   0.0%  50.0%   0.0%   0.0%   0.0%
        aux loss x0.01  11.3%  13.3%  14.9%  11.7%   9.6%  17.1%  11.6%  10.5%
```

Winner-take-all is the default outcome, not a rare failure; the aux loss is what keeps
every expert alive.

> Still spec-only: [`../kvcache`](../kvcache) — its equivalence test (greedy with cache
> == greedy recomputing) builds directly on `nano.py`'s model — and the 2025 branch
> ([`../mla`](../mla), [`../localattn`](../localattn), [`../specdec`](../specdec)),
> all of which depend on kvcache landing first.
