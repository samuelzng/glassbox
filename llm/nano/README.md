# llm/nano — a decoder-only mini-Llama

`nano` is where the four mechanisms stop being isolated parts and become a model. The modern
decoder block is two **pre-norm residuals** — `x = x + GQA(RMSNorm(x))` then
`x = x + SwiGLU(RMSNorm(x))`, with RoPE rotating Q/K inside the attention. Stack `N` of these
between a token embedding and a tied LM head and you have a decoder-only language model: the
Llama architecture, minus scale. The counter-intuitive part is how little the assembly adds —
every piece was already written and unit-tested on its own, so `nano` is mostly wiring, plus the
two things that only exist at the model level: the residual stream and next-token training.
Llama / Qwen / Mistral are all this block, wider and deeper.

> **References** — Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, 2023
> ([arXiv:2302.13971](https://arxiv.org/abs/2302.13971)) · architecture from Vaswani et al.,
> *Attention Is All You Need*, 2017 ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762)) ·
> reference impls: [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT),
> [meta-llama/llama](https://github.com/meta-llama/llama).

```
NanoLLM =
  token embedding
  → N × [ x = x + GQA( RMSNorm(x), RoPE, causal )   # causal self-attention — the decoder-only part
          x = x + SwiGLU( RMSNorm(x) ) ]            # two pre-norm residuals per block
  → final RMSNorm
  → tied LM head
  → next-token loss
```

Compared with the classic Transformer block the residual shape is unchanged; the swapped parts
are MHA→GQA, LayerNorm→RMSNorm, GELU-MLP→SwiGLU, and absolute-PE→RoPE. Three wiring facts carry
the rest. **Pre-norm**: each sub-layer normalizes a *copy* of the stream and adds the result back
to the un-normalized `x`, so the residual highway runs clean from embedding to head — which is
exactly why a final `RMSNorm` is needed before the LM head. **RoPE lives inside attention**:
`build_rope_cache` runs once at construction (sized by `d_k` and `max_len`), and the same
`cos`/`sin`, sliced to the current length, feed every layer — the rotation hits Q and K *after*
the head split, never the residual stream. **Decoder-only is the causal mask**: GQA runs with
`causal=True`, so position `t` attends only to `≤ t` — flip it to `False` and the same block is a
bidirectional encoder. That one constraint, plus next-token teacher forcing (a single forward
scores every position at once, `loss = CrossEntropy(logits[:, :-1], ids[:, 1:])`) and a
weight-tied LM head (`lm_head.weight = embed.weight`, one shared `[vocab, d]` Parameter), is what
makes this a generative decoder rather than an encoder.

Unlike the single-file reference, this `nano.py` **imports the real component files**: a two-line
path shim puts `llm/` on `sys.path`, then `from rope.rope import build_rope_cache`,
`from gqa.gqa import GQA`, and so on. The components stop being throwaway exercises and become the
engine — a fix in `rope.py` propagates here, and the same `NanoLLM` is later reused by the scaled
experiment instead of being rewritten.

## Experiment (`python nano.py`)

A 2-layer `d=64` model (92K params) trained 300 steps on a char-level description of its own
architecture:

```
vocab 26  params 91,840
shape  ids (1, 8) --NanoLLM--> logits (1, 8, 26)
step    0  loss 3.280
step   50  loss 0.352
step  150  loss 0.062
step  299  loss 0.051

greedy from 'rope rot':
rope rotates the queries and keys. rmsnorm rescales the stream. swiglu gates the feed forward. gqa shares the kv heads. stack th
```

The loss starts at `ln(26) ≈ 3.26` — the entropy of a uniform guess over 26 characters — and
falls to `~0.05` as the model memorizes the text, after which greedy decoding recites it back.
This is memorization on purpose, not generalization: on a few hundred characters the only claim
being proved is that the assembled pipeline trains and generates end-to-end. It is also the first
time the four hand-written components, composed through imports, run together as one model.
