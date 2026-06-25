# glassbox

Minimal from-scratch implementations of the components inside a vision-language
model (VLM), anchored on Qwen-VL.

Two tracks. `llm` is the language backbone — how a modern decoder-only LLM differs
from the original Transformer. `cv` is the vision tower — pixels to patch tokens.
They are independent bodies of work; the point of this repo is the seam where they
join: `cv/vl` splices vision tokens into the backbone and `cv/mrope` gives them
multimodal positions. That join is the VLM. Each directory builds one mechanism in
isolation.

## llm

How a current decoder-only LLM differs from the original Transformer.

| dir | mechanism | paper |
|---|---|---|
| [`rope`](llm/rope) | rotary position embedding | [2104.09864](https://arxiv.org/abs/2104.09864) |
| [`rmsnorm`](llm/rmsnorm) | RMS normalization | [1910.07467](https://arxiv.org/abs/1910.07467) |
| [`swiglu`](llm/swiglu) | gated feed-forward network | [2002.05202](https://arxiv.org/abs/2002.05202) |
| [`gqa`](llm/gqa) | grouped-query attention | [2305.13245](https://arxiv.org/abs/2305.13245) |
| [`nano`](llm/nano) | the above assembled into a small Llama | [2302.13971](https://arxiv.org/abs/2302.13971) |
| [`kvcache`](llm/kvcache) | KV-cached decoding and sampling | [1911.02150](https://arxiv.org/abs/1911.02150) |
| [`moe`](llm/moe) | sparse mixture-of-experts | [1701.06538](https://arxiv.org/abs/1701.06538) |
| [`bpe`](llm/bpe) | byte-pair encoding tokenizer | [1508.07909](https://arxiv.org/abs/1508.07909) |
| [`mla`](llm/mla) | multi-head latent attention | [2405.04434](https://arxiv.org/abs/2405.04434) |
| [`localattn`](llm/localattn) | sliding-window + local/global attention | [2310.06825](https://arxiv.org/abs/2310.06825) |
| [`specdec`](llm/specdec) | speculative decoding | [2211.17192](https://arxiv.org/abs/2211.17192) |

## cv

From pixels to tokens, then the 2024–25 question: how to spend the visual-token budget.

| dir | mechanism | paper |
|---|---|---|
| [`vit`](cv/vit) | image as a sequence of patches | [2010.11929](https://arxiv.org/abs/2010.11929) |
| [`mae`](cv/mae) | masked-autoencoder pretraining | [2111.06377](https://arxiv.org/abs/2111.06377) |
| [`videomae`](cv/videomae) | masked autoencoding for video | [2203.12602](https://arxiv.org/abs/2203.12602) |
| [`clip`](cv/clip) | contrastive image-text pretraining | [2103.00020](https://arxiv.org/abs/2103.00020) |
| [`vl`](cv/vl) | vision tokens spliced into an LLM | [2304.08485](https://arxiv.org/abs/2304.08485) |
| [`tokbudget`](cv/tokbudget) | the visual-token cost ledger | [2403.06764](https://arxiv.org/abs/2403.06764) |
| [`framesample`](cv/framesample) | temporal budget before the encoder | [2502.21271](https://arxiv.org/abs/2502.21271) |
| [`tokmerge`](cv/tokmerge) | spatial token merging | [2404.16821](https://arxiv.org/abs/2404.16821) |
| [`mrope`](cv/mrope) | multimodal rotary position ids | [2409.12191](https://arxiv.org/abs/2409.12191) |
| [`siglip`](cv/siglip) | sigmoid contrastive loss | [2303.15343](https://arxiv.org/abs/2303.15343) |

## Layout

Each directory documents one mechanism (what it is, how it differs from the
baseline it replaces, its tensor-shape contract) as the spec for a single-file
PyTorch implementation. No pretrained weights, no large datasets. `docs/` holds a
generated mastery map of these components — run
`.venv/bin/python docs/build.py` and open `docs/index.html`.

---
Part of the glass\* line of from-scratch reproductions:
`glassbox` (a VLM's internals) · `glassalign` (post-training) · `glassloop` (the agent loop).
