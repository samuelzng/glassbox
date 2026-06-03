# glassbox

Minimal from-scratch implementations of the mechanisms behind modern AI: LLM architecture, RL post-training, and vision. Each component covers one mechanism in isolation.

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

## posttrain

Aligning a base model, from RLHF through the critic-free methods behind recent reasoning models.

| dir | mechanism | paper |
|---|---|---|
| [`sft`](posttrain/sft) | supervised fine-tuning | [2203.02155](https://arxiv.org/abs/2203.02155) |
| [`reward-model`](posttrain/reward-model) | scalar reward from preference pairs | [2009.01325](https://arxiv.org/abs/2009.01325) |
| [`ppo-rlhf`](posttrain/ppo-rlhf) | the RLHF loop with PPO | [1707.06347](https://arxiv.org/abs/1707.06347) |
| [`dpo`](posttrain/dpo) | preference optimization without a reward model | [2305.18290](https://arxiv.org/abs/2305.18290) |
| [`grpo`](posttrain/grpo) | group-relative advantages, no critic | [2402.03300](https://arxiv.org/abs/2402.03300) |

## cv

| dir | mechanism | paper |
|---|---|---|
| [`vit`](cv/vit) | image as a sequence of patches | [2010.11929](https://arxiv.org/abs/2010.11929) |
| [`mae`](cv/mae) | masked-autoencoder pretraining | [2111.06377](https://arxiv.org/abs/2111.06377) |
| [`videomae`](cv/videomae) | masked autoencoding for video | [2203.12602](https://arxiv.org/abs/2203.12602) |
| [`clip`](cv/clip) | contrastive image-text pretraining | [2103.00020](https://arxiv.org/abs/2103.00020) |
| [`vl`](cv/vl) | vision tokens spliced into an LLM | [2304.08485](https://arxiv.org/abs/2304.08485) |

## Layout

Each directory documents one mechanism (what it is, how it differs from the baseline it replaces, its tensor-shape contract) as the spec for a single-file PyTorch implementation. No pretrained weights, no large datasets.
