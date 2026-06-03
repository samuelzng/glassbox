# llm/kvcache — KV 缓存 + 采样（推理时的事）

> **📌 一手出处 / official reference**
> - KV-cache 思想 *Fast Transformer Decoding: One Write-Head is All You Need* — Shazeer, 2019 · [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
> - 采样 *The Curious Case of Neural Text Degeneration*（top-p / nucleus）— Holtzman et al., ICLR 2020 · [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)；*Hierarchical Neural Story Generation*（top-k）— Fan et al., ACL 2018 · [arXiv:1805.04833](https://arxiv.org/abs/1805.04833)
> - 参考实现 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) 的 `generate`

> 学习笔记 + 复现 spec。目标：复现完 [`kvcache.py`](kvcache.py) 后，能讲清
> **「训练并行 vs 推理自回归的不对称，以及为什么缓存 K/V 能把每步从 O(L²) 降到 O(L)」**。

## 一句话 mental model

训练时整条序列一次算完。**生成**时只能一个一个吐：吐第 `t` 个 token 时，它要 attend 前面
`t−1` 个位置的 K、V——而这些**上一步已经算过了**。把它们**缓存**起来，每新一步只算「当前这一个
token 的 q/k/v」，不再重算历史 → 每步从 `O(L²)` 降到 `O(L)`。

## 🎯 盯死的反直觉机制 1：缓存的是 K 和 V，不是 Q

```
第 t 步:  只为新 token 算  q_t, k_t, v_t
          k_t, v_t  ──追加进 cache──►  [k_1..k_t], [v_1..v_t]
          注意力:   q_t  对  缓存里的所有 K/V  做一次 attention  →  下一个 token 的 logits
```
- **Q 不缓存**：每步只用当前的 `q_t`，过去的 q 再也用不到。
- **K/V 缓存**：过去每个位置的 k/v 会被未来每一步反复用到，所以存下来。
- 这正是 [`../gqa`](../gqa) 省显存的兑现点：cache 大小 ∝ KV 头数。

## 🎯 盯死的反直觉机制 2：模型只吐分布，"聪明"一半在 sampler

模型每步输出的是 vocab 上的一个**概率分布**，不是一个确定的词。怎么从分布里挑下一个词，
极大影响生成质量/多样性：

| 采样 | 做法 | 效果 |
|---|---|---|
| greedy | 每步取 argmax | 确定、易复读 |
| temperature `T` | logits / T 后再 softmax | T↑ 更随机，T↓ 更尖锐 |
| top-k | 只在概率最高的 k 个里采 | 砍掉长尾噪声 |
| top-p (nucleus) | 只在累积概率达 p 的最小集合里采 | 自适应候选数 |

## 你要复现的目标（shape 契约 + TODO）

文件：`kvcache.py`（基于 [`../nano`](../nano) 的模型）。

```python
# 每层维护 (k_cache, v_cache)，形状随生成增长 [B, H_kv, t, d_k]
@torch.no_grad()
def generate(model, prompt_ids, max_new, sampler):
    # 1) prefill: 一次喂入 prompt，填满 cache
    # 2) decode: 循环——只喂上一个 token，读 cache，算 logits，采样，追加进 cache
    ...
def sample(logits, temperature=1.0, top_k=None, top_p=None): ...
```

## ⭐ 必做的 sanity

> **等价性测试**（仿 `common.py` 那种具体不变量）：同一 prompt + **贪心**解码，
> 「带 KV-cache 增量生成」的输出，必须和「每步全序列重算」的输出**逐 token 完全相同**。
> 这条 `assert` 证明 cache 是纯加速、不改变语义。

再附一个采样 demo：固定随机种子，展示 temperature / top-k / top-p 怎么改变生成的多样性。

## ✅ 盲讲检查点

1. 缓存的是 Q、K、V 里的哪些？为什么 Q 不用缓存？
2. 没有 cache 时，生成第 t 个 token 是 `O(?)`；有 cache 后是 `O(?)`？为什么？
3. prefill 和 decode 两个阶段各干嘛？
4. 为什么「贪心 + cache」必须等于「贪心 + 每步重算」？
5. temperature / top-k / top-p 各自改变了什么？greedy 为什么容易复读？

> 这把 [`../gqa`](../gqa) 砍 KV 头的显存收益真正兑现了。回 [`../`](..) 看 moe / bpe 进阶。
