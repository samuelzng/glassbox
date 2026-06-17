# llm/kvcache — KV 缓存 + 采样（推理时的事）

> **📌 一手出处 / official reference**
> - KV-cache 思想 *Fast Transformer Decoding: One Write-Head is All You Need* — Shazeer, 2019 · [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
> - 采样 *The Curious Case of Neural Text Degeneration*（top-p / nucleus）— Holtzman et al., ICLR 2020 · [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)；*Hierarchical Neural Story Generation*（top-k）— Fan et al., ACL 2018 · [arXiv:1805.04833](https://arxiv.org/abs/1805.04833)
> - 参考实现 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) 的 `generate`

> 学习笔记 + 复现 spec。目标：复现完 [`kvcache.py`](kvcache.py) 后，能讲清
> **「训练并行 vs 推理自回归的不对称，以及为什么缓存 K/V 能把每步从 O(L²) 降到 O(L)」**。
> 复习时：先做文末盲讲检查点，全答上来就直接跑代码，不必重读正文。

## 一句话 mental model

训练时整条序列一次算完。**生成**时只能一个一个吐：吐第 `t` 个 token 时，它要 attend 前面
`t−1` 个位置的 K、V——而这些**上一步已经算过了**。把它们**缓存**起来，每新一步只算「当前这一个
token 的 q/k/v」，不再重算历史 → 每步从 `O(L²)` 降到 `O(L)`。

## 🎯 反直觉 1 ｜ 先猜：K/V 要缓存，为什么 Q 不用？（想 30 秒再往下）

**答案：注意力的不对称——过去位置的 k/v 会被未来每一步反复查询，过去的 q 用完即弃。**

```
第 t 步:  只为新 token 算  q_t, k_t, v_t
          k_t, v_t  ──追加进 cache──►  [k_1..k_t], [v_1..v_t]
          注意力:   q_t  对  缓存里的所有 K/V  做一次 attention  →  下一个 token 的 logits
```
- **Q 不缓存**：每步只用当前的 `q_t`，过去的 q 再也用不到。
- **K/V 缓存**：过去每个位置的 k/v 会被未来每一步反复用到，所以存下来。
- 这正是 [`../gqa`](../gqa) 省显存的兑现点：cache 大小 ∝ KV 头数。

## 🎯 反直觉 2 ｜ 先猜：同一个模型，换个采样策略为什么质量天差地别？

**因为模型每步只吐 vocab 上的一个概率分布，「聪明」的另一半在 sampler 怎么挑。**

| 采样 | 做法 | 效果 |
|---|---|---|
| greedy | 每步取 argmax | 确定、易复读 |
| temperature `T` | logits / T 后再 softmax | T↑ 更随机，T↓ 更尖锐 |
| top-k | 只在概率最高的 k 个里采 | 砍掉长尾噪声 |
| top-p (nucleus) | 只在累积概率达 p 的最小集合里采 | 自适应候选数 |

实现要点（照这个写不踩坑）：

- temperature：`logits / T` 再 softmax；T→0 的极限就是 argmax。
- top-k：`torch.topk` 拿到第 k 大的值，比它小的 logits 置 `-inf`。
- top-p：logits 降序排序 → softmax → `cumsum` → 砍掉「累积概率已超 p」的尾巴，
  **砍之前把超标 mask 右移一位、永远保留第 1 名**（否则 p 比最大概率还小时会一个候选都不剩）
  → 按排序索引散射回原顺序置 `-inf`。
- greedy 必须走 `argmax` 而不是 `multinomial`，等价性测试要求它完全确定。

## 管线约定（设计决策，照抄即可——学习价值在机制，不在猜接口）

[`../ref/nano.py`](../ref/nano.py) 的 `forward(ids)` 不认识 cache，先做一次小重构。约定一条
**唯一的 forward 路径**，训练 / prefill / 逐 token decode 全是它的特例：

```python
class KVCache:
    # 每层一份，沿 t 维增长:  k[i], v[i]: [B, H_kv, t, d_k]
    def update(self, i, k_new, v_new): ...   # 第 i 层追加，返回追加后的全量 (k, v)
    def truncate(self, t): ...               # 所有层切回前 t 位（specdec 拒稿回滚用）
    @property
    def t(self): ...                         # 当前已缓存长度，空 cache 为 0

def NanoLLM.forward(ids, cache=None)         # ids [B, n_new] -> logits [B, n_new, vocab]
# 训练:    cache=None        —— 行为与重构前完全一致
# prefill: cache 空, n_new=L₀
# decode:  cache 长 t, n_new=1（specdec 的验证步则是 n_new=k+1）

@torch.no_grad()
def decode_step(model, ids_new, cache): return model(ids_new, cache)   # 薄封装
def generate(model, prompt_ids, max_new, sampler):
    # 1) prefill: decode_step(prompt)   2) 循环: decode_step(上一个 token) → 采样 → 追加
def sample(logits, temperature=1.0, top_k=None, top_p=None): ...
```

带 cache 时 attention 内部只有三处不同（=== DIFF vs 训练 forward ===）：

1. **rope 按绝对位置取片**：`cos[t : t+n_new]` 而不是 `cos[:n_new]`——新 token 的绝对位置
   从 t 起跳，从 0 取会把它们当成序列开头。
2. **k/v 先进 cache 再用全量**：`k, v = cache.update(layer_idx, k, v)`。
3. **mask 变长方形** `[n_new, t+n_new]`：新 chunk 内部 causal、对 cache 全可见——

   ```python
   mask = torch.triu(torch.ones(n_new, t + n_new, dtype=torch.bool), diagonal=t + 1)
   ```

   自查：第 i 行是绝对位置 t+i，应能看见第 0..t+i 列。`n_new=1` 时 mask 全 False（看全部
   历史）；`t=0, n_new=L` 时退化为训练用的下三角——所以训练路径不需要单独的 mask 代码。

## 🪜 实现阶梯（每级跑通断言再上一级）

1. **`sample()` 单独写、单独测**：greedy / temperature / top-k / top-p。
   断言：`top_p=1.0`、`top_k=vocab` 不改变分布；T 很小时 sample 结果 == argmax。
2. **forward 接 cache，只支持 n_new=1**：prefill 暂时用「全序列重算取最后一位」凑合。
   断言（主断言的最小版）：贪心生成 20 token == 每步全序列重算的贪心，逐 token 相同。
3. **推广 n_new>1**：写代码前先在纸上画出 `t=3, n_new=2` 的 mask。prefill 改用
   `decode_step(整个 prompt)`。
   断言：`decode_step(prompt)` 的 logits == 训练式 `forward(prompt)` 的 logits（allclose）。
4. **`generate()` 串起来**，跑 ⭐ 的完整等价性测试。
5. **`truncate()` 回滚**。断言：「算到 t=10 → 回滚到 5 → 续算 5 个」 == 「一口气算到 10」。
6. **采样 demo**：固定种子，同一 prompt 对比 greedy / T=0.8 / top-k=5 / top-p=0.9。

第 3、5 级和 rope 取片正是 [`../specdec`](../specdec) 的接口契约（验证步一次喂 k+1 个
draft token、拒稿后回滚），现在写对免得返工。

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
6. `n_new > 1` 时 mask 长什么样？为什么 prefill 和逐 token decode 是同一个函数的两个特例？
7. 带 cache 时 rope 的 cos/sin 从哪个下标开始取？取错了会发生什么？

> 这把 [`../gqa`](../gqa) 砍 KV 头的显存收益真正兑现了。2025 支线由此出发：
> [`../mla`](../mla) 把 cache 再砍一个量级，[`../specdec`](../specdec) 直接复用这里的
> `decode_step`。moe / bpe 进阶仍在 [`../`](..)。
