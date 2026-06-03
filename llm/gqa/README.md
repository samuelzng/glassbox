# llm/gqa — Grouped-Query Attention

> **📌 一手出处 / official reference**
> - 论文 *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* — Ainslie et al., EMNLP 2023 · [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
> - 前身 MQA *Fast Transformer Decoding: One Write-Head is All You Need* — Shazeer, 2019 · [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)

> 学习笔记 + 复现 spec。目标：复现完 [`gqa.py`](gqa.py) 后，能讲清
> **「为什么 K/V 头比 Q 头少——而且动机在推理显存，不在训练算力」**。

## 一句话 mental model

标准 MHA：Q、K、V 各有 `H` 个头。GQA：Q 仍 `H` 个头，但 **K/V 只有 `H_kv` 个头**
（`H_kv < H`），每个 K/V 头被 `H/H_kv` 个 Q 头**共享**。MQA 是极端情形 `H_kv = 1`。

## 🎯 盯死的反直觉机制：砍 K/V 头是为了推理时的 KV-cache 显存

直觉会以为「砍头是为省训练算力」。**不是。** 训练时 attention 仍是 `O(L²)`，砍 K/V 头省的算力很有限。
真正的痛点在**推理**：自回归生成要把过去所有位置的 K、V 缓存在显存里（见 [`../kvcache`](../kvcache)），

```
KV-cache 大小  ∝  层数 × 序列长 × KV头数 × d_k
```

KV 头数从 `H` 砍到 `H_kv`，**cache 直接小 `H/H_kv` 倍** → 同样显存能塞更长上下文 / 更大 batch。
而质量几乎不掉（论文实测 GQA 介于 MHA 和 MQA 之间，接近 MHA）。这是**纯推理工程驱动**的设计。

```
MHA  : H 个 Q 头 ↔ H 个 KV 头      （质量最好，cache 最大）
GQA  : H 个 Q 头 ↔ H_kv 个 KV 头   （折中，主流选择）
MQA  : H 个 Q 头 ↔ 1 个 KV 头       （cache 最小，质量略降）
```

## === DIFF vs classic block ===

| | MHA | GQA |
|---|---|---|
| wq 输出 | `H · d_k` | `H · d_k` |
| wk / wv 输出 | `H · d_k` | **`H_kv · d_k`**（更小） |
| 算 attention 前 | 直接用 | **把 K/V 沿头维 repeat `H/H_kv` 次**对齐到 H |

## 你要复现的目标（shape 契约 + TODO）

文件：`gqa.py`（shape walk + cache 大小对比）。

```python
class GQA(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads):
        assert n_heads % n_kv_heads == 0
        self.wq = nn.Linear(d, n_heads    * d_k)
        self.wk = nn.Linear(d, n_kv_heads * d_k)   # 更小
        self.wv = nn.Linear(d, n_kv_heads * d_k)   # 更小
    def forward(self, x, causal):
        # q: [B, H, L, d_k];  k,v: [B, H_kv, L, d_k]
        # k,v = repeat_kv(k, H//H_kv), repeat_kv(v, ...)  -> [B, H, L, d_k]
        # 之后和标准注意力一样
```
`repeat_kv`：把 K/V 在头维上每个头复制 `H/H_kv` 份。

## ⭐ 必做的 sanity

`gqa.py` 打印：① shape walk，显式标出 `k,v` 在 `repeat_kv` 前是 `H_kv` 头、之后是 `H` 头；
② **KV-cache 大小对比**：同 `(层数, L, d_k)` 下，断言 GQA/MQA 的 cache 比 MHA 小 `H/H_kv` / `H` 倍。

## ✅ 盲讲检查点

1. GQA 里 Q、K、V 各几个头？谁共享谁？
2. 砍 K/V 头主要省的是**训练算力**还是**推理显存**？为什么？
3. KV-cache 大小正比于哪几个量？砍 KV 头怎么影响它？
4. MHA / GQA / MQA 三者的连续谱，各自的 trade-off？
5. `repeat_kv` 在哪一步、把什么对齐到什么？

> 集齐 rope + rmsnorm + swiglu + gqa，进 [`../nano`](../nano) 组装；推理收益在 [`../kvcache`](../kvcache) 兑现。
