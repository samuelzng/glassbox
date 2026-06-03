# llm/swiglu — SwiGLU 门控 FFN

> **📌 一手出处 / official reference**
> - 论文 *GLU Variants Improve Transformer* — Noam Shazeer, 2020 · [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
> - 参考实现 Llama 官方 [meta-llama/llama](https://github.com/meta-llama/llama) 里的 `FeedForward`

> 学习笔记 + 复现 spec。目标：复现完 [`swiglu.py`](swiglu.py) 后，能讲清
> **「为什么 FFN 从 2 个矩阵变成 3 个、hidden 维要 ×⅔」**。

## 一句话 mental model

原版 FFN 是「升维 → 激活 → 降维」两个矩阵。SwiGLU 多塞一条**门控分支**：一条做内容、
一条做「闸门」（SiLU 激活），两者**逐元素相乘**后再降维。门让网络自己决定每个隐藏单元放多少进去。

## 🎯 盯死的反直觉机制：加一条门，比单纯加宽更划算

```
classic FFN:  out = W2( GELU( W1 x ) )                 # 2 个矩阵
SwiGLU   :    out = W_down( SiLU(W_gate x) ⊙ (W_up x) ) # 3 个矩阵，⊙ = 逐元素乘
                          └─ 门（闸门）   └─ 内容
```

**为什么 hidden 维要 ×⅔？** 为了和原版**参数量持平**。原版 `d→4d→d` 用 `2·(d·4d)=8d²`。
SwiGLU 有 3 个矩阵，若 hidden 仍取 `4d` 会多 50% 参数；所以把 hidden 缩到 `⅔·4d ≈ 8/3·d`，
`3·(d·8/3 d)=8d²`，刚好相等。即：**用同样的参数预算，把「加宽」换成「加门」。**

## === DIFF vs classic block ===

| | classic FFN (GELU-MLP) | SwiGLU |
|---|---|---|
| 矩阵数 | 2（W1, W2） | 3（W_gate, W_up, W_down） |
| 非线性 | 单个 GELU | SiLU 作用在**门**分支，再与内容**逐元素相乘** |
| hidden 维 | `4d` | `≈ ⅔ · 4d`（保参数量） |

## 你要复现的目标（shape 契约 + TODO）

文件：`swiglu.py`（shape walk + 参数量对比打印）。

```python
class SwiGLU(nn.Module):
    def __init__(self, d, hidden=None):
        hidden = hidden or int(2/3 * 4 * d)     # ⅔ 规则
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up   = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)
    def forward(self, x):                       # [B,L,d] -> [B,L,d]
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```
集成：替换 classic block 里的 `FeedForward`。

## ⭐ 必做的 sanity

`swiglu.py` 打印两件事：① shape walk `[B,L,d]→[B,L,hidden]→[B,L,d]`；
② **参数量对比**：断言 SwiGLU(d, hidden=⅔·4d) 的参数量 ≈ classic FFN(d, 4d)，验证 ⅔ 规则确实保了预算。

## ✅ 盲讲检查点

1. SwiGLU 的三个矩阵各干嘛？哪条是「门」？
2. 门和内容怎么结合？（逐元素相乘）SiLU 加在哪条分支？
3. 为什么 hidden 维要 ×⅔？（3 个矩阵下保持参数量）
4. 一句话：SwiGLU 是把参数预算从「加宽」换成了「加门」——对/错，为什么？

> 下一站 [`../gqa`](../gqa)，集齐后进 [`../nano`](../nano)。
