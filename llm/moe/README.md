# llm/moe — Mixture of Experts 稀疏 FFN

> **📌 一手出处 / official reference**
> - 论文 *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* — Shazeer et al., 2017 · [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
> - 现代实践 *Switch Transformers* — Fedus et al., 2021 · [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)；*Mixtral of Experts* — Jiang et al., 2024 · [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)

> 学习笔记 + 复现 spec。目标：复现完 [`moe.py`](moe.py) 后，能讲清
> **「为什么参数量能涨 N 倍，而每个 token 的算力几乎不变」**，以及 load-balance loss 防的是什么。

## 一句话 mental model

把一个 FFN 换成 **N 个并列的 expert FFN** + 一个 **router**。每个 token 进来，router 只挑
**top-k 个** expert（通常 k=1 或 2）去算，其余 expert 这个 token 完全不碰。输出 = 被选中
expert 输出的加权和。**容量（参数）和算力（FLOPs）从此解耦。**

## 🎯 盯死的反直觉机制：参数↑ 但单 token FLOPs≈不变

```
              ┌─ expert 1 (SwiGLU) ─┐
 token ─ router(挑 top-k) ─┼─ expert 2 ──────────┼─ 加权和 ─► out
              ├─ ...                │
              └─ expert N ──────────┘
   N 个 expert 的参数全在显存里 (容量↑N)，
   但每个 token 只过 k 个 (算力≈k 个 FFN，与 N 无关)
```

直觉里「更多参数 = 更多计算」，MoE 打破了它：**总参数随 N 线性涨，单 token 计算只随 k 涨**。
这就是 Mixtral-8x7B「8 个专家、每 token 只激活 2 个」能用接近 13B 的算力跑出 ~47B 容量的原因。

## 🎯 反直觉机制 2：必须有 load-balance 辅助 loss

只用主任务 loss 训练，router 会**塌缩**：所有 token 都涌向少数几个 expert（赢家通吃），
其余 expert 饿死、白占参数。所以加一个**辅助 loss** 惩罚「分配不均」，逼 token 大致均匀散到各 expert。

## === DIFF vs classic / swiglu FFN ===

| | dense FFN（如 SwiGLU） | MoE FFN |
|---|---|---|
| FFN 个数 | 1，所有 token 共用 | N 个 expert |
| 每 token 过几个 | 1（全部） | top-k 个 |
| 总参数 | `P` | `≈ N·P`（router 极小） |
| 单 token FLOPs | `~P` | `~k·P`（与 N 无关） |
| 额外 loss | 无 | **load-balance 辅助 loss** |

## 你要复现的目标（shape 契约 + TODO）

文件：`moe.py`（shape walk + expert 使用率直方图）。

```python
class MoE(nn.Module):
    def __init__(self, d, n_experts, top_k):
        self.router  = nn.Linear(d, n_experts)         # 打分
        self.experts = nn.ModuleList(SwiGLU(d) ...)     # N 个
    def forward(self, x):                               # [B,L,d] -> [B,L,d]
        # gate = softmax(router(x));  取 top_k 的 index + 权重
        # 每个 token 只送进它的 top_k experts，输出按 gate 权重加权和
        # 同时返回 load-balance aux loss
```

## ⭐ 必做的 sanity

`moe.py` / `train_sanity.py`：
- 断言每个 token 恰好路由到 `k` 个 expert（top-k 选择正确）；
- 打印 **expert 使用率直方图**：训练几十步后，对比「加 vs 不加 load-balance loss」——
  加了的分布明显更均匀（证明辅助 loss 防住了塌缩）。

## ✅ 盲讲检查点

1. 为什么总参数随 N 涨、但单 token FLOPs 只随 k 涨？容量和算力怎么解耦的？
2. router 输出什么？top-k 是在挑什么？
3. 不加 load-balance loss 会发生什么？（router 塌缩 / 赢家通吃）辅助 loss 防的就是它。
4. Mixtral「8x7B 激活 2 个」是什么意思？为什么算力≈13B 而非 47B？
5. expert 内部用什么结构？（通常就是 [`../swiglu`](../swiglu)）

> 进阶组件，做完可回 [`../`](..) 或转 [`../bpe`](../bpe)。
