# llm/moe — Mixture of Experts 稀疏 FFN

> **📌 一手出处 / official reference**
> - 论文 *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* — Shazeer et al., 2017 · [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
> - 现代实践 *Switch Transformers* — Fedus et al., 2021 · [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)；*Mixtral of Experts* — Jiang et al., 2024 · [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)

> 学习笔记 + 复现 spec。目标：复现完 [`moe.py`](moe.py) 后，能讲清
> **「为什么参数量能涨 N 倍，而每个 token 的算力几乎不变」**，以及 load-balance loss 防的是什么。
> 复习时：先做文末盲讲检查点，全答上来就直接跑代码。

## 一句话 mental model

把一个 FFN 换成 **N 个并列的 expert FFN** + 一个 **router**。每个 token 进来，router 只挑
**top-k 个** expert（通常 k=1 或 2）去算，其余 expert 这个 token 完全不碰。输出 = 被选中
expert 输出的加权和。**容量（参数）和算力（FLOPs）从此解耦。**

## 🎯 先猜：8 个 expert、top-2，总参数是 dense FFN 的几倍？单 token FLOPs 呢？（想 30 秒）

**答案：参数 ≈ ×8，FLOPs 只 ≈ ×2——容量随 N 涨，计算只随 k 涨。**

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

## 🎯 先猜：只用主任务 loss 训练，router 会学成什么样？

**答案：塌缩——所有 token 涌向少数几个 expert（赢家通吃），其余 expert 饿死、白占参数。**
被选中的 expert 训得更好 → 下次得分更高 → 更常被选中，正反馈一旦启动就锁死。
所以必须加一个**辅助 loss** 惩罚「分配不均」，逼 token 大致均匀散到各 expert。

### 辅助 loss 的公式（Switch Transformer §2.2，照这个写）

对一个 batch 的全部 token、N 个 expert：

```
f_i   = 被路由到 expert i 的 token 占比     （硬计数，不可导）
P_i   = router 给 expert i 的 softmax 概率在全 batch 上的均值（可导）
L_aux = α · N · Σᵢ f_i · P_i                （α ≈ 0.01；总 loss = CE + L_aux）
```

两边都均匀时 `Σ fᵢPᵢ = N·(1/N)(1/N) = 1/N`，乘 N 后 L_aux = α（最小值附近）；全塌到一个
expert 时 ≈ α·N。梯度全部从可导的 P 流回 router，硬计数 f 只负责给「谁超载」配权重。
（Switch 按 top-1 定义 f；k>1 时把「被路由到」理解为进入该 token 的 top-k，f 除以 k·T 归一。）

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

路由的写法——**先写慢的、对的**，教学版循环 expert 即可，不做 dispatch 优化：

```python
gate = softmax(router(x), dim=-1)            # [B,L,N]
w, idx = gate.topk(k, dim=-1)                # 各 [B,L,k]
w = w / w.sum(-1, keepdim=True)              # top-k 内重归一化（Mixtral 做法）
out = zeros_like(x)
for e in range(N):                           # 循环 expert，不循环 token
    sel = (idx == e)                         # [B,L,k] 哪些 token 的第几名选了 e
    tok = sel.any(-1)                        # [B,L] 布尔掩码
    out[tok] += w[sel][:, None] * experts[e](x[tok])   # 该 expert 一次算完它的 token
```

aux loss 在同一个 forward 里顺手算（f 从 `idx` 计数，P 从 `gate` 取均值）。

## 🪜 实现阶梯（每级跑通断言再上一级）

1. **router + top-k 选择**（先不接 expert）：断言每个 token 恰好选中 k 个、权重和为 1。
2. **接上 N 个 SwiGLU expert**：断言输出 shape == 输入 shape；N=1, k=1 时输出 ==
   单个 SwiGLU 直接算（MoE 退化为 dense 的等价性）。
3. **aux loss**：手算一个 2 expert、4 token 的小例子核对公式，再断言均匀路由时 ≈ α。
4. **塌缩实验**（⭐ 的直方图）：同任务同初始化，加 / 不加 aux loss 各训几十步，对比使用率。
   跑之前先预测：不加的那组，8 个 expert 大概几个存活？

## ⭐ 必做的 sanity

`moe.py` / `train_sanity.py`：
- 断言每个 token 恰好路由到 `k` 个 expert（top-k 选择正确）；
- 打印 **expert 使用率直方图**：训练几十步后，对比「加 vs 不加 load-balance loss」——
  加了的分布明显更均匀（证明辅助 loss 防住了塌缩）。

## ✅ 盲讲检查点

1. 为什么总参数随 N 涨、但单 token FLOPs 只随 k 涨？容量和算力怎么解耦的？
2. router 输出什么？top-k 是在挑什么？
3. 不加 load-balance loss 会发生什么？为什么是正反馈锁死？
4. aux loss 公式里 f 和 P 各是什么？为什么梯度只能从 P 走？
5. Mixtral「8x7B 激活 2 个」是什么意思？为什么算力≈13B 而非 47B？
6. expert 内部通常用什么结构？

> 进阶组件，做完可回 [`../`](..) 或转 [`../bpe`](../bpe)。
