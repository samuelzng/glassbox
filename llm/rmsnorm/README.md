# llm/rmsnorm — RMSNorm

> **📌 一手出处 / official reference**
> - 论文 *Root Mean Square Layer Normalization* — Zhang & Sennrich, NeurIPS 2019 · [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
> - 参考实现 Llama 官方 [meta-llama/llama](https://github.com/meta-llama/llama) 里的 `RMSNorm`

> 学习笔记 + 复现 spec。目标：复现完 [`rmsnorm.py`](rmsnorm.py) 后，能讲清
> **「LayerNorm 里减均值那一步其实可以扔掉」**。

## 一句话 mental model

LayerNorm 做两件事：①**减均值**（centering）②**除以标准差再缩放**（scaling）。
RMSNorm 只保留 ② 的精神——直接除以**均方根 RMS**，**不减均值、不要 bias**。更省，还更稳。

## 🎯 盯死的反直觉机制：centering 可省

```
LayerNorm(x) = (x − mean(x)) / sqrt(var(x) + ε) · γ + β     # 减均值、有 bias
RMSNorm(x)   =  x           / sqrt(mean(x²) + ε) · g         # 不减均值、无 bias
```

实验发现：让 LayerNorm 之所以 work 的主要是**re-scaling（控制每个 token 向量的模长）**，
而 **re-centering（减均值）几乎没贡献**。于是 RMSNorm 干脆砍掉它——少一次求均值、少一组 bias 参数，
深层 Transformer 反而训得更稳。Llama / T5 / Qwen 全用它。

## === DIFF vs classic block ===

| | LayerNorm | RMSNorm |
|---|---|---|
| 减均值 | ✅ | ❌ |
| 分母 | std（含均值信息） | RMS = `sqrt(mean(x²))` |
| 可学习参数 | gain γ + bias β | **只有 gain g** |
| 对「整体 +c 平移」 | **不变**（减均值消掉了） | **会变**（没减均值） ← 用这条做 sanity |

## 你要复现的目标（shape 契约 + TODO）

文件：`rmsnorm.py`（shape walk）。

```python
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        self.g = nn.Parameter(torch.ones(d))   # 只有 gain，无 bias
    def forward(self, x):                       # [B, L, d] -> [B, L, d]
        # 在最后一维上算 RMS，x / rms * g
        ...
```
集成：把 classic block 里两处 `nn.LayerNorm` 直接换成 `RMSNorm`，其余不动。

## ⭐ 必做的 sanity

> 给输入的**每个 token 整体加一个常数 c**（uniform shift）。
> - 过 LayerNorm：输出**不变**（减均值把 c 消掉了）。
> - 过 RMSNorm：输出**会变**（没减均值，c 进了 RMS）。
>
> 这条 `assert` 直接把两者的本质差异钉死。

## ✅ 盲讲检查点

1. LayerNorm 的两步分别是什么？RMSNorm 砍掉了哪一步？
2. RMSNorm 还剩哪些可学习参数？（只有 gain，无 bias）
3. 为什么砍掉 centering 还能 work？（起作用的主要是 re-scaling）
4. 在哪一维上做归一化？（最后的特征维 d）
5. 为什么对「整体平移」LN 不变而 RMSNorm 会变？

> 下一站 [`../swiglu`](../swiglu) / [`../gqa`](../gqa)。
