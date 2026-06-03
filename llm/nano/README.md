# llm/nano — 组装一个 decoder-only mini-Llama

> **📌 一手出处 / official reference**
> - 论文 *LLaMA: Open and Efficient Foundation Language Models* — Touvron et al., 2023 · [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
> - 架构源头 *Attention Is All You Need* — Vaswani et al., 2017 · [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
> - 教学级参考实现 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)、[meta-llama/llama](https://github.com/meta-llama/llama)

> 学习笔记 + 复现 spec。目标：复现完 [`nano.py`](nano.py) 后，能讲清
> **「rope + rmsnorm + swiglu + gqa 怎么拼成一个能 train、能生成的现代 block」**，
> 并产出本主线的「现代积木」参考实现（类比旧 repo 的 `common.py`）。

## 一句话 mental model

把前四个组件按 **pre-norm 残差**拼成一个 modern block，堆 N 层，前面接 token embedding、
后面接一个 LM head，就是一个 decoder-only 语言模型。用 **next-token（teacher forcing）**
在一小段文本上训练，证明它能学会续写。

## modern block 长这样（== 四件套各就各位 ==）

```
x = x + GQA( RMSNorm(x),  rope=on,  causal=True )   # rope 在 GQA 内部对 Q/K 施加
x = x + SwiGLU( RMSNorm(x) )                         # 门控 FFN
```
对比旧 `common.py` 的 classic block：`LayerNorm→MHA`、`LayerNorm→GELU-MLP`、位置靠外部加。
**四处零件全换了**，残差结构没变。

## 🎯 这个组件要盯死的：next-token 训练到底在干嘛

```
输入  tokens:  [BOS]  the   cat   sat
标签(右移一位): the    cat   sat   [EOS]
                每个位置预测「下一个 token」，所有位置一次并行算（teacher forcing）
loss = CrossEntropy(logits[:, :-1], labels[:, 1:])   # 经典的 shift-by-one
```

反直觉点：训练时**所有位置同时**算 loss（causal mask 保证第 t 位看不到未来），一次 forward
就拿到整条序列每一步的监督信号——这和推理时**一个一个吐**（见 [`../kvcache`](../kvcache)）形成鲜明对照。

**weight tying**：token embedding 矩阵和 LM head 的输出矩阵**共享同一份权重**（省参数、且有道理：
「把向量映射回词表」和「把词表映射成向量」本就是一对互逆操作）。

## 你要复现的目标（shape 契约 + TODO）

文件：`nano.py`（shape walk）、`train_sanity.py`（训练验证）。

```python
class ModernBlock(nn.Module):     # RMSNorm + GQA(rope, causal) + RMSNorm + SwiGLU，pre-norm 残差
class NanoLLM(nn.Module):
    # embed: nn.Embedding(vocab, d)
    # blocks: N × ModernBlock
    # norm:  RMSNorm(d)
    # lm_head: Linear(d, vocab, bias=False)，weight 与 embed 共享（tying）
    def forward(self, ids):       # [B, L] -> logits [B, L, vocab]
```
shape walk：`ids [B,L] → embed [B,L,d] → N blocks → norm → logits [B,L,vocab]`。

## ⭐ 必做的 sanity（train_sanity.py）

在一小段文本（char-level 或 tiny vocab）上 next-token 训练几百步：
- 断言 loss 显著下降（如后 20 步均值 < 前 20 步 × 0.9）；
- 用贪心解码打印一段生成样本，肉眼可见它学到了语料的模式（能背出/续写）。

## ✅ 盲讲检查点

1. modern block 的两条残差分别是什么？四件套各在哪一步？
2. next-token 训练怎么造标签？为什么能「所有位置并行算 loss」？（causal mask + teacher forcing）
3. weight tying 把哪两个矩阵绑一起？为什么合理？
4. 训练（并行）和推理（自回归逐 token）的根本不对称在哪？
5. causal mask 在这里是必须的吗？去掉会怎样？（会偷看未来，训练作弊）

> 推理 + 采样在 [`../kvcache`](../kvcache)；想加稀疏 FFN 去 [`../moe`](../moe)；这个 mini-Llama 也是
> [`../../posttrain`](../../posttrain) 整条后训练线的底座。
