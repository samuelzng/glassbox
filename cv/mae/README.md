# cv/mae — Masked Autoencoder

> **📌 一手出处 / official reference**
> - 论文 *Masked Autoencoders Are Scalable Vision Learners* — He et al., 2021 · [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
> - 官方实现 [facebookresearch/mae](https://github.com/facebookresearch/mae)

> 学习笔记 + 复现 spec。目标：复现完 [`mae.py`](mae.py) 后，能对着代码盲讲
> **非对称 encoder–decoder + masking 的数据流**。
> 复习时：先做文末盲讲检查点，全答上来就直接跑代码。

## 一句话 mental model

图切 patch → **随机扔掉 75%** → 一个**深** encoder *只看*剩下的 25% → 一个**浅** decoder
把扔掉的部分**画回来** → **只在被扔的 patch 上**算重建 MSE。预训练完，encoder 就是通用视觉特征
提取器（下游把 decoder 扔掉）。本质是 BERT 完形填空搬到图像，但有一个图像独有的转折。

## 🎯 先猜：那个又深又贵的 encoder，要不要处理被遮掉的 75% patch？（想 30 秒）

**答案：完全不碰——encoder 只吃 25% 可见 patch，mask token 要到浅 decoder 才出现。**
这就是「非对称 (asymmetric) encoder–decoder」，也是标题里 "scalable" 的来源。

```
        25% 可见 patch ──►  ENCODER (深,宽) ──► latent
                                                  │
   全部位置 = latent + 共享 mask token ──► DECODER (浅,窄) ──► 重建像素
```

- **encoder 从不看 mask token**：它的输入序列长度只有 `len_keep`（25%），不是全长。
- mask token 只在 **decoder** 里才插进来补满位置。和 BERT 不同（BERT 的 `[MASK]` 第一层就进 encoder）。

## 三个关键设计 & 为什么

1. **为什么遮 75% 还能重建？** 自然图像空间冗余极高，缺的能从周围推出来。语言信息密度高（BERT 只遮 15%），图像稀疏，遮 75% 才逼模型学**语义级**理解而非抄邻块像素。
2. ⭐ **为什么 encoder 不碰 mask token 省这么多算力？** attention 是 `O(L²)`。全看 64 个：`64²=4096`；只看 16 个：`16²=256`，**省 16×**。贵的深 encoder 只花在真信息上，补全交给便宜的浅 decoder。这就是标题里 "scalable" 的来源。
3. **为什么 shuffle + gather 而非布尔 mask？** 每样本保留位置不同→ragged 张量没法 batch。技巧：`noise→argsort=ids_shuffle`，`argsort(ids_shuffle)=ids_restore`（逆排列），gather 前 `len_keep` 个得规则形状；decoder 端用 `ids_restore` 一次 gather 还原原始顺序。全程向量化、无循环。
4. **norm_pix_loss**：target 用每个 patch 自己的均值/方差归一化，去掉亮度/对比度绝对值，重建更锐利。

## 数据流（`mae.py` 的 shape walk，toy：32×32, patch4 → 64 patch, 遮 75% 留 16）

| 步骤 | shape | 说明 |
|---|---|---|
| patchify | `[B,64,48]` | 64 patch，每个 48 维 |
| embed + 位置（**masking 前加**） | `[B,64,D]` | 可见 token 带着「我是第几块」进 encoder |
| random masking | `[B,16,D]` | shuffle+gather 留 16，序列变短 |
| encoder（只吃可见） | `[B,16,D]` | ⚠️ 16 不是 64 |
| 补 mask token + 反 shuffle | `[B,64,d_dec]` | 空位填同一个可学习向量，`ids_restore` 还原顺序 |
| + decoder 位置 → decoder → 预测像素 | `[B,64,48]` | |
| loss | 标量 | **只在被遮的 patch 上**算 MSE |

## 🪜 实现阶梯（每级跑通断言再上一级）

1. **shuffle/gather masking 单独写、单独测**——这是全组件最容易错的一步，先隔离它：
   `random_masking(x, ratio)` 返回 `(x_kept, mask, ids_restore)`。断言
   `gather(scatter回去, ids_restore)` 能还原原始顺序；mask 里 1 的个数 == 被遮数。
2. **encoder 只吃可见**：断言 encoder 输入序列长度 == `len_keep`（16），不是 64——这条断言
   直接证明「非对称」落地了。
3. **补 mask token + ids_restore 还原 + decoder**：断言 decoder 输出回到 `[B,64,48]`。
4. **masked-only loss + 训练**：跑 ⭐。先预测起步 loss（归一化像素的 MSE，量级约多少？）。
   再加一条对照：在**可见 patch** 上也算 loss，会发现它几乎为 0（模型在抄自己的输入）——
   反证「loss 只在 masked patch 上算」的必要性。

## ⭐ 必做的 sanity

`train_sanity.py`：CIFAR 训 ~400 步，断言「后 20 步均值 loss < 前 20 步 × 0.9」，存
`原图 | 75% 遮挡 | 重建` 三联图。toy 规模重建偏糊但**结构对**即证明机制 work。

## ✅ 盲讲检查点

1. encoder 输入序列长度是多少？mask token 在哪一步才出现？
2. 位置编码为什么在 masking **之前**加？
3. `ids_restore` 是什么、用来干嘛？怎么从 `ids_shuffle` 得到它？
4. loss 为什么只在 masked patch 上算？在可见 patch 上算会怎样？
5. 省算力的根本原因？为什么遮 75% 而 BERT 只遮 15%？

> 下一站 [`../videomae`](../videomae)：同一骨架，只改 3 处就变视频版。
