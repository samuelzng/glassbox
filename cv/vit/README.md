# cv/vit — Vision Transformer

> **📌 一手出处 / official reference**
> - 论文 *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale* — Dosovitskiy et al., ICLR 2021 · [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
> - 官方实现 [google-research/vision_transformer](https://github.com/google-research/vision_transformer)；社区 [huggingface/pytorch-image-models (timm)](https://github.com/huggingface/pytorch-image-models)

> 学习笔记 + 复现 spec。目标：复现完 [`vit.py`](vit.py) 后，能讲清
> **「图像怎么变成一串 token 喂进纯 Transformer，且不靠任何视觉特化设计」**。

## 一句话 mental model

把图切成固定大小的 patch（如 16×16），每个 patch 拉平后过一个**线性投影**变成一个 D 维向量
——这就等价于 NLP 里的 token embedding。加一个 `[CLS]` token、加**可学习位置编码**，整串送进
**标准 Transformer encoder**，最后拿 `[CLS]` 的输出去分类。

## 🎯 盯死的反直觉机制：几乎不引入图像专属归纳偏置

CNN 自带**局部性**和**平移等变性**两个强先验。ViT 几乎全扔了：唯一的图像假设就是「切成 2D patch」。
self-attention 从**第一层**就有**全局感受野**。代价是数据量小时不如 CNN，但数据足够大时上限更高
——「归纳偏置是小数据的捷径、大数据的天花板」。

## 数据流（`vit.py` 的 shape walk）

```
图 [B,3,32,32]
  │ patchify（Conv2d stride=patch 或 unfold），patch=4 → 8×8=64 个 patch
  ▼
patches [B, 64, 48]            (每个 patch 3×4×4=48 维像素)
  │ 线性投影到 D
  ▼
tokens  [B, 64, D]
  │ 前面拼 [CLS] → [B,65,D]；加可学习位置编码
  ▼
Transformer encoder (pre-norm, L 层)  → [B,65,D]
  │ 取第 0 个（CLS）→ Linear(D, n_classes)
  ▼
logits  [B, n_classes]
```

## 复现注意（常见坑）

- **scaling 用 `√d_k`（每头维度），不是 `√D`**。搞错 attention 分布会崩。
- 位置编码用最简单的 **1D learnable** 即可（论文消融：1D≈2D≈relative）。
- 分类头可用 `[CLS]`，也可用 **GAP（全局平均池化）**，效果几乎一样、实现更简单。

## 你要复现的目标（TODO）

文件：`vit.py`（shape walk）、`train_sanity.py`。可复用一个标准 pre-norm Transformer encoder。

```python
class ViT(nn.Module):
    # patch_embed (Conv2d or unfold+Linear), cls_token, pos_embed(learnable),
    # encoder (N × pre-norm block), head(Linear)
    def forward(self, img):   # [B,3,H,W] -> logits [B, n_classes]
```

## ⭐ 必做的 sanity

在 CIFAR-10 子集或合成图上训几百步：断言训练准确率明显上升（机制 work），不追 SOTA。

## ✅ 盲讲检查点

1. 一个「token」在 ViT 里对应什么？patchify 怎么做？
2. 位置信息从哪来？（可学习 1D 位置编码）`[CLS]` 是干嘛的、可用什么替代（GAP）？
3. ViT 相对 CNN 扔掉了哪两个归纳偏置？带来什么 trade-off？
4. attention 的 scaling 该用 `√d_k` 还是 `√D`？为什么？
5. 为什么说 self-attention「第一层就有全局感受野」？

> 下一站 [`../mae`](../mae)：不用标签，靠「遮住再重建」自监督预训练这个 backbone。
