# cv/clip — Contrastive Language-Image Pre-training

> **📌 一手出处 / official reference**
> - 论文 *Learning Transferable Visual Models From Natural Language Supervision* — Radford et al., 2021 · [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
> - 官方实现 [openai/CLIP](https://github.com/openai/CLIP)；社区 [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)

> 学习笔记 + 复现 spec。目标：复现完 [`clip.py`](clip.py) 后，能讲清
> **「没有分类标签，只靠『哪句话配哪张图』就能学出可零样本分类的表征」**。
> 复习时：先做文末盲讲检查点，全答上来就直接跑代码。

## 一句话 mental model

两个塔：图像塔（如 [ViT](../vit)）、文本塔（Transformer）。各自把输入编码成向量、投影到**同一个
共享空间**、L2 归一化。一个 batch 里有 B 对配对的 (图, 文)：让**配对的**图文向量相似度高、**不配对的**
低。训练信号全来自「同 batch 内谁配谁」，不需要任何人工类别标签。

## 🎯 先猜：没有任何类别标签，CLIP 的监督信号从哪来？（想 30 秒）

**答案：一个 batch 内「谁配谁」——相似度矩阵的对角线是正样本，其余全是负样本。**

```
        文本_1  文本_2  ...  文本_B
图_1   [ ✓     ✗          ✗  ]        相似度矩阵 S = 图嵌入 @ 文嵌入ᵀ / τ
图_2   [ ✗     ✓          ✗  ]        正样本 = 对角线（配对）
...                                    负样本 = 同 batch 其它所有（不配对）
图_B   [ ✗     ✗          ✓  ]        loss = 行 softmax-CE + 列 softmax-CE（对称）
```

- **正负样本是 batch「免费」给的**：第 i 张图的正例是第 i 句话，负例是 batch 里其它 B−1 句。
  所以 **batch 越大、负样本越多、对比越强**（CLIP 用了 32k 的 batch）。
- **温度 τ** 缩放相似度，控制分布尖锐程度（可学习）。
- 这套就是 **InfoNCE / 对称交叉熵**，标签是 `arange(B)`。

## 🎯 零样本分类怎么白送出来

训练完，要分类一张图到 {cat, dog, ...}：把每个类名套模板（"a photo of a {cat}"）过文本塔得到
向量，和图像向量算相似度，取最高的那个类。**没在分类数据上训过，却能分类**——因为图文已被对齐到
同一空间。这是 CLIP 最有名的能力。

## === DIFF: CLIP vs MAE vs vl ===

| | MAE | CLIP | [vl](../vl) |
|---|---|---|---|
| 怎么打通图文 | 不打通（纯视觉自监督） | **对比**对齐到共享空间 | projector 投进 LLM 词空间后拼接 |
| 监督 | 重建像素 | batch 内配对 | LLM 的 next-token |
| 产物 | 视觉 encoder | 图/文双塔 + 共享空间 | 能看图说话的 LLM |

## 你要复现的目标（接口 + TODO）

文件：`clip.py`、`train_sanity.py`。图像塔复用 [`../vit`](../vit)。

```python
class CLIP(nn.Module):
    # image_encoder (ViT) + proj;  text_encoder (Transformer) + proj;  logit_scale(温度)
    def forward(self, images, texts):
        zi = F.normalize(img_proj(image_encoder(images)))   # [B, d]
        zt = F.normalize(txt_proj(text_encoder(texts)))     # [B, d]
        logits = zi @ zt.t() * logit_scale                  # [B, B]
        labels = torch.arange(B)
        loss = (CE(logits, labels) + CE(logits.t(), labels)) / 2
```

**文本塔 / tokenizer 在 toy 里怎么省**（否则会卡在「文本怎么变 token」）：合成描述词表只有几十个词
（颜色 × 形状），直接 **char-level 或 word-level 查表**——`vocab = sorted(set(所有描述里的词))`，
文本塔 = `nn.Embedding + 几层标准 Transformer encoder + 取末位/池化`。不需要 BPE、不需要预训练
tokenizer。`logit_scale` 存成 `log τ` 用 `.exp()` 取回，并 clamp 上限（防数值爆炸，CLIP 原版做法）。

## 🪜 实现阶梯（每级跑通断言再上一级）

1. **两个塔出归一化向量**：断言 `zi, zt` 都是单位长度（`norm ≈ 1`）、shape `[B,d]`。
2. **对称 InfoNCE loss**：手算一个 B=2 的小例子核对 `(行CE + 列CE)/2`；断言随机初始化时
   loss ≈ `ln(B)`（跑前先预测这个数）。
3. **训练 + 对角准确率**：跑 ⭐ 第一条，`argmax(logits, dim=1)` 落在 `arange(B)` 的比例上升。
4. **零样本演示**：跑 ⭐ 第二条。再做个对照——把 batch 调小（负样本变少），看对齐速度变慢,
   直观体会「batch 越大对比越强」。

## ⭐ 必做的 sanity

玩具配对数据（如「彩色形状图 ↔ 它的文字描述」）：
- 断言对比 loss 下降、**对角匹配准确率**（`argmax` 落在对角线）上升；
- 演示**零样本**：拿几个类名描述，给一张新图排名，取最相似的类。

## ✅ 盲讲检查点

1. CLIP 一个 batch 里的正样本/负样本分别是什么？为什么 batch 越大越好？
2. loss 为什么是「对称」的两个交叉熵？标签是什么？
3. 温度 τ 控制什么？为什么存成 `log τ`？
4. 零样本分类是怎么从「对齐的共享空间」白送出来的？
5. CLIP 打通图文的方式，和 [`../vl`](../vl) 的 projector 拼接有何本质不同？

> 下一站 [`../vl`](../vl)：另一条图文路线——把视觉 token 直接塞进 LLM 序列。
