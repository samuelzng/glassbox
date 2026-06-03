# 主线 3 · 视觉 & 视频理解

> 把 Transformer 用到像素上。一条从「图像分类 backbone」→「自监督预训练」→「视频」
> →「图文对齐」→「视觉接进 LLM」的线，每一步只引入一个新机制。

## 组件表（按读的顺序）

| # | 组件 | 论文 | 盯死的反直觉机制 |
|---|---|---|---|
| 1 | [`vit/`](vit) | ViT (Dosovitskiy 2020) | 把图当一串「词」：切 patch → 线性投影 → 当 token 喂进纯 Transformer，无任何视觉特化设计 |
| 2 | [`mae/`](mae) | MAE (He 2021) | **非对称 encoder–decoder**：深 encoder 只看 25% 可见 patch，从不碰 mask token |
| 3 | [`videomae/`](videomae) | VideoMAE (Tong 2022) | **tube masking + 90%**：沿时间遮同一批空间位置，防止抄相邻帧 |
| 4 | [`clip/`](clip) | CLIP (Radford 2021) | **对比对齐**：图、文各过一个塔，用 InfoNCE 把配对的图文拉近 → 零样本分类的根 |
| 5 | [`vl/`](vl) | LLaVA-style | **projector = 图像的 embedding 层**：视觉 token 拼进序列后 LLM 一视同仁 |

## 复现顺序

```
vit (backbone) → mae (自监督) → videomae (视频) → clip (对比多模态) → vl (视觉→LLM)
```

- `mae` / `videomae` 共享非对称 enc/dec + shuffle-gather masking 的骨架，`videomae` 只改 3 处。
- `clip` 和 `vl` 都是「图文打通」，但路线不同：CLIP 用**对比**对齐到共享空间；vl 用 **projector** 把视觉 token 投进 LLM 词空间后拼接。读完能讲清两者区别。
- `vl` 是这条线的收尾，也是接 [`../llm`](../llm) 的桥：视觉 token 进了 LLM 序列后，走的就是普通 causal attention。
