# 主线 3 · 视觉 & 视频理解

> 把 Transformer 用到像素上。主线：从「图像分类 backbone」→「自监督预训练」→「视频」
> →「图文对齐」→「视觉接进 LLM」，每一步只引入一个新机制。
> 支线回答 2024–25 的第一性问题：**视觉 token 怎么省着花**——长视频 MLLM 的成本账。

## 主线（按读的顺序）

| # | 组件 | 论文 | 盯死的反直觉机制 |
|---|---|---|---|
| 1 | [`vit/`](vit) | ViT (Dosovitskiy 2020) | 把图当一串「词」：切 patch → 线性投影 → 当 token 喂进纯 Transformer，无任何视觉特化设计 |
| 2 | [`mae/`](mae) | MAE (He 2021) | **非对称 encoder–decoder**：深 encoder 只看 25% 可见 patch，从不碰 mask token |
| 3 | [`videomae/`](videomae) | VideoMAE (Tong 2022) | **tube masking + 90%**：沿时间遮同一批空间位置，防止抄相邻帧 |
| 4 | [`clip/`](clip) | CLIP (Radford 2021) | **对比对齐**：图、文各过一个塔，用 InfoNCE 把配对的图文拉近 → 零样本分类的根 |
| 5 | [`vl/`](vl) | LLaVA-style | **projector = 图像的 embedding 层**：视觉 token 拼进序列后 LLM 一视同仁 |

## 支线 · 视觉 token 预算（以 vl 完成为前提）

[`../llm`](../llm) 的 2025 支线管**每 token 的 KV 单价**（mla / localattn），这条支线管 **token
的数量**。两者相乘就是长视频 MLLM 的全部成本：一张 336px 图 = 576 token，1 fps × 10 min ≈ 35 万
token，仅 KV cache 就 ~42 GiB（7B 级、bf16，单价见 [`../llm/gqa`](../llm/gqa)）。
完整账本与机制谱系见 [`tokbudget/`](tokbudget)。

| # | 组件 | 预算环节 | 盯死的反直觉机制 |
|---|---|---|---|
| 6 | [`tokbudget/`](tokbudget) | 总账 | 成本是**连乘**：帧数 × 每帧 token 数 × 每 token KV 单价——没有哪个单一「压缩算法」能独自救场 |
| 7 | [`framesample/`](framesample) | encoder 之前 | 同样 8 帧预算，uniform 对 5 帧事件的召回 ≈7%、query-guided ≈100%；**最上游的损失不可逆** |
| 8 | [`tokmerge/`](tokmerge) | encoder 之后 | pixel-shuffle 把信息搬进通道维而非丢掉；**冗余度是任务相关的**——粗任务 16× 不掉点，细任务早崩了 |
| 9 | [`mrope/`](mrope) | 进序列之时 | 零新参数，position id 从标量变 (t,h,w) 三元组；纯文本时**数值上精确退化**为 1D RoPE |
| 10 | [`siglip/`](siglip)（可做） | 视觉塔的训练目标 | 去掉 softmax 的 batch 内竞争居然不掉点；bias 初始化 −10 对付 B 正对 vs B²−B 负对的不平衡 |

暂缓（挂名不建目录）：**dividedattn**——TimeSformer
[2102.05095](https://arxiv.org/abs/2102.05095) 的时空分解注意力，和
[`../llm/localattn`](../llm/localattn) 的「感受野靠叠层合成」同构，机制经典；但 2025 的 video
MLLM 主流是 image encoder 逐帧 + LLM 管时序，归档为 classic video-encoder mechanism，8 月后看。

## 复现顺序

```
主线   vit ─┬─► mae ─► videomae            （自监督枝，可后置）
            ├─► clip ─► siglip（可做）      （对齐枝）
            └─(+ ../llm/nano)─► vl          （MLLM 主干）
支线   tokbudget（纯算账，无依赖，半天）
       vl ─► tokmerge ─► framesample
       vl (+ ../llm/rope) ─► mrope
```

- `mae` / `videomae` 共享非对称 enc/dec + shuffle-gather masking 的骨架，`videomae` 只改 3 处。
- `clip` 和 `vl` 都是「图文打通」，但路线不同：CLIP 用**对比**对齐到共享空间；vl 用 **projector**
  把视觉 token 投进 LLM 词空间后拼接。读完能讲清两者区别。
- **NTHU 动线**（出发前 → 新竹周末）：vit → vl → tokbudget → tokmerge → mrope → framesample
  → clip → siglip。自监督枝和 dividedattn 不挡 MLLM 主干，可排 8 月后。
- `vl` 是主线收尾、支线的 gate，也是接 [`../llm`](../llm) 的桥：视觉 token 进了 LLM 序列后，
  走的就是普通 causal attention——从那一刻起，每个视觉 token 都按
  [`../llm/kvcache`](../llm/kvcache) 的价目表收费。
