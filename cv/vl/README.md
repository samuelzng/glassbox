# cv/vl — Vision → token → LLM 拼接（LLaVA-style）

> **📌 一手出处 / official reference**
> - 论文 *Visual Instruction Tuning*（LLaVA）— Liu et al., NeurIPS 2023 · [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)
> - 官方实现 [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)

> 学习笔记 + 复现 spec。目标：复现完 [`vl.py`](vl.py) 后，能讲清
> **视觉 token 如何拼进 LLM 序列、projector 的角色**。这是 CV 线接 [`../../llm`](../../llm) 的桥。
> 复习时：先做文末盲讲检查点，全答上来就直接跑代码。

## 一句话 mental model

图过一个**冻结的** ViT 拿 patch 特征 → 一个 **projector (MLP)** 把它投到 LLM 的词向量空间 →
把这些视觉向量**塞进文本序列里 `<image>` 占位符的位置** → LLM 拿到一条 `[文本|视觉|文本]` 的混合
序列，做**普通 causal self-attention**，根本不区分哪些位置来自像素、哪些来自单词。

## 🎯 先猜：LLM 怎么区分序列里哪些 token 来自图、哪些来自文字？（想 30 秒）

**答案：它根本不区分。projector 把视觉单位变成 `d_llm` 向量后，和文本 token 一视同仁做 causal
attention。projector 扮演的就是「图像的 embedding 层」。**

```
文本:  token id        ──nn.Embedding──►  d_llm 向量
图像:  ViT patch 特征   ──projector(MLP)──►  d_llm 向量      ← 同一个目标空间！
```

文本进 Transformer 靠 `token id → 查表 → 向量`；图像没有 id，但 projector 扮演**完全对称**的
角色：把视觉单位变成一个 LLM 能吃的 `d_llm` 向量。一旦都变成 `d_llm` 向量，拼在一起 LLM 一视同仁。
**整个组件最值钱的就是 `embed_sequence` 里 `merged[is_img] = vis` 那一行。**

## tensor 流（`vl.py` 的 shape walk）

```
pixels [B,3,32,32] ─ frozen ViT(patch16→4 patch) ─► feats [B,4,d_vision]
                       │ projector (Linear→GELU→Linear)
                       ▼
                     vis [B,4,d_llm]              ← 视觉 token，已在词向量空间
文本 ids [B,7]=[BOS,IMG,IMG,IMG,IMG,<colour>,EOS] ─ tok_embed ─► [B,7,d_llm]
                       │  *** 把 vis 覆盖到 IMG 占位符的 4 个位置 ***
                       ▼
merged [B,7,d_llm]=[BOS|vis0 vis1 vis2 vis3|colour|EOS]
                       │ + 位置编码 → causal Transformer
                       ▼
logits [B,7,vocab]     → 在最后一个 IMG 位置预测颜色词
```

## toy 任务为什么能验证「视觉真进了 LLM」

图像是一个**随机颜色方块**，文本模板对所有样本都一样、要 LLM 说出颜色名。**区分样本的唯一信息
就是图像**。所以只有当梯度沿 `LM → 视觉 token → projector` 回传、projector 学会把像素特征变成
LM 读得懂的 token，颜色准确率才可能上去。✓ 证明 concat pipeline 端到端可微、视觉信息确实流进了 LM。

## 和真实 LLaVA 的关系（两阶段）

| | 冻结 | 训练 | 目的 |
|---|---|---|---|
| Stage 1 对齐 | 预训练 LLM + ViT | **只训 projector** | 让 projector 学会翻译到 LLM 词空间 |
| Stage 2 指令微调 | ViT（通常） | projector + **解冻 LLM** | 教模型按指令用视觉信息 |

本 toy 的小 LM 是**从零和 projector 一起训**的（它没有语言先验，冻结无意义）；真实 LLaVA stage-1
是冻结预训练 LM、只训 projector。被演示的**机制**（projector + splice + 混合序列 causal attention）两者一样。

## 你要复现的目标（TODO）

文件：`vl.py`、`train_sanity.py`。视觉塔可复用 [`../vit`](../vit)，LM 用 [`../../llm/nano`](../../llm/nano) 的 causal 版。

```python
class VLModel(nn.Module):
    # vision (frozen ViT) + projector(MLP d_vision→d_llm) + lm(causal Transformer)
    def embed_sequence(self, ids, pixels):
        emb = tok_embed(ids)                 # [B,L,d_llm]
        vis = projector(vision(pixels))      # [B,n_img,d_llm]
        emb[ids == IMG_TOKEN] = vis.reshape(-1, d_llm)   # ★ splice
        return emb
```

splice 那行的隐藏前提（写错就静默错位）：`ids == IMG_TOKEN` 的布尔 mask 按**行优先**展平，
`vis.reshape(-1, d_llm)` 也按 batch×n_img 行优先展平——两边顺序必须一致，且
**占位符个数 == 视觉 token 数**（每张图 n_img 个，整个 batch 共 `B·n_img` 个 IMG）。
先写一条断言把这两个数卡死，再训练。

## 🪜 实现阶梯（每级跑通断言再上一级）

1. **冻结 ViT + projector 出视觉 token**：断言 `vis` 是 `[B,n_img,d_llm]`，且
   `vision` 的参数 `requires_grad=False`（冻结真的生效了）。
2. **splice**：断言 IMG 占位符个数 == `B·n_img`；splice 后非 IMG 位置的 embedding 和
   `tok_embed(ids)` 逐元素相等（只覆盖了该覆盖的位置）。
3. **端到端 forward + 训练**：跑 ⭐，颜色准确率 → 1.0。
4. **可微性反证**（机制的直接证据）：把 projector 的梯度 detach 掉再训一次 → 准确率卡在
   chance。证明「颜色信息只能沿 LM→视觉 token→projector 这条路回传」。

## ⭐ 必做的 sanity

`train_sanity.py`：训 ~400 步，颜色准确率 →1.0，存 `predictions.png`。

## ✅ 盲讲检查点

1. projector 的作用用一句话类比？
2. 视觉向量怎么「进入」序列？维度为什么必须是 `d_llm`？
3. LLM 怎么区分视觉 token 和文本 token？
4. 为什么这个 toy 必须靠图像才能答对颜色？
5. 真实 LLaVA 两阶段分别冻结/训练什么？本 toy 哪里不同、为什么？
6. 和 [`../clip`](../clip) 的「对比对齐」相比，这条路线打通图文的方式有何本质不同？

> CV 主线收尾，也接上了 [`../../llm`](../../llm)。三条主线读完回 [根 README](../..) 总览。
