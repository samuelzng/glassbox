# cv/videomae — VideoMAE

> **📌 一手出处 / official reference**
> - 论文 *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training* — Tong et al., NeurIPS 2022 Spotlight · [arXiv:2203.12602](https://arxiv.org/abs/2203.12602)
> - 官方实现 [MCG-NJU/VideoMAE](https://github.com/MCG-NJU/VideoMAE)

> 学习笔记 + 复现 spec。目标：复现完 [`videomae.py`](videomae.py) 后，能讲清
> **VideoMAE 相对 image MAE 改了哪几处、为什么改**。
> 复习时：先做文末盲讲检查点，全答上来就直接跑代码。

## 一句话 mental model

把 [image MAE](../mae) 搬到视频：patch 变成**时空小块 (tubelet)**，masking 变成**沿时间整条
遮的 tube masking**，掩码率从 75% 拉到 **90%**。骨架（非对称 enc/dec、共享 mask token、
shuffle/gather 还原、masked 归一化像素 loss）**完全没变**。

## ⭐ MAE → VideoMAE 的全部跨越就这张 diff 表

| 维度 | image MAE | VideoMAE | 为什么 |
|---|---|---|---|
| token | 2D patch `p×p` | **tubelet `tt×th×tw`**（Conv3d） | 一个 token 要覆盖一小段时间才能编码「运动」 |
| 输入 | `[B,3,32,32]` | `[B,3,8,32,32]`（多 T 维） | |
| masking | 每 token 独立随机遮 | **tube masking**：沿时间遮**同一批空间位置** | 防止抄相邻帧 |
| 掩码率 | 75% | **90%** | 视频时间冗余远大于图像空间冗余 |
| 注意力 | 全局 | joint / divided（时空分解省算力） | token 数 `T×H×W` 爆炸 |
| 其余 | — | **一模一样** | |

## 🎯 先猜：视频遮 patch，按「每帧独立随机遮」会出什么问题？（想 30 秒）

**答案：模型会抄相邻帧的同位置像素——任务退化。解法是 tube masking：沿时间遮同一批空间位置。**

```
   每帧独立随机遮 (✗)                 tube masking (✓)
t=0  ▓ · ▓ ·                      t=0  ▓ ▓ · ·
t=1  · ▓ · ▓  ← 换帧就露同位置      t=1  ▓ ▓ · ·  ← 同位置全程都遮
t=2  ▓ · · ▓     模型抄隔壁帧即可    t=2  ▓ ▓ · ·
```

**为什么必须沿时间遮同一批空间位置？** 合成数据里方块每帧只平移几像素，位置 `(t,h,w)` 被遮但
`(t±1,h,w)` 大概率还在原地可见。若**每帧独立**遮，模型只要学会「从相邻帧复制同位置像素」就能把
loss 压低——任务退化，学不到时序理解。tube masking 把一个空间位置从**所有帧**一起拿掉，逼模型
真正建模运动/语义。配合 90% 高掩码率，捷径被彻底堵死。

## 你要复现的目标（TODO）

文件：`videomae.py`、`data_synthetic.py`（合成移动方块）、`train_sanity.py`。
**直接拷 [`../mae`](../mae) 的骨架，只在上面四处打 `=== DIFF vs MAE ===` 标注的地方改**。

```python
# tubelet embed: nn.Conv3d(3, D, kernel=(tt,th,tw), stride=(tt,th,tw))
# tube masking:  先决定空间位置的遮挡，再沿 T 维广播到所有时间
```

## 🪜 实现阶梯（每级跑通断言再上一级）

直接拷 [`../mae`](../mae) 跑通的骨架，逐处替换：

1. **tubelet embed**：Conv2d → Conv3d，输入加 T 维。断言 `[B,3,8,32,32]` → patch 数
   `(8/tt)·(32/th)·(32/tw)`，shape walk 对上。
2. **tube masking**：先在 `(h,w)` 网格上抽要遮的空间位置，再沿 T 广播。断言**任一空间位置在
   所有帧上的遮挡状态完全一致**（这就是 tube 的定义，错了就退化成 per-frame）。
3. **掩码率 75% → 90%**，其余骨架不动，跑训练。
4. **对照实验**（⭐）：tube vs per-frame 各训一遍。跑之前先预测：哪种的 loss 会**异常低**？
   为什么低反而是坏事？

## ⭐ 必做的 sanity

`train_sanity.py`：合成视频训练，断言 loss 显著下降，存 `clips.png`（输入片段）和重建图。
可加对照：**每帧独立遮**时 loss 异常低（说明在抄帧）vs **tube masking** 下 loss 正常——直观证明 tube 的必要性。

## ✅ 盲讲检查点

1. tubelet 和 2D patch 的区别？为什么用 Conv3d？
2. tube masking 具体怎么遮？为什么不能每帧独立遮？
3. 掩码率为什么从 75% 提到 90%？
4. joint vs divided 注意力的取舍？（token 数爆炸时省算力）
5. 哪些部分和 image MAE 完全一样？哪 4 处变了？

> 下一站 [`../clip`](../clip)：换条路打通图文——用对比学习对齐，而不是重建。
