# 主线 2 · RL 后训练 / 对齐

> base model 只会「续写」，不会「听话」「讲道理」。后训练就是把 base model 变成
> 听指令、合人类偏好、会推理的模型。这条线最好的学法是**沿演化读**——每一步都是
> 「上一步嫌太重，砍掉一个零件」。

## 一句话主线

> **没有标准答案（只有偏好 / 可验证的对错）时，怎么把模型往「更被偏好 / 更正确」推。**
> 从 RM+PPO 的重型，到 DPO 去掉 RL，到 GRPO 去掉 critic——一路做减法。

## 依赖

需要一个语言模型当底座。本仓库用 [`../llm/nano`](../llm/nano) 训出来的 mini-Llama，
不下大模型（代价：能力弱，但**机制完全一样**，能看清梯度怎么流）。

## 组件表（按读的顺序）

| # | 组件 | 它在做什么 | 盯死的反直觉机制 |
|---|---|---|---|
| 1 | [`sft/`](sft) | 监督微调 | 本质还是 **next-token**，没有任何 RL；唯一花活：loss **只在 response 上算，mask 掉 prompt** |
| 2 | [`reward-model/`](reward-model) | 训打分器 | 把「人类偏好对 chosen>rejected」训成一个**标量分**（Bradley–Terry / pairwise loss） |
| 3 | [`ppo-rlhf/`](ppo-rlhf) | 经典 RLHF | 重型：policy + RM 当奖励 + **critic/value 网络** + **KL 罚**防策略跑飞 |
| 4 | [`dpo/`](dpo) | 直接偏好优化 | **跳过 RM、跳过 RL**：一个分类式 loss 直接吃偏好对，隐式奖励 = log 概率比 |
| 5 | [`grpo/`](grpo) ⭐ | group-relative PO | 仍做 RL 但**砍掉 critic**：同 prompt 采 G 个答案，用**组内相对得分**当 advantage + **可验证奖励** → DeepSeek-R1 的核 |

## 怎么看这条线

```
        要 RM?   要 RL?   要 critic?
SFT      否       否        否        ← 只是模仿
PPO      是       是        是        ← 全套
DPO      否       否        否        ← 把偏好直接塞进 loss
GRPO     看情况    是        否        ← RL 但用组内相对优势代替 value 网络
```

读完能回答：**为什么 DPO 能不要 RM？GRPO 砍掉 critic 后 advantage 从哪来？**
