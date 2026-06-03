# posttrain/ppo-rlhf — 经典 RLHF（PPO）

> **📌 一手出处 / official reference**
> - RLHF 流程 *Training language models to follow instructions with human feedback*（InstructGPT）— Ouyang et al., 2022 · [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
> - PPO 算法 *Proximal Policy Optimization Algorithms* — Schulman et al., 2017 · [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
> - 参考实现 HuggingFace [trl](https://github.com/huggingface/trl) 的 `PPOTrainer`

> 学习笔记 + 复现 spec。目标：复现完 [`ppo.py`](ppo.py) 后，能讲清
> **「RLHF 为什么要 4 个模型，以及 KL 罚到底防的是什么」**。

## 一句话 mental model

让模型**自己生成**回答 → 用 [`../reward-model`](../reward-model) 给回答打分当奖励 → 用强化学习
（PPO）把策略往「高奖励」方向推。但为了不让它为了刷分变成胡言乱语，加一个 **KL 惩罚**把它
拴在原始模型附近。这是最完整、也最重的一条路线。

## 🎯 盯死的反直觉机制 1：同时有 4 个模型

| 模型 | 是否训练 | 作用 |
|---|---|---|
| **policy**（策略） | ✅ 训练 | 正在被优化的 LM（通常从 SFT 模型初始化） |
| **reference**（参考） | ❄️ 冻结 | policy 的初始副本，用来算 KL，防跑偏 |
| **reward model** | ❄️ 冻结 | 给完整回答打标量分 |
| **value / critic**（价值网络） | ✅ 训练 | 估计每个状态的期望回报，给 advantage 当 baseline |

## 🎯 盯死的反直觉机制 2：KL 罚防 reward hacking

奖励信号 = `RM 分数 − β · KL(policy ‖ reference)`。

- 没有 KL 项时，policy 会去**钻 RM 的空子**（reward hacking）：找到一些 RM 误判高分、
  但其实是重复/乱码的回答，把奖励刷爆，语言能力崩坏。
- KL 项像根皮筋，把 policy 拴在「还像个正常语言模型」的 reference 附近，**只允许小步偏移**。

PPO 本身的贡献是「**clip** 掉过大的策略更新」，让 on-policy 训练更稳。

## 你要复现的目标（流程 + TODO）

文件：`ppo.py`、`train_sanity.py`。policy/ref 用 [`../../llm/nano`](../../llm/nano)，奖励用 [`../reward-model`](../reward-model)。

```
for prompt in batch:
    completion ~ policy(prompt)                  # rollout，采样生成
    reward = RM(prompt, completion) - β*KL(policy‖ref, 逐 token)
    advantage = reward - value(state)            # critic 当 baseline（可用 GAE）
    loss = PPO_clip(policy, advantage) + value_loss(critic)
    更新 policy 和 critic
```

## ⭐ 必做的 sanity

用一个**可验证的玩具奖励**（如「回答里数字等于 prompt 两数之和则 +1」）：
- 断言训练中**平均奖励上升**、同时 **KL 保持有界**；
- 对照：**把 β 调到 0（去掉 KL）**，观察 policy 退化（奖励虚高但生成崩坏）——直观看到 reward hacking。

## ✅ 盲讲检查点

1. 说出 RLHF 的 4 个模型，各自训练还是冻结、作用是什么。
2. critic / value 网络是干嘛的？（给 advantage 当 baseline）
3. 奖励信号里的 KL 项防的是什么？去掉会怎样？
4. PPO 的 clip 在解决什么问题？（限制单步策略更新幅度）
5. 这条线最重的地方在哪？（4 个模型同时在显存里）——这正是 DPO / GRPO 要简化的。

> 下一站 [`../dpo`](../dpo)：把 RM + RL + critic **全部省掉**，只留偏好对。
