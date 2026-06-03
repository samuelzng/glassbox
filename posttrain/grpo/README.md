# posttrain/grpo — Group Relative Policy Optimization

> **📌 一手出处 / official reference**
> - 提出 GRPO *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* — Shao et al., 2024 · [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
> - 大规模应用 *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* — DeepSeek-AI, 2025 · [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
> - 参考实现 HuggingFace [trl](https://github.com/huggingface/trl) 的 `GRPOTrainer`

> 学习笔记 + 复现 spec。目标：复现完 [`grpo.py`](grpo.py) 后，能讲清
> **「GRPO 砍掉了 PPO 的 critic，advantage 改从哪来」**，以及它为什么和「可验证奖励」绝配。

## 一句话 mental model

还是 RL、还是在线采样，但**不要 value/critic 网络**了。对**同一个 prompt 一次采 G 个回答**，
每个打个分（最好是**可验证奖励**：数学答案对不对、代码过不过测试）。把每个回答的奖励**在这一组内
做归一化**（减组均值、除组标准差），就当作它的 advantage。**这一组样本自己当彼此的 baseline。**

## 🎯 盯死的反直觉机制：用「组内相对」替代 critic

PPO 要训一个 value 网络去估「这个状态平均能得多少分」当 baseline——这是另一个和 policy 同
量级的模型，占掉约一半显存。GRPO 的观察是：

```
对同一个 prompt 采 G 个回答  y_1 .. y_G，奖励 r_1 .. r_G
advantage_i = (r_i − mean(r_1..r_G)) / std(r_1..r_G)      ← 组内 z-score，无需 critic
```

「比这组的平均水平好多少」天然就是一个好 baseline。**省掉整个 critic 网络**，这是 GRPO 相对
PPO 最大的减法，也是它能在大模型推理训练里跑得起的关键。

## 🎯 反直觉机制 2：和「可验证奖励」(RLVR) 绝配

GRPO 常配**规则奖励**而非 RM：数学题对答案、代码跑测试 → 0/1 奖励，无需训奖励模型、不会被
reward hacking。这正是 **DeepSeek-R1** 让模型「自己涌现出长链推理」的训练配方核心。

推论：若**一组 G 个回答奖励全相同**（全对或全错），组内 advantage 全为 0 → **这条 prompt 这一步
没有学习信号**。所以题目难度要让模型「有时对有时错」，组内才有区分度。

## === DIFF vs PPO ===

| | PPO | GRPO |
|---|---|---|
| value / critic | **要**（占约一半显存） | **不要** |
| advantage 来源 | critic 估的 baseline | **同 prompt 的 G 个样本组内归一化** |
| 奖励 | 常用 RM | 常用**可验证规则奖励**（对错） |
| KL 约束 | 有 | 有（拴住 ref） |

## 你要复现的目标（流程 + TODO）

文件：`grpo.py`、`train_sanity.py`。policy 用 [`../../llm/nano`](../../llm/nano)。

```
for prompt in batch:
    answers = [policy.sample(prompt) for _ in range(G)]      # 一组 G 个
    rewards = [verify(prompt, a) for a in answers]           # 可验证奖励，如算术对错
    adv = (rewards - rewards.mean()) / (rewards.std() + eps) # 组内归一化，无 critic
    loss = PPO_clip(policy, adv, per-token) + β*KL(policy‖ref)
```

## ⭐ 必做的 sanity

玩具**可验证任务**（如两数相加，答对=1 答错=0）：
- 断言训练中**准确率 / 平均奖励上升**；
- 演示「**一组答案奖励全相同 → advantage 全 0 → 该步无梯度信号**」这条性质（GRPO 的标志特征）。

## ✅ 盲讲检查点

1. GRPO 相对 PPO 砍掉了哪个组件？（critic / value 网络）
2. 砍掉后 advantage 从哪来？（同 prompt 的一组 G 个样本，组内归一化）
3. 什么是「可验证奖励」？为什么 GRPO 和它绝配、且不易 reward hacking？
4. 一组 G 个回答奖励全一样时会发生什么？对出题难度有什么要求？
5. DPO 和 GRPO 都在简化 PPO，路线有何不同？（DPO=不要 RL；GRPO=要 RL 但不要 critic）

> 后训练主线收尾。三条减法路线（SFT→RM+PPO→DPO / GRPO）走完，回 [`../`](..) 总览。
