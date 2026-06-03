# posttrain/reward-model — 奖励模型（Reward Model）

> **📌 一手出处 / official reference**
> - 论文 InstructGPT — Ouyang et al., 2022 · [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
> - RLHF 起源 *Deep reinforcement learning from human preferences* — Christiano et al., 2017 · [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)；*Learning to summarize from human feedback* — Stiennon et al., 2020 · [arXiv:2009.01325](https://arxiv.org/abs/2009.01325)
> - 参考实现 HuggingFace [trl](https://github.com/huggingface/trl) 的 `RewardTrainer`

> 学习笔记 + 复现 spec。目标：复现完 [`reward_model.py`](reward_model.py) 后，能讲清
> **「只用『A 比 B 好』的相对偏好，怎么训出一个给回答打分的标量模型」**。

## 一句话 mental model

拿一个 LM 当骨架，在最后一个 token 上接一个 **标量头 `Linear(d, 1)`**，输入 `(prompt, response)`
输出**一个分数**。训练数据是**偏好对**：人类标了 `chosen ≻ rejected`。优化目标——让
`r(chosen) > r(rejected)`。它就是后面 PPO 的「奖励函数」。

## 🎯 盯死的反直觉机制：只需相对偏好，不需绝对分数

人很难给「这个回答 7.3 分」打绝对分，但很容易说「A 比 B 好」。RM 正是吃这种**成对比较**，
用 **Bradley–Terry / pairwise logistic** loss：

```
loss = − log σ( r(chosen) − r(rejected) )      # 只关心两者的差
```

推论：**奖励值只在「相对」意义上有效**——整体加一个常数 `r+c` 完全不影响排序和 loss。
所以 RM 的分数没有绝对刻度，只有「谁比谁高」。

## === 和分类/回归的区别 ===

| | 普通回归 | Reward Model |
|---|---|---|
| 标签 | 绝对数值 | 只有**成对偏好** chosen≻rejected |
| loss | MSE 到真值 | `−log σ(r_chosen − r_rejected)` |
| 输出刻度 | 有绝对意义 | **只有相对意义**（差不变即等价） |

## 你要复现的目标（接口 + TODO）

文件：`reward_model.py`、`train_sanity.py`。骨架可复用 [`../../llm/nano`](../../llm/nano)。

```python
class RewardModel(nn.Module):
    # backbone = nano LM (去掉 lm_head)，加 self.score = nn.Linear(d, 1)
    def forward(self, ids):           # [B, L] -> 取最后一个有效 token 的 hidden -> [B] 标量分
        ...
# loss = -F.logsigmoid(r_chosen - r_rejected).mean()
```

## ⭐ 必做的 sanity

造一个有明确偏好规则的玩具数据（例如「更长 / 更礼貌 / 含某关键词」的回答记为 chosen）：
- 训练后在 held-out 偏好对上断言 `r(chosen) > r(rejected)` 的准确率明显高于 0.5；
- 演示「给所有分数 +c」不改变 loss/排序，验证**奖励只有相对意义**。

## ✅ 盲讲检查点

1. RM 的输入和输出分别是什么？标量分从哪取（最后 token 的 head）？
2. 训练用的是绝对评分还是成对偏好？loss 长什么样（Bradley–Terry）？
3. 为什么说奖励值「只有相对意义」？整体平移为何无影响？
4. RM 在后训练流程里给谁用？（PPO 的奖励信号）
5. RM 和后面 DPO 的关系？（DPO 把 RM 这一步**省掉**了——见下一站）

> 下一站 [`../ppo-rlhf`](../ppo-rlhf)：用这个 RM 当奖励，跑经典 RLHF。
