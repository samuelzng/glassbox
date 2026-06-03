# posttrain/dpo — 直接偏好优化（Direct Preference Optimization）

> **📌 一手出处 / official reference**
> - 论文 *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* — Rafailov et al., NeurIPS 2023 · [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
> - 官方实现 [eric-mitchell/direct-preference-optimization](https://github.com/eric-mitchell/direct-preference-optimization)；HuggingFace [trl](https://github.com/huggingface/trl) 的 `DPOTrainer`

> 学习笔记 + 复现 spec。目标：复现完 [`dpo.py`](dpo.py) 后，能讲清
> **「为什么一个分类式 loss 就能替代整套 RM + PPO」**。

## 一句话 mental model

DPO 的洞见：RLHF「最大化奖励 + KL 约束」的**最优策略有闭式解**，把它反过来解，会发现
**奖励可以用 `log(π/π_ref)` 隐式表示**。代入偏好的 Bradley–Terry 公式，整个 RL 流程就坍缩成
一个**直接在偏好对上算的分类式 loss**——不训奖励模型、不采样、不要 critic。

## 🎯 盯死的反直觉机制：RM + rollout + critic → 一个 loss

```
PPO:   训 RM → policy 采样 → critic 估 advantage → KL 约束的 RL 更新   (4 模型，在线)
DPO:   只要 policy + 冻结的 ref，直接吃离线偏好对 (chosen, rejected)：

loss = − log σ( β·[ (logπ(y_w|x) − logπ_ref(y_w|x))
                   −(logπ(y_l|x) − logπ_ref(y_l|x)) ] )
                  └────── y_w=chosen 的隐式奖励 ──┘ └── y_l=rejected ──┘
```

**隐式奖励** `r(x,y) = β·log(π(y|x)/π_ref(y|x))`：policy 比 ref 更愿意生成的回答 = 高奖励。
DPO 就是直接把「chosen 的隐式奖励 − rejected 的隐式奖励」推大。

## === DIFF vs PPO-RLHF ===

| | PPO-RLHF | DPO |
|---|---|---|
| 奖励模型 | 要单独训 | **不要**（奖励是隐式的） |
| 在线采样 / rollout | 要 | **不要**（用现成离线偏好对） |
| critic / value | 要 | **不要** |
| 需要的模型 | 4 个 | **2 个**（policy + 冻结 ref） |
| 训练范式 | 强化学习 | **监督式分类 loss** |

## 你要复现的目标（接口 + TODO）

文件：`dpo.py`、`train_sanity.py`。policy/ref 用 [`../../llm/nano`](../../llm/nano)（ref = policy 的冻结副本）。

```python
def dpo_loss(policy, ref, prompt, chosen, rejected, beta):
    # 对 chosen/rejected 各算 policy 和 ref 的序列 log-prob（对 response token 求和）
    # logits = beta * ((lp_pol_w - lp_ref_w) - (lp_pol_l - lp_ref_l))
    # return -F.logsigmoid(logits).mean()
```

## ⭐ 必做的 sanity

用和 [`../reward-model`](../reward-model) 同一套玩具偏好对：
- 断言训练后 chosen 相对 rejected 的 **隐式奖励 margin 持续增大**（chosen 的 log-prob 优势上升）；
- 和「RM + PPO」对照：偏好方向一致，但 DPO 代码量和显存都少一大截。

## ✅ 盲讲检查点

1. DPO 需要几个模型？分别是什么、谁冻结？（policy + 冻结 ref）
2. 隐式奖励的表达式是什么？为什么不再需要单独的 RM？
3. 为什么 DPO 不需要在线采样 / critic？（离线偏好对 + 闭式解代入）
4. ref 模型在 loss 里起什么作用？（提供 KL 约束的锚，防 policy 跑偏）
5. β 控制什么？（偏离 ref 的强度 / 隐式 KL 温度）

> 下一站 [`../grpo`](../grpo)：DPO 走「不要 RL」，GRPO 走「要 RL 但不要 critic」——两条不同的简化路线。
