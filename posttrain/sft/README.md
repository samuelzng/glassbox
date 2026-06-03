# posttrain/sft — 监督微调（Supervised Fine-Tuning）

> **📌 一手出处 / official reference**
> - 论文 *Training language models to follow instructions with human feedback*（InstructGPT，SFT 是其第一阶段）— Ouyang et al., 2022 · [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
> - 参考实现 HuggingFace [trl](https://github.com/huggingface/trl) 的 `SFTTrainer`

> 学习笔记 + 复现 spec。目标：复现完 [`sft.py`](sft.py) 后，能讲清
> **「SFT 其实就是 next-token，唯一的花活是 loss 只算在 response 上」**。

## 一句话 mental model

拿一批 `(指令/prompt, 理想回答/response)` 配对，把它们拼成一条序列喂给 base model，
做**和预训练一模一样的 next-token**——但 **loss 只在 response 的 token 上算，prompt 部分屏蔽掉**。
模型于是学会「看到这种指令，就该产出这种回答」。

## 🎯 盯死的反直觉机制：没有 RL，关键只是 loss mask

很多人以为「对齐」从这步就开始 RL 了。**没有。** SFT 是纯监督的交叉熵，和预训练唯一的差别是：

```
序列:   [ prompt tokens .......... | response tokens .......... ]
labels: [ -100  -100  ...  -100    | r1   r2   r3   ...   <eos> ]
                 ↑ 屏蔽(不算loss)              ↑ 只在这里算 loss
```

**为什么屏蔽 prompt？** 我们要模型学会「**给出回答**」，不是学会「复述指令」。
若 prompt 也算 loss，模型会把容量浪费在背诵指令分布上，甚至倾向自己生成指令。

## === DIFF vs 预训练 ===

| | 预训练 | SFT |
|---|---|---|
| 数据 | 海量原始文本 | 少量 `(prompt, response)` 对 |
| loss | 每个 token 都算 | **只在 response 上算**（prompt 置 -100） |
| 目标 | 学语言/世界知识 | 学「按指令格式回答」 |
| 是否 RL | 否 | **否**（仍是交叉熵） |

## 你要复现的目标（接口 + TODO）

文件：`sft.py`、`train_sanity.py`。底座 = [`../../llm/nano`](../../llm/nano)。

```python
def build_example(prompt_ids, response_ids):
    input_ids = prompt_ids + response_ids
    labels    = [-100]*len(prompt_ids) + response_ids        # 屏蔽 prompt
    return input_ids, labels
# loss = F.cross_entropy(logits[:, :-1], labels[:, 1:], ignore_index=-100)
```

## ⭐ 必做的 sanity

在几条玩具指令对上训练：
- 断言模型在这些 prompt 下能**复现对应 response**（loss→很低）；
- 对照实验：**去掉 prompt 屏蔽**重训，观察模型开始倾向于「续写/生成 prompt」而非干净作答——
  直观证明 loss mask 的作用。

## ✅ 盲讲检查点

1. SFT 和预训练的唯一本质差别是什么？（loss mask 到 response）
2. SFT 里有 RL 吗？（没有，纯交叉熵）
3. 为什么要屏蔽 prompt 的 loss？不屏蔽会怎样？
4. `-100`（ignore_index）在 PyTorch 交叉熵里起什么作用？
5. SFT 在整条后训练流程里处于什么位置？（先 SFT 给个像样起点，再上偏好优化）

> 下一站 [`../reward-model`](../reward-model)：从「告诉它标准答案」转向「只告诉它哪个更好」。
