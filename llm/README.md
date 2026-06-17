# 主线 1 · 现代 LLM 架构

> 一个 2024 的 decoder-only LLM（Llama / Qwen 系），相对 2017 的原版 Transformer，**到底改了哪几块**。
> 每个组件单独拎出来一个机制，最后 `nano/` 把它们拼成一个能 train、能生成的 mini-Llama。

## 起点：classic pre-norm block（你已经会的）

```
x = x + MHA(LayerNorm(x))          # 全量多头注意力，位置靠外部 absolute/learned PE 加一次
x = x + FFN(LayerNorm(x))          # FFN = Linear → GELU → Linear（2 个矩阵）
```

下面每个组件就是把这块里的**一个零件**换成现代版，README 里用 `=== DIFF vs classic block ===` 标出来。

## 组件表

| # | 组件 | 换掉的零件 | 盯死的反直觉机制 |
|---|---|---|---|
| 1 | [`rope/`](rope) | 位置编码 | 位置不再「加」一个向量，而是把 Q/K **旋转**；点积只依赖**相对**位置差，每层都重施 |
| 2 | [`rmsnorm/`](rmsnorm) | LayerNorm | 扔掉**减均值**和 bias，只按 RMS 缩放——centering 居然可省 |
| 3 | [`swiglu/`](swiglu) | FFN | 2 矩阵→3 矩阵 + **门控**，hidden 维 ×⅔ 保参数量 |
| 4 | [`gqa/`](gqa) | 多头注意力 | K/V head 数 < Q head 数、分组共享。砍它**不为省训练算力，为推理 KV-cache 显存** |
| 5 | [`nano/`](nano) | —（组装） | 把 1–4 拼成 decoder-only mini-Llama，next-token 训练，证明能生成 |
| 6 | [`kvcache/`](kvcache) | —（推理） | 训练**并行**算全序列；推理**自回归**缓存过去 K/V，每步 `O(L²)→O(L)`；输出靠 sampler |
| 7 | [`moe/`](moe) | FFN（进阶） | 每 token 只**路由**到 top-k 个 expert → 参数涨但单 token FLOPs 几乎不变 |
| 8 | [`bpe/`](bpe) | 输入（独立） | 不按词不按字符，**统计高频字节对反复合并**；解释 token 为什么带前导空格 |

### 2025 推理效率支线（全部以 kvcache 完成为前提）

| # | 组件 | 换掉的零件 | 盯死的反直觉机制 |
|---|---|---|---|
| 9 | [`mla/`](mla) | 多头注意力（进阶） | cache 里存的**既不是 K 也不是 V**，是低秩 latent；RoPE 与矩阵吸收冲突，逼出 decoupled 子空间 |
| 10 | [`localattn/`](localattn) | attention mask（进阶） | 大多数层只看最近 W 个，感受野靠深度叠回来；streaming 时**开头 token 删不得**（sink）|
| 11 | [`specdec/`](specdec) | —（推理，进阶） | 小模型打草稿、大模型**一次 forward 并行验收**；rejection sampling 保证分布与 target 逐 token 一致 |

## 复现顺序

```
rope ─┐
rmsnorm ─┼─► nano(组装+训练) ─► kvcache(推理生成) ─► moe / bpe（进阶，任选）
swiglu ─┤                          │
gqa ─────┘                         └─► mla ─► localattn ─► specdec（2025 支线）
```

1–4 互相独立、可任意顺序，但都建议先各自单文件验证（shape walk + 一个 sanity 断言）再进 `nano/` 组装。
9–11 必须排在 kvcache 之后：mla 的等价测试、specdec 的 `n_new` 验证步都直接吃 kvcache 的 `decode_step` 接口。
