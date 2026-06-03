# llm/rope — RoPE 旋转位置编码

> **📌 一手出处 / official reference**
> - 论文 *RoFormer: Enhanced Transformer with Rotary Position Embedding* — Su et al., 2021 · [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
> - 参考实现 Llama 官方 [meta-llama/llama](https://github.com/meta-llama/llama) 里的 `apply_rotary_emb`

> 学习笔记 + 复现 spec。目标：复现完 [`rope.py`](rope.py) 后，能对着代码盲讲
> **「为什么旋转 Q/K 就等价于相对位置编码」**。

## 一句话 mental model

不再像原版那样给 embedding **加**一个位置向量，而是在每层 attention 里，把 query / key
向量按它所在的位置**旋转一个角度**（位置越靠后转得越多）。神奇之处：两个向量旋转后再做点积，
结果**只取决于它俩的位置差 `m−n`**——绝对地转，却换来了相对位置。

## 🎯 盯死的反直觉机制：绝对旋转 → 相对注意力

把 `d_k` 维特征**两两配对**成一个个 2D 平面。第 `i` 对在位置 `m` 处旋转角度 `m·θ_i`，
不同对用不同频率 `θ_i = base^(−2i/d_k)`（像一排转速不同的钟表指针）。

```
位置 m 的 query:  q  ──旋转 m·θ──►  q_m
位置 n 的 key  :  k  ──旋转 n·θ──►  k_n

点积   q_m · k_n  =  f(q, k, m−n)      ← 只依赖相对偏移 m−n，绝对位置消掉了
```

**为什么点积只剩相对位置？** 旋转矩阵满足 `R(m)ᵀ R(n) = R(n−m)`。q 转 `m`、k 转 `n`，
做内积时一个转置吃掉绝对部分，只留下角度差 `n−m`。这就是 RoPE 全部的魔法。

由此自然得到两个性质：
- **相对位置免费**：模型天然知道「你在我前面几格」，而不是「你在第几格」。
- **可外推**：训练时没见过的更长距离，旋转角度照样能算 → 长上下文扩展（NTK / 位置插值）就建立在这上面。

## === DIFF vs classic block ===

| 维度 | 原版 absolute / learned PE | RoPE |
|---|---|---|
| 加在哪 | 输入 embedding 上**加**一个向量 | attention 里把 **Q、K 旋转**（V 不动） |
| 加几次 | 只在第 0 层输入加 **1 次** | **每一层** attention 都重施 |
| 位置性质 | 绝对 | 点积后变**相对** |
| 可学习参数 | 有（learned 版） | **无**（纯几何变换，零参数） |
| 外推 | 差（超过训练长度就乱） | 好（配 scaling 可大幅外延） |

## 你要复现的目标（shape 契约 + TODO）

文件：`rope.py`（shape walk）、`train_sanity.py`（机制验证）。

**接口**（建议签名）：
```python
def build_rope_cache(seq_len, d_k, base=10000.0):
    # 返回 cos, sin，形状 [seq_len, d_k]（每对维度共享角度，按需 repeat 到 d_k）
    ...

def apply_rope(x, cos, sin):
    # x: [B, H, L, d_k] -> 旋转后 [B, H, L, d_k]，形状不变
    # 核心: x*cos + rotate_half(x)*sin     （rotate_half: [x1,x2] -> [-x2,x1]）
    ...
```

**集成点**：在注意力里，算完 `q,k = wq(x), wk(x)`、reshape 成 `[B,H,L,d_k]` 之后、
做 `qkᵀ` 之前，对 **q 和 k**（不含 v）各调一次 `apply_rope`。其余和 classic MHA 一样。

**shape walk（`rope.py` 要打印）**：
```
q [B,H,L,d_k] --apply_rope--> [B,H,L,d_k]   (形状不变，只是被旋转)
```

## ⭐ 必做的 sanity（这是这个组件的「证明机制 work」）

仿照 [`common.py` 的因果性检查](../..)那种「具体不变量」测法，验证**相对位置不变性**：

> 取同一对 `q,k`，分别放在位置 `(m, n)` 和 `(m+s, n+s)`。RoPE 后两组的 attention
> 分数应当**逐元素相等**（因为相对偏移都是 `m−n`）。把 `s` 取几个值断言 `allclose`。

这一条过了，就直接证明了「绝对旋转 = 相对注意力」。`train_sanity.py` 可再加一个需要相对位置
才能解的 toy 任务（如「复制 k 格前的 token」），对比**有 RoPE vs 无任何 PE** 的可学习性。

## ✅ 盲讲检查点

1. RoPE 加在 Q/K/V 的哪几个上？在 attention 的哪一步施加？每层都加还是只加一次？
2. 为什么「绝对地旋转」点积后只剩**相对**位置？（`R(m)ᵀR(n)=R(n−m)`）
3. `θ_i = base^(−2i/d_k)` 里不同维度频率不同意味着什么？（高低频指针，编码远近不同尺度）
4. RoPE 有可学习参数吗？为什么说它「可外推」、长上下文扩展建立在它之上？
5. `rotate_half` 在干嘛？为什么旋转能写成 `x*cos + rotate_half(x)*sin`？

> 下一站任选 [`../rmsnorm`](../rmsnorm) / [`../swiglu`](../swiglu) / [`../gqa`](../gqa)，集齐四块后进 [`../nano`](../nano) 组装。
