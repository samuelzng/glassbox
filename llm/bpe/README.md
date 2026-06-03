# llm/bpe — Byte-Pair Encoding 分词器

> **📌 一手出处 / official reference**
> - 论文 *Neural Machine Translation of Rare Words with Subword Units* — Sennrich et al., ACL 2016 · [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
> - 教学级参考实现 [karpathy/minbpe](https://github.com/karpathy/minbpe)

> 学习笔记 + 复现 spec。目标：复现完 [`bpe.py`](bpe.py) 后，能讲清
> **「token 既不是词也不是字符，而是统计出来的高频片段」**，并解释那些奇怪现象（前导空格、数字被拆碎）。

## 一句话 mental model

从**字节/字符**开始，统计语料里**相邻出现最频繁的一对**，把它**合并**成一个新 token，
记下这条合并规则；重复几千几万次，直到词表到目标大小。常见词被合成整块，罕见词留成碎片。
**词表 = 一串有序的合并规则。**

## 🎯 盯死的反直觉机制：分词是「学」出来的，不是规则切的

```
训练（统计合并）:
  语料: "low lower lowest ..."
  初始: l o w   l o w e r   l o w e s t          (全是单字符)
  最频繁对是 (l,o) → 合并成 "lo"
  接着 (lo,w) → "low" ...                         每步并掉当前最高频对
  → 学到一串 merges: [(l,o),(lo,w),...]

编码（应用规则）:  对新文本，按 merges 的顺序反复合并 → token 序列
```

由此解释几个「怪现象」：
- **token 常带前导空格**（如 `" the"`）：空格被并进词里，因为「空格+常见词」高频共现。
- **数字 / 罕见词被拆碎**：它们没高频到能合成整块，只能留成更小的片段。
- **同一个词在不同上下文 token 数可能不同**：取决于周围能触发哪些 merge。
- **byte-level 兜底**：从 256 个字节起步，任何字符（emoji、生僻字）都能表示 → **永不 OOV**。

## === DIFF: 三种分词粒度 ===

| | 按词 (word) | 按字符 (char) | BPE（折中） |
|---|---|---|---|
| 词表 | 巨大，且有 OOV | 极小，无 OOV | 中等，无 OOV（byte 兜底） |
| 序列长度 | 短 | 很长 | 适中 |
| 罕见词 | 整个变 `<unk>` | 逐字符 | 拆成已知片段 |

## 你要复现的目标（接口 + TODO）

文件：`bpe.py`（训练 + 编解码 + 演示）。

```python
def train_bpe(corpus: str, vocab_size: int) -> list[tuple]:
    # 返回有序 merges。循环：统计相邻对频次 → 合并最高频对 → 记录，直到词表达标
def encode(text: str, merges) -> list[int]: ...
def decode(ids: list[int], merges) -> str: ...
```
建议 byte-level 起步（256 基础 token），保证无 OOV。

## ⭐ 必做的 sanity

`bpe.py` 打印/断言：
- **round-trip**：`decode(encode(s)) == s`，对任意输入（含 emoji）都成立；
- 训练后展示**前几条学到的 merge**，和「常见词压成 1~2 个 token、随机串被拆很碎」的对比；
- 词表增长曲线 / 不同文本的压缩率（token 数 / 字符数）。

## ✅ 盲讲检查点

1. 一次 BPE 合并步在干嘛？「训练」和「编码」的区别？
2. 为什么很多 token 带前导空格？为什么数字常被拆碎？
3. byte-level 起步如何保证「永不 OOV」？
4. 按词 / 按字符 / BPE 在词表大小、序列长度、罕见词处理上的权衡？
5. 词表本质上是什么？（一串有序的合并规则）

> LLM 主线收尾。回 [`../`](..) 总览，或转去 [`../../posttrain`](../../posttrain) 用 nano 当底座做后训练。
