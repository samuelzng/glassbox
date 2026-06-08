# llm/bpe — Byte-Pair Encoding tokenizer

A byte-level BPE tokenizer (~180-line core + showcase), trained on the Taylor Swift Wikipedia article
in **two languages** to show how byte-level BPE behaves on Latin vs CJK script. Two
training paths (naive + a heap-based fast one) produce **identical** merges, full
`encode`/`decode` round-trip, and a vocabulary you can read line by line.

> **References** — Sennrich et al., *Neural Machine Translation of Rare Words with
> Subword Units*, ACL 2016 ([arXiv:1508.07909](https://arxiv.org/abs/1508.07909)) ·
> teaching implementation [karpathy/minbpe](https://github.com/karpathy/minbpe).

## Results (`python bpe.py`)

The same Taylor Swift article in both languages, ~60 KB each, trained to vocab 512:

| | English | Chinese 中文 |
|---|---:|---:|
| corpus | 60,853 chars / 60,940 bytes | 22,146 chars / 60,230 bytes |
| bytes / char | 1.00 | 2.72 |
| tokens @ vocab 512 | 28,386 | 29,449 |
| compression (bytes/token) | **2.15×** | **2.05×** |
| chars / token | 2.14 | 0.75 |
| round-trip `decode(encode(x)) == x` | ✓ | ✓ |
| `train_bpe_fast` == `train_bpe` | ✓ | ✓ |

Same byte budget, same vocab — byte-level compression looks similar, but character-level efficiency diverges sharply. 

**In other words, Chinese pays an early character-assembly cost: many merges are spent rebuilding UTF-8 bytes into characters. After that threshold, its short, repetitive words compress efficiently — enough to overtake English in bytes/token. (→ [English vs Chinese](#english-vs-chinese))**

Round-trip is also exact on emoji, mixed-script, and empty inputs (the `sanity()` check).

**The fast path matters more as the vocabulary grows** (English, naive vs fast):

| vocab | tokens | bytes/token | naive | fast | speedup |
|------:|-------:|------------:|------:|-----:|--------:|
| 512   | 28,386 | 2.15× | 1.42s | 0.09s | **16×** |
| 1024  | 21,231 | 2.87× | 4.01s | 0.11s | **37×** |
| 2048  | 16,926 | 3.60× | 7.63s | 0.12s | **62×** |

The naive cost grows with every merge; the fast path stays flat, so its advantage
widens with both vocab size and corpus length.

## What it learns

Merges are learned **in frequency order** — the earliest are the most common byte
pairs, and whole words assemble out of them later:

```
rank   0  256  ' a'      rank   6  262  ' the'  = ' t' + 'he'
rank   1  257  'he'      rank  13  269  ' and'  = ' a' + 'nd'
rank   2  258  ' t'      ...
rank   3  259  'in'      longest tokens by the end:
rank   4  260  'on'        ' Billboard'  ' country'  ' Swift'  ' album'
rank   5  261  'er'        ' Awards'  ' albums'  ' music'  ' artist'
```

Common strings collapse to a few tokens; rare ones stay in pieces; numbers never
clump; spaces fuse onto the following word:

```
'Taylor Swift'  ->  ['T', 'aylor', ' Swift']     a frequent name
' the album'    ->  [' the', ' album']           space merges INTO the word
're-recording'  ->  ['re', '-', 're', 'cord', 'ing']
'1989'          ->  ['19', '8', '9']             digits never form long tokens
'qwxz'          ->  ['q', 'w', 'x', 'z']         rare run -> raw bytes, never OOV
```

## English vs Chinese

Same article, same ~60 KB. The Chinese corpus
([`data/taylorswift_zh.txt`](data/taylorswift_zh.txt)) is the same page from
[zh.wikipedia](https://zh.wikipedia.org/wiki/泰勒·斯威夫特), almost identical in bytes
(60,230 vs 60,940). But English is ASCII (**1.00 bytes/char**) while Chinese is CJK
(**2.72 bytes/char**) — and byte-level BPE sees only raw bytes.

| vocab | EN tokens | EN B/tok | EN ch/tok | ZH tokens | ZH B/tok | ZH ch/tok |
|------:|----------:|---------:|----------:|----------:|---------:|----------:|
| 512   | 28,386 | 2.15 | **2.14** | 29,449 | 2.05 | **0.75** |
| 1024  | 21,231 | 2.87 | 2.87 | 20,661 | 2.92 | 1.07 |
| 2048  | 16,926 | 3.60 | 3.60 | 15,140 | **3.98** | 1.46 |

`bytes/token` is nearly identical — byte-level BPE is **script-agnostic on the byte
axis**. The `chars/token` column exposes the hidden cost. Where do the 256 merges @512 go?

| | multi-char words | single chars | byte fragments |
|---|---:|---:|---:|
| **English** | 255 | 0 | 1 |
| **Chinese** | 29 | 118 | 109 |

```
first merges, EN:  ' a'  'he'  ' t'  'in'  'on'  'er'  ' the'  'ed'   -> straight to words
first merges, ZH:  '�'  '�'  '�'  '�'  '，'  '�'  '的'  '�'           -> bytes into characters
```

- **English** is already one byte per char, so all 256 merges go straight to
  words/subwords → **2.14 chars/token** immediately.
- **Chinese** must first rebuild each character from its 3 bytes: **227 of 256 merges**
  @512 are byte→char assembly (109 still mid-character `�`, 118 complete single chars),
  leaving only **29** for real words (`斯威夫特` Taylor Swift, `公告牌` Billboard,
  `专辑` album) — hence **< 1 char/token** at 512.
- But Chinese words are short and repetitive, so once characters exist it catches up
  fast and **overtakes English on bytes/token by vocab 2048** (3.98 vs 3.60).

Takeaway: byte-level BPE pays a **character-assembly tax** on non-Latin scripts out of
its early vocab budget — which is why multilingual production tokenizers use much
larger vocabularies.

## Files

| file | what |
|---|---|
| [`bpe.py`](bpe.py) | tokenizer + training showcase (run it directly) |
| [`data/`](data) | `taylorswift{,_zh}.txt` corpora (~60 KB each) + generated `.vocab` / `.model` |

```bash
python bpe.py    # sanity, train EN + ZH, write data/*.vocab / *.model, print the comparison
```

`data/` holds both the input corpora and the generated artifacts: `*.model` (the merges,
reloadable) and `*.vocab` (every learned token in merge order, human-readable).

## API

```python
merges = train_bpe(corpus, vocab_size)       # ordered dict[(int,int) -> int]
merges = train_bpe_fast(corpus, vocab_size)  # identical result, much faster
ids    = encode(text, merges)                # -> list[int]
text   = decode(ids, merges)                 # -> str   (round-trips)
```

Ties are broken explicitly — highest count, then **lexicographically smallest pair**
(not dict insertion order). That determinism is what lets the heap-based
`train_bpe_fast` reproduce the naive merges exactly.

## naive vs fast

`train_bpe` rescans every chunk each round — `O(merges · N)`. `train_bpe_fast` keeps a
doubly-linked list over the bytes plus a lazy-deletion max-heap and only touches the
pairs around each merged position, giving the 16–62× speedups above on identical output.
