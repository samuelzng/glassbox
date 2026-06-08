"""Byte-level BPE tokenizer (minbpe-style).

Byte-level base (256 tokens) so it is never OOV; a regex pre-split keeps merges
inside each chunk. train_bpe and train_bpe_fast produce identical merges; encode
and decode round-trip any string.

"""

import heapq
import regex as re

# GPT-4 (cl100k_base) pre-split pattern. BPE only merges inside a chunk, never across
# one -- the real source of leading-space tokens (" the") and digits capped at 3.
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


def get_stats(ids: list[int], counts: dict | None = None) -> dict[tuple[int, int], int]:
    """Count adjacent pairs in ids. Pass counts to accumulate across chunks."""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """Replace every adjacent occurrence of pair in ids with the single token idx."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(corpus: str, vocab_size: int) -> dict[tuple[int, int], int]:
    """Naive BPE: each round merge the most frequent pair (ties -> smallest pair)."""
    assert vocab_size >= 256
    num_merges = vocab_size - 256

    chunks = re.findall(SPLIT_PATTERN, corpus)
    ids = [list(ch.encode("utf-8")) for ch in chunks]

    merges = {}
    for k in range(num_merges):
        stats = {}
        for chunk in ids:
            get_stats(chunk, stats)
        if not stats:
            break
        pair = min(stats, key=lambda p: (-stats[p], p))
        idx = 256 + k
        ids = [merge(chunk, pair, idx) for chunk in ids]
        merges[pair] = idx
    return merges


def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    """Encode text to ids; inside each chunk repeatedly merge the lowest-rank pair."""
    out = []
    for ch in re.findall(SPLIT_PATTERN, text):
        ids = list(ch.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            ids = merge(ids, pair, merges[pair])
        out.extend(ids)
    return out


def build_vocab(merges: dict[tuple[int, int], int]) -> dict[int, bytes]:
    """Map every token id to its byte string: 256 base bytes + each learned merge."""
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), idx in merges.items():
        vocab[idx] = vocab[a] + vocab[b]
    return vocab


def decode(ids: list[int], merges: dict[tuple[int, int], int]) -> str:
    """Join each token's bytes, then decode (replace on partial utf-8)."""
    vocab = build_vocab(merges)
    data = b"".join(vocab[i] for i in ids)
    return data.decode("utf-8", errors="replace")


def train_bpe_fast(corpus: str, vocab_size: int) -> dict[tuple[int, int], int]:
    """Incremental BPE: doubly-linked list + lazy-deletion heap. Same merges as train_bpe.

    Instead of rescanning the whole corpus each merge, only the pairs around each merged
    position are updated. Chunk ends keep -1 links so merges never cross a chunk.
    """
    assert vocab_size >= 256
    num_merges = vocab_size - 256

    sym, prev, nxt = [], [], []
    for ch in re.findall(SPLIT_PATTERN, corpus):
        b = ch.encode("utf-8")
        base, m = len(sym), len(b)
        for k, byte in enumerate(b):
            sym.append(byte)
            prev.append(base + k - 1 if k > 0 else -1)
            nxt.append(base + k + 1 if k < m - 1 else -1)

    pair_counts: dict[tuple[int, int], int] = {}
    pair_positions: dict[tuple[int, int], set] = {}
    heap: list[tuple[int, tuple[int, int]]] = []

    def add_pair(i):
        j = nxt[i]
        if j == -1:
            return
        p = (sym[i], sym[j])
        pair_counts[p] = pair_counts.get(p, 0) + 1
        pair_positions.setdefault(p, set()).add(i)
        heapq.heappush(heap, (-pair_counts[p], p))

    def remove_pair(i):
        j = nxt[i]
        if j == -1:
            return
        p = (sym[i], sym[j])
        pair_counts[p] -= 1
        pair_positions[p].discard(i)
        if pair_counts[p] > 0:
            heapq.heappush(heap, (-pair_counts[p], p))

    for i in range(len(sym)):
        add_pair(i)

    merges = {}
    for k in range(num_merges):
        pair = None
        while heap:
            negc, p = heapq.heappop(heap)
            if pair_counts.get(p, 0) == -negc:  # lazy deletion: skip stale entries
                pair = p
                break
        if pair is None:
            break

        a, b = pair
        idx = 256 + k
        merges[pair] = idx

        for i in sorted(pair_positions[pair]):  # ascending order matches naive on AAAA
            j = nxt[i]
            if j == -1 or sym[i] != a or sym[j] != b:  # consumed by an overlap, skip
                continue
            h, k2 = prev[i], nxt[j]
            if h != -1:
                remove_pair(h)
            if k2 != -1:
                remove_pair(j)
            sym[i] = idx
            nxt[i] = k2
            if k2 != -1:
                prev[k2] = i
            sym[j] = -1
            prev[j] = nxt[j] = -1
            if h != -1:
                add_pair(h)
            if k2 != -1:
                add_pair(i)

        pair_counts[pair] = 0
        pair_positions[pair] = set()

    return merges


def render(token: bytes) -> str:
    """Token bytes -> visible text; a partial utf-8 fragment shows as the replace char."""
    return token.decode("utf-8", errors="replace")


# ══════════════════════════════════════════════════════════════════════════
# showcase: train on data/*.txt, dump artifacts, print an EN-vs-ZH comparison
# ══════════════════════════════════════════════════════════════════════════

import os

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def save(stem: str, merges: dict[tuple[int, int], int]) -> None:
    """Write data/<stem>.model (reloadable) and data/<stem>.vocab (human-readable)."""
    vocab = build_vocab(merges)
    with open(os.path.join(DATA, stem + ".model"), "w", encoding="utf-8") as f:
        f.write("# bpe merges in order; line N (0-based) defines token 256+N\n")
        for a, b in merges:
            f.write(f"{a} {b}\n")
    with open(os.path.join(DATA, stem + ".vocab"), "w", encoding="utf-8") as f:
        f.write(f"# {len(merges)} learned merges, in order. token = left + right\n")
        for rank, ((a, b), idx) in enumerate(merges.items()):
            f.write(f"  {rank:>4}  {idx:>4}  {render(vocab[idx])!r:<16}"
                    f"  =  {render(vocab[a])!r} + {render(vocab[b])!r}\n")


def token_kinds(merges: dict[tuple[int, int], int]) -> tuple[int, int, int]:
    """Count learned tokens as (multi-char words, single chars, byte fragments)."""
    vocab = build_vocab(merges)
    word = single = frag = 0
    for i in range(256, 256 + len(merges)):
        try:
            length = len(vocab[i].decode("utf-8"))
        except UnicodeDecodeError:
            frag += 1
        else:
            word += length >= 2
            single += length == 1
    return word, single, frag


def sanity() -> None:
    """Round-trip on tricky inputs (emoji / CJK / empty) and fast == naive."""
    toy = "low lower lowest newer wider 🤗 中文 123456"
    merges = train_bpe(toy, 300)
    for s in [toy, "hello world", "🤗🤗🤗", ""]:
        assert decode(encode(s, merges), merges) == s, f"round-trip failed: {s!r}"
    assert train_bpe_fast(toy, 300) == merges, "fast != naive"


def report(stem: str, label: str, check_naive: bool = False) -> tuple[str, dict]:
    """Train one corpus at vocab 512, dump artifacts, print a summary; return (text, merges)."""
    import time

    text = open(os.path.join(DATA, stem + ".txt"), encoding="utf-8").read()
    nbytes = len(text.encode("utf-8"))

    t0 = time.time()
    merges = train_bpe_fast(text, 512)
    t_fast = time.time() - t0
    if check_naive:
        t0 = time.time()
        assert train_bpe(text, 512) == merges, "fast != naive"
        t_naive = time.time() - t0
    assert decode(encode(text, merges), merges) == text, "round-trip failed on corpus"
    save(stem, merges)

    vocab = build_vocab(merges)
    ids = encode(text, merges)
    longest = sorted((vocab[i] for i in range(256, 256 + len(merges))), key=len, reverse=True)

    print(f"{label}   {len(text)} chars / {nbytes} bytes  ({nbytes / len(text):.2f} bytes/char)")
    print(f"  vocab 512 -> {len(ids)} tokens, {nbytes / len(ids):.2f}x compression")
    if check_naive:
        print(f"  train  naive {t_naive:.2f}s | fast {t_fast:.2f}s | {t_naive / t_fast:.0f}x speedup")
    print("  first merges :", "  ".join(repr(render(vocab[i])) for i in range(256, 266)))
    print("  longest      :", "  ".join(repr(render(t)) for t in longest[:8]))
    return text, merges


def examples(merges: dict[tuple[int, int], int]) -> None:
    """Show how a few strings tokenize: common stays whole, rare splits apart."""
    vocab = build_vocab(merges)
    for s in ["Taylor Swift", " the album", "1989", "re-recording", "qwxz"]:
        ids = encode(s, merges)
        print(f"    {s!r:>14} -> {[render(vocab[i]) for i in ids]}")


def compare(en_text: str, en_merges: dict, zh_text: str, zh_merges: dict) -> None:
    """Print the EN-vs-ZH scaling table and where each language spends its merges."""
    en_bytes, zh_bytes = len(en_text.encode("utf-8")), len(zh_text.encode("utf-8"))

    print("\nEnglish vs Chinese  (same ~60 KB byte budget):")
    print(f"  {'vocab':>5} | {'EN tok':>7} {'B/tok':>5} {'ch/tok':>6}"
          f" | {'ZH tok':>7} {'B/tok':>5} {'ch/tok':>6}")
    for vs in [512, 1024, 2048]:
        ei = encode(en_text, train_bpe_fast(en_text, vs))
        zi = encode(zh_text, train_bpe_fast(zh_text, vs))
        print(f"  {vs:>5} | {len(ei):>7} {en_bytes / len(ei):>5.2f} {len(en_text) / len(ei):>6.2f}"
              f" | {len(zi):>7} {zh_bytes / len(zi):>5.2f} {len(zh_text) / len(zi):>6.2f}")

    ew, es, ef = token_kinds(en_merges)
    zw, zs, zf = token_kinds(zh_merges)
    print(f"  of 256 merges @512:  EN words={ew} chars={es} fragments={ef}"
          f"  |  ZH words={zw} chars={zs} fragments={zf}")


if __name__ == "__main__":
    sanity()
    en_text, en_merges = report("taylorswift", "English", check_naive=True)
    examples(en_merges)
    print()
    zh_text, zh_merges = report("taylorswift_zh", "Chinese")
    compare(en_text, en_merges, zh_text, zh_merges)
