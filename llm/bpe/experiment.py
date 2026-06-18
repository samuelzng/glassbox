"""BPE as an allocation problem — the capstone's thesis, one layer down the stack.

A byte-level BPE has a fixed merge budget. On a mixed English+Chinese corpus, frequency-based
merging hands that budget to whichever script makes the most frequent byte-pairs — English, at
1 byte/char — leaving Chinese (3 bytes/char) under-tokenized. Same shape as the MoE capstone:
naive allocation under-serves the rarer/denser distribution, and fixing it trades against the
dominant one.

Imports the hand-written bpe (single source of truth); runs on the capstone's two corpora.
    python experiment.py        # after llm/nano/prepare.py has fetched the corpora
"""
import pathlib
from collections import Counter

import regex as re
from bpe import train_bpe_fast, encode, build_vocab

DATA = pathlib.Path(__file__).resolve().parents[1] / "nano" / "data"
HAN = re.compile(r"\p{Han}")


def load(name, nbytes):
    return (DATA / f"{name}.txt").read_bytes()[:nbytes].decode("utf-8", "ignore")


def ch_per_tok(text, merges):
    return len(text) / len(encode(text, merges))


def hanzi_composed(text, merges):
    """Fraction of hanzi (types, occurrences) that have a dedicated single token."""
    vocab_bytes = set(build_vocab(merges).values())
    counts = Counter(c for c in text if HAN.match(c))
    composed = [h for h in counts if h.encode("utf-8") in vocab_bytes]
    n_occ = sum(counts.values()) or 1
    return len(composed) / (len(counts) or 1), sum(counts[h] for h in composed) / n_occ


def main():
    en, zh = load("arxiv", 200_000), load("gushi", 200_000)      # equal *byte* budget
    mix = en + "\n" + zh
    print(f"equal byte budget | arxiv {len(en):,} chars | gushi {len(zh):,} chars (3 B/char, denser)")

    shared = {vs: train_bpe_fast(mix, vs) for vs in (512, 1024, 2048)}

    print("\n=== one shared BPE on the mix — who wins the merge budget? ===")
    print(f"  {'vocab':>5} | {'EN ch/tok':>9} | {'ZH ch/tok':>9} | {'ZH hanzi with own token':>24}")
    for vs, m in shared.items():
        t, o = hanzi_composed(zh, m)
        print(f"  {vs:>5} | {ch_per_tok(en, m):>9.2f} | {ch_per_tok(zh, m):>9.2f} | "
              f"{t:>8.1%} types  {o:>5.1%} occ")

    vocab = build_vocab(shared[2048])
    ascii_word = cjk = frag = 0
    for i in range(256, 256 + len(shared[2048])):
        try:
            s = vocab[i].decode("utf-8")
        except UnicodeDecodeError:
            frag += 1
            continue
        if any(HAN.match(ch) for ch in s):
            cjk += 1
        elif len(s) >= 2:
            ascii_word += 1
    print(f"  of {len(shared[2048])} merges @2048:  ascii-word {ascii_word}   cjk {cjk}   "
          f"byte-fragment {frag}")

    print("\n=== the fix: dedicate the budget to Chinese ===")
    for vs in (512, 2048):
        t_shared, _ = hanzi_composed(zh, shared[vs])
        t_zh, _ = hanzi_composed(zh, train_bpe_fast(zh, vs))
        print(f"  vocab {vs:>4}: hanzi with own token   shared {t_shared:>5.1%}   vs   "
              f"Chinese-only {t_zh:>5.1%}")

    print("\nthrough-line: a fixed merge budget is an allocation problem. Frequency hands it to "
          "English and starves Chinese; dedicating budget reverses it at English's cost — the same "
          "balance vs. specialization tension as the MoE capstone, one layer down the stack.")


if __name__ == "__main__":
    main()
