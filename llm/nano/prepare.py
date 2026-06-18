"""Fetch the two char-level corpora for the MoE capstone (see experiment.py).

Two deliberately different distributions so the router has something to specialize on:
  - data/arxiv.txt : English ML abstracts (arXiv cs.CV + cs.CL), ASCII-ish, ~1 byte/char
  - data/gushi.txt : Classical Chinese poems (chinese-poetry 全唐诗), CJK, ~3 bytes/char

Stdlib only (no torch). Run once before experiment.py:

    python prepare.py            # fetch both, skip any that already exist
    python prepare.py --force    # refetch

Sizes are byte budgets; because CJK is ~3 bytes/char the two files hold very different
*character* counts — experiment.py mixes them 50/50 by sampling, not by size, so that is fine.
Both endpoints are public + auth-free. data/ lives under llm/nano/ and is committed (<~10MB).
"""
import argparse
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DATA = pathlib.Path(__file__).resolve().parent / "data"
UA = {"User-Agent": "glassbox-prepare/1.0 (educational; from-scratch LLM notes)"}

ARXIV_CATS = "cat:cs.CV+OR+cat:cs.CL"
ARXIV_TARGET_BYTES = 4 * 1024 * 1024          # ~4M English chars
GUSHI_TARGET_BYTES = 6 * 1024 * 1024          # ~2M CJK chars (denser bytes, see note above)

CP_OWNER, CP_REPO, CP_BRANCH, CP_DIR = "chinese-poetry", "chinese-poetry", "master", "全唐诗"


def _get(url, tries=4, pause=3.0):
    """GET with a User-Agent and simple backoff; returns raw bytes."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            if attempt == tries - 1:
                raise
            print(f"  retry {attempt + 1}/{tries} after {e}", file=sys.stderr)
            time.sleep(pause * (attempt + 1))


def fetch_arxiv(target_bytes):
    """Page the arXiv Atom API, pull each entry's <summary> (the abstract)."""
    base = "https://export.arxiv.org/api/query"
    out, start, page = [], 0, 200
    total = 0
    while total < target_bytes:
        url = f"{base}?search_query={ARXIV_CATS}&start={start}&max_results={page}"
        xml = _get(url).decode("utf-8", "replace")
        summaries = re.findall(r"<summary>(.*?)</summary>", xml, re.S)
        if not summaries:                                   # ran out of results
            break
        for s in summaries:
            s = " ".join(s.split())                         # collapse the hard line breaks
            if len(s) > 80:                                 # skip stubs / parse noise
                out.append(s)
                total += len(s.encode("utf-8")) + 2
        print(f"  arxiv start={start:>5}  +{len(summaries)} abstracts  ~{total/1e6:.1f} MB")
        start += page
        time.sleep(3.0)                                     # arXiv asks for >=3s between calls
    return "\n\n".join(out)


def fetch_gushi(target_bytes):
    """List the 全唐诗 dir, keep only poet.tang.*.json (it also holds poet.song.*),
    fetch each from the raw CDN, and flatten poem `paragraphs` to plain text."""
    q = urllib.parse.quote(CP_DIR)                          # 全唐诗 -> %E5%85%A8%E5%94%90%E8%AF%97
    listing = json.loads(_get(
        f"https://api.github.com/repos/{CP_OWNER}/{CP_REPO}/contents/{q}?ref={CP_BRANCH}"
    ))
    files = sorted(e["name"] for e in listing
                   if re.fullmatch(r"poet\.tang\.\d+\.json", e["name"]))
    if not files:
        raise RuntimeError("no poet.tang.*.json found — chinese-poetry layout changed?")

    raw_base = f"https://raw.githubusercontent.com/{CP_OWNER}/{CP_REPO}/{CP_BRANCH}/{q}"
    out, total = [], 0
    for name in files:
        poems = json.loads(_get(f"{raw_base}/{name}"))
        for poem in poems:
            paras = [p.strip() for p in poem.get("paragraphs", []) if p.strip()]
            if not paras or any("□" in p for p in paras):   # drop poems with missing glyphs
                continue
            poem_text = "\n".join(paras)
            out.append(poem_text)
            total += len(poem_text.encode("utf-8")) + 2
        print(f"  gushi {name:>20}  ~{total/1e6:.1f} MB")
        if total >= target_bytes:
            break
        time.sleep(0.1)
    return "\n\n".join(out)


def report(name, path, text):
    chars = sorted(set(text))
    print(f"\n{name}: {len(text):,} chars, {path.stat().st_size/1e6:.2f} MB on disk, "
          f"{len(chars)} unique chars")
    print(f"  sample: {text[:70]!r}")
    return set(chars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="refetch even if files exist")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)

    jobs = [("arxiv.txt", "arXiv abstracts", fetch_arxiv, ARXIV_TARGET_BYTES),
            ("gushi.txt", "全唐诗 poems", fetch_gushi, GUSHI_TARGET_BYTES)]
    vocabs = {}
    for fname, label, fetch, budget in jobs:
        path = DATA / fname
        if path.exists() and not args.force:
            print(f"{label}: {path} exists, skipping (--force to refetch)")
            vocabs[fname] = report(label, path, path.read_text("utf-8"))
            continue
        print(f"fetching {label} -> {path} (target ~{budget/1e6:.0f} MB)")
        text = fetch(budget)
        path.write_text(text, "utf-8")
        vocabs[fname] = report(label, path, text)

    shared = vocabs.get("arxiv.txt", set()) | vocabs.get("gushi.txt", set())
    print(f"\nshared char vocab (both corpora): {len(shared)} tokens "
          f"— dominated by hanzi, so char-level Chinese is the harder side (expected).")
    print("next: `python experiment.py` (after the nano.py ffn_factory edits) once you say 出发.")


if __name__ == "__main__":
    main()
