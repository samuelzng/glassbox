"""MoE capstone — one NanoLLM, two very different char streams, a router that has to choose.

Reuses the hand-written components as the single source of truth: it imports NanoLLM and
swaps its FFN for the hand-written MoE via an injected factory. Trains char-level on a 50/50
mix of English ML abstracts and 全唐诗 poems, then produces four read-outs:

  1. dual-sample generation         — an abstract-flavoured and a poem-flavoured continuation
  2. P(expert | language) table     — does the router specialize by script without being told?
  3. load-balance ablation (β)      — β=0 collapses to a few experts; β>0 keeps them all alive
  4. through-line                   — router = input-conditioned *adaptive allocation*

Honest scope: MoE here *separates* the two domains (it is not magic fusion); cross-language
generalization lives in the shared embedding / attention / head, not the experts. char-level
Chinese is the weaker side (huge vocab, rare glyphs). The deliverable is the routing behaviour,
not the loss.

RUN ORDER (after you say 出发):
    python prepare.py            # writes data/arxiv.txt, data/gushi.txt
    python experiment.py --profile-only   # time 100 steps, set CFG sizes to your time budget
    python experiment.py         # full run

----------------------------------------------------------------------------------------------
TARGET API — this script assumes two edits you make by hand (your components, not mine):

  llm/moe/moe.py
      MoE(d, n_experts, top_k).forward(x: [B,L,d]) -> (out: [B,L,d],
                                                       aux: scalar tensor,   # raw, =1 at uniform
                                                       idx: [B*L, top_k])    # chosen expert ids
      (exactly the signature in llm/ref/moe.py.)

  llm/nano/nano.py  — make the FFN injectable and thread the MoE side-outputs:
      class ModernBlock(nn.Module):
          def __init__(self, d, n_heads, n_kv_heads, ffn_factory=None):
              ...
              self.ffn = (ffn_factory or (lambda dim: SwiGLU(dim)))(d)
          def forward(self, x, cos, sin):
              x = x + self.att(self.att_norm(x), cos, sin)
              out = self.ffn(self.ffn_norm(x))
              if isinstance(out, tuple):            # MoE -> (out, aux, idx); SwiGLU -> tensor
                  out, self.aux, self.idx = out
              else:
                  self.aux, self.idx = None, None
              return x + out

      class NanoLLM(nn.Module):
          def __init__(self, vocab, d, n_layers, n_heads, n_kv_heads, max_len, ffn_factory=None):
              ...
              self.blocks = nn.ModuleList(
                  ModernBlock(d, n_heads, n_kv_heads, ffn_factory) for _ in range(n_layers))
          def forward(self, ids):
              ...                                   # unchanged body
              self.aux     = sum((b.aux for b in self.blocks if b.aux is not None), 0)
              self.routing = [b.idx for b in self.blocks if b.idx is not None]
              return self.lm_head(self.norm(x))

  Both edits are backward compatible: ffn_factory=None keeps nano.py's own __main__ (SwiGLU,
  aux=0, routing=[]) working unchanged.
----------------------------------------------------------------------------------------------
"""
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # put llm/ on the path
import torch
import torch.nn.functional as F

from nano import NanoLLM         # sibling file in this dir ('nano.nano' would clash with nano.py)
from moe.moe import MoE          # your hand-written MoE (single source; you'll rewrite it later)

DATA = pathlib.Path(__file__).resolve().parent / "data"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

CFG = dict(                      # provisional — calibrate with --profile-only to your time budget
    d=256, n_layers=4, n_heads=8, n_kv_heads=2, max_len=256,
    n_experts=8, top_k=1,                # Switch: one expert per token, every layer
    batch=32, lr=3e-3, steps=3000, beta=0.01, seed=0,
)


# ----- data ------------------------------------------------------------------------------------
def load_corpora():
    paths = {n: DATA / f"{n}.txt" for n in ("arxiv", "gushi")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        sys.exit(f"missing {missing} — run `python prepare.py` first.")
    texts = {n: p.read_text("utf-8") for n, p in paths.items()}
    chars = sorted(set().union(*(set(t) for t in texts.values())))   # shared char vocab
    stoi = {c: i for i, c in enumerate(chars)}
    encode = lambda t: torch.tensor([stoi[c] for c in t], dtype=torch.long)
    data = {n: encode(t) for n, t in texts.items()}
    print(f"vocab {len(chars)} chars | arxiv {len(data['arxiv']):,} | gushi {len(data['gushi']):,}"
          f" | device {DEVICE}")
    return data, chars, stoi


def get_batch(data, B, L):
    ix = torch.randint(len(data) - L - 1, (B,))
    return torch.stack([data[i:i + L + 1] for i in ix]).to(DEVICE)


# ----- model -----------------------------------------------------------------------------------
def build_model(vocab, dense=False):
    ffn = None if dense else (lambda d: MoE(d, CFG["n_experts"], CFG["top_k"]))   # None -> SwiGLU
    return NanoLLM(vocab, CFG["d"], CFG["n_layers"], CFG["n_heads"], CFG["n_kv_heads"],
                   CFG["max_len"], ffn_factory=ffn).to(DEVICE)


def train(data, vocab, beta, steps, tag="", profile=False, ckpt=None, dense=False):
    torch.manual_seed(CFG["seed"])
    model = build_model(vocab, dense)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])
    start = 0
    if ckpt and ckpt.exists():                                  # resume an interrupted run
        st = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start = st["step"]
        print(f"  {tag} resumed @ {start}/{steps}")
        if start >= steps:
            return model
    L, half = CFG["max_len"], CFG["batch"] // 2
    t0 = time.time()
    for step in range(start, steps):
        batch = torch.cat([get_batch(data["arxiv"], half, L),       # 50/50 mix every step
                           get_batch(data["gushi"], half, L)], 0)
        logits = model(batch[:, :-1])
        ce = F.cross_entropy(logits.reshape(-1, vocab), batch[:, 1:].reshape(-1))
        loss = ce + beta * model.aux
        opt.zero_grad(); loss.backward(); opt.step()
        if profile and step == 100:
            ms = (time.time() - t0) / 101 * 1000
            print(f"  [profile] {ms:.0f} ms/step on {DEVICE} -> {CFG['steps']} steps "
                  f"~= {ms * CFG['steps'] / 60000:.1f} min")
            return model
        if ckpt and (step + 1) % 200 == 0:                      # checkpoint so a kill is resumable
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1}, ckpt)
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            aux_v = model.aux.detach().item() if torch.is_tensor(model.aux) else float(model.aux)
            print(f"  {tag} step {step:>4}  ce {ce.item():.3f}  aux {aux_v:.3f}")
    if ckpt:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": steps}, ckpt)
    return model


# ----- read-outs -------------------------------------------------------------------------------
@torch.no_grad()
def expert_usage(model, datas, batches=24):
    """Fraction of routed tokens (summed over all MoE layers) landing on each expert."""
    counts = torch.zeros(CFG["n_experts"])
    for _ in range(batches):
        for d in datas:
            model(get_batch(d, CFG["batch"], CFG["max_len"])[:, :-1])
            for idx in model.routing:
                counts += torch.bincount(idx.reshape(-1).cpu(),
                                         minlength=CFG["n_experts"]).float()
    return counts / counts.sum().clamp(min=1)


@torch.no_grad()
def generate(model, prompt, chars, stoi, n=220, temp=0.8):
    ids = torch.tensor([[stoi[c] for c in prompt if c in stoi]], device=DEVICE)
    for _ in range(n):
        logits = model(ids[:, -CFG["max_len"]:])[:, -1] / temp
        ids = torch.cat([ids, torch.multinomial(logits.softmax(-1), 1)], 1)
    return "".join(chars[i] for i in ids[0].tolist())


def save_heatmap(rows, labels):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    data = torch.stack(rows) * 100
    fig, ax = plt.subplots(figsize=(CFG["n_experts"] - 0.5, 0.55 * len(labels) + 1.2))
    im = ax.imshow(data, aspect="auto", cmap="magma", vmin=0, vmax=40)
    ax.set_xticks(range(CFG["n_experts"])); ax.set_xticklabels([f"e{i}" for i in range(CFG["n_experts"])])
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(CFG["n_experts"]):
            v = float(data[i, j])
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v < 22 else "black", fontsize=8)
    ax.set_xlabel("expert"); ax.set_title("P(expert | language)  routing %")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout(); fig.savefig(DATA / "router_heatmap.png", dpi=130)
    print(f"  saved {DATA / 'router_heatmap.png'}")


def usage_row(tag, u):
    return (f"{tag:>16}" + "".join(f"{p:>6.0%}" for p in u)
            + f"   {u.max():>4.0%}  {int((u > 0.05).sum())}/{CFG['n_experts']}")


def gen_samples(model, chars, stoi):
    print("\n=== dual-sample generation ===")
    print("--- abstract prompt ---"); print(generate(model, "We propose a", chars, stoi))
    print("\n--- 古文 prompt ---");    print(generate(model, "白玉誰家", chars, stoi))


# ----- main ------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-only", action="store_true", help="time 100 steps and exit")
    ap.add_argument("--steps", type=int, default=None, help="override CFG['steps']")
    ap.add_argument("--betas", type=str, default=None, help="comma β list to sweep, e.g. 0,0.001,0.01")
    ap.add_argument("--dense", action="store_true", help="dense SwiGLU baseline (no MoE) — the control")
    args = ap.parse_args()
    data, chars, stoi = load_corpora()
    vocab = len(chars)
    steps = args.steps or CFG["steps"]

    if args.profile_only:
        train(data, vocab, beta=CFG["beta"], steps=120, tag="profile", profile=True)
        return

    if args.dense:                          # control: same FLOPs/token as top-1, ~1/8 the FFN capacity
        ck = pathlib.Path("/tmp/glassbox_ck_dense.pt")
        m = train(data, vocab, beta=0.0, steps=steps, tag="dense", ckpt=ck, dense=True)
        print(f"\ndense SwiGLU baseline | params {sum(p.numel() for p in m.parameters()):,}")
        gen_samples(m, chars, stoi)
        ck.unlink(missing_ok=True)
        return

    betas = [float(b) for b in args.betas.split(",")] if args.betas else [0.0, CFG["beta"]]
    cks = [pathlib.Path(f"/tmp/glassbox_ck_b{b}.pt") for b in betas]
    print(f"\n=== routing P(expert|language) · β sweep {betas} · steps={steps} ===")
    print(f"{'':16}" + "".join(f"e{e:<5}" for e in range(CFG["n_experts"])) + "   max  live")
    rows, labels, last = [], [], None
    for b, ck in zip(betas, cks):
        last = train(data, vocab, beta=b, steps=steps, tag=f"β={b}", ckpt=ck)
        for lang in ("arxiv", "gushi"):
            u = expert_usage(last, [data[lang]]); rows.append(u); labels.append(f"β{b} {lang}")
            print(usage_row(f"β={b} {lang}", u))
    print("  down each β: higher β → lower max-share / more live (balance); specialization softens.")
    print("  across the two langs: the hot experts differ = the router found the boundary itself.")
    save_heatmap(rows, labels)
    gen_samples(last, chars, stoi)
    print("\nthrough-line: top-k routing is input-conditioned adaptive compute — "
          "capacity (×E) decoupled from FLOPs (×k).")
    for ck in cks:                                              # full success -> drop resume state
        ck.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
