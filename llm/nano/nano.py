import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # put llm/ on the path
import torch
import torch.nn as nn
import torch.nn.functional as F

from rope.rope import build_rope_cache         
from rmsnorm.rmsnorm import RMSNorm            
from swiglu.swiglu import SwiGLU               
from gqa.gqa import GQA                       

class ModernBlock(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, ffn_factory=None):
        super().__init__()
        self.att_norm = RMSNorm(d)
        self.att = GQA(d, n_heads, n_kv_heads)
        self.ffn_norm = RMSNorm(d)
        self.ffn = (ffn_factory or (lambda dim: SwiGLU(dim)))(d)

    def forward(self, x, cos, sin):
        x = x + self.att(self.att_norm(x), cos, sin)
        out = self.ffn(self.ffn_norm(x))
        if isinstance(out, tuple):              # MoE -> (out, aux, idx); SwiGLU -> tensor
            out, self.aux, self.idx = out
        else:
            self.aux, self.idx = None, None
        return x + out
    

class NanoLLM(nn.Module):
    def __init__(self, vocab, d, n_layers, n_heads, n_kv_heads, max_len, ffn_factory=None):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(ModernBlock(d, n_heads, n_kv_heads, ffn_factory) for _ in range(n_layers))
        self.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.lm_head.weight = self.embed.weight   
        nn.init.normal_(self.embed.weight, std=0.02)
        self.max_len = max_len
        cos, sin = build_rope_cache(max_len, d // n_heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, ids):
        L = ids.shape[1]
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x, self.cos[:L], self.sin[:L])
        self.aux = sum((b.aux for b in self.blocks if b.aux is not None), 0)  # scalar; 0 if dense
        self.routing = [b.idx for b in self.blocks if b.idx is not None]      # per-MoE-layer expert ids
        return self.lm_head(self.norm(x))


if __name__ == "__main__":
    torch.manual_seed(0)

    # the assembled model memorizes a description of itself
    text = ("rope rotates the queries and keys. rmsnorm rescales the stream. "
            "swiglu gates the feed forward. gqa shares the kv heads. "
            "stack the blocks and the nano llm predicts the next token. ") * 4
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text])

    B, L = 16, 64
    model = NanoLLM(vocab=len(chars), d=64, n_layers=2, n_heads=4, n_kv_heads=2, max_len=L)
    assert model.lm_head.weight is model.embed.weight            # weight tying: same Parameter
    print(f"vocab {len(chars)}  params {sum(p.numel() for p in model.parameters()):,}")

    ids = data[:8].view(1, -1)
    print(f"shape  ids {tuple(ids.shape)} --NanoLLM--> logits {tuple(model(ids).shape)}")

    # next-token training: every position gets a loss in one forward (teacher forcing)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for step in range(300):
        starts = torch.randint(0, len(data) - L - 1, (B,))
        batch = torch.stack([data[s:s + L + 1] for s in starts])
        logits = model(batch[:, :-1])                            # [B, L, vocab]
        loss = F.cross_entropy(logits.reshape(-1, len(chars)), batch[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 50 == 0 or step == 299:
            print(f"step {step:>4}  loss {loss.item():.3f}")

    first, last = sum(losses[:20]) / 20, sum(losses[-20:]) / 20
    assert last < first * 0.9, (first, last)                     # the pipeline actually learns

    # greedy decode (naive, recompute the whole sequence each step — kvcache optimizes this)
    ids = data[:8].view(1, -1)                                   # prompt: 'rope rot'
    for _ in range(120):
        logits = model(ids[:, -model.max_len:])                  # crop to the rope cache
        ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    print(f"\ngreedy from {text[:8]!r}:\n{''.join(chars[i] for i in ids[0])}")


