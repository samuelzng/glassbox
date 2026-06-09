import torch

def build_rope_cache(seq_len, d_k, base=10000.0):

    # [d_k//2]  
    inv_freq = 1 / (base ** (torch.arange(0, d_k, 2).float() / d_k)) 

    # [seq_len]
    positions = torch.arange(0, seq_len).float()

    angles = torch.outer(positions, inv_freq)

    angles = torch.cat([angles, angles], dim=-1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def rotate_half(x):
    # Split Half
    half = x.shape[-1]//2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin

if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, L, d_k = 2, 4, 16, 32
    cos, sin = build_rope_cache(L, d_k)

    x = torch.randn(B, H, L, d_k)
    print(f"shape  {tuple(x.shape)} --apply_rope--> {tuple(apply_rope(x, cos, sin).shape)}")
    assert apply_rope(x, cos, sin).shape == x.shape

    # one (q, k) pair: the score q·k depends only on the offset m−n, not absolute position.
    q, k = torch.randn(d_k), torch.randn(d_k)
    def score(m, n):
        qm = apply_rope(q.view(1, 1, 1, d_k), cos[m:m + 1], sin[m:m + 1])
        kn = apply_rope(k.view(1, 1, 1, d_k), cos[n:n + 1], sin[n:n + 1])
        return (qm * kn).sum().item()

    print(f"\n{'offset':>8}{'s=0':>9}{'s=3':>9}{'s=6':>9}")
    for off in (1, 4, 8):
        row = [score(off + s, s) for s in (0, 3, 6)]          # same offset, shifted by s
        print(f"{off:>8}" + "".join(f"{v:>9.4f}" for v in row))
        assert max(abs(v - row[0]) for v in row) < 1e-5         # constant across the shift