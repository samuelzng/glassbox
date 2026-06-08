import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or int (2/3 * 4 * d) # 2 / 3 parameters for each weight matrix, the same as origin
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)  
        self.w_down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def nparams(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    d = 512
    x = torch.randn(2, 5, d)
    ff = SwiGLU(d)
    classic = nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(),
                            nn.Linear(4 * d, d, bias=False))   # d→4d→d

    print(f"{'':12}{'hidden':>8}{'params':>12}")
    for name, m, h in [("classic FFN", classic, 4 * d), ("SwiGLU", ff, ff.w_gate.out_features)]:
        print(f"{name:12}{h:>8}{nparams(m):>12,}")

    print(f"\n{'shape':12}{str(tuple(x.shape)):>14} -> {tuple(ff(x).shape)}")
    assert ff(x).shape == x.shape

