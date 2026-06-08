import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.para = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)) 
    
    def forward(self, x):
        return self.para * self._norm(x.float()).type_as(x) # bf16 - fp32 - bf16


def stats(y):
    return y.mean(-1).abs().mean(), y.pow(2).mean(-1).sqrt().mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    ln, rms = nn.LayerNorm(8), RMSNorm(8)

    print(f"{'':10}{'row mean':>10}{'row rms':>10}")
    for name, f in [("LayerNorm", ln), ("RMSNorm", rms)]:
        m, r = stats(f(x))
        print(f"{name:10}{m:>10.3f}{r:>10.3f}")

    print(f"\n{'shift +3':10}{'max|Δ|':>10}")
    for name, f in [("LayerNorm", ln), ("RMSNorm", rms)]:
        print(f"{name:10}{(f(x + 3) - f(x)).abs().max():>10.3f}")
    
    
