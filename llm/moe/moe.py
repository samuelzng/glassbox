import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    # the expert is just a dense FFN (see ../swiglu)
    def __init__(self, d, hidden=None):
        super().__init__()
        hidden = hidden or int(2 / 3 * 4 * d)
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoE(nn.Module):
    def __init__(self, d, n_experts, top_k, hidden=None):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.router = nn.Linear(d, n_experts, bias=False)        # tiny vs the experts
        self.experts = nn.ModuleList(SwiGLU(d, hidden) for _ in range(n_experts))

    def forward(self, x):                                        # [B, L, d] -> [B, L, d]
        shape = x.shape
        x = x.reshape(-1, shape[-1])                             # [N, d] tokens
        gate = self.router(x).softmax(-1)                        # [N, E]
        weight, idx = gate.topk(self.top_k, dim=-1)              # [N, k] each
        if self.top_k > 1:                                       # renormalize the top-k weights
            weight = weight / weight.sum(-1, keepdim=True)        # (Mixtral); top-1 keeps raw gate
        #                                                          prob, else router loses its grad

        out = torch.zeros_like(x)
        for e in range(self.n_experts):                          # token never touches the others
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel():
                out[tok] += weight[tok, slot, None] * self.experts[e](x[tok])

        # load-balance aux loss (Switch-style): E * Σ_e f_e·P_e, =1 when uniform
        f = F.one_hot(idx, self.n_experts).float().mean((0, 1))  # fraction of slots routed to e
        P = gate.mean(0)                                         # mean router prob for e
        aux = self.n_experts * (f * P).sum()
        return out.reshape(shape), aux, idx


def usage(idx, n_experts):
    return F.one_hot(idx, n_experts).float().mean((0, 1))


if __name__ == "__main__":
    torch.manual_seed(0)
    d, E, k = 64, 8, 2
    moe = MoE(d, E, k)
    x = torch.randn(4, 16, d)
    out, aux, idx = moe(x)

    print(f"shape  {tuple(x.shape)} --MoE--> {tuple(out.shape)}")
    assert out.shape == x.shape
    assert idx.shape == (4 * 16, k)                          # every token: exactly k experts
    assert (idx.sort(-1).values.diff(dim=-1) != 0).all()     # ...and they are distinct

    dense = SwiGLU(d)
    nparams = lambda m: sum(p.numel() for p in m.parameters())
    print(f"\n{'':14}{'params':>10}{'FFNs/token':>12}")
    print(f"{'dense SwiGLU':14}{nparams(dense):>10,}{1:>12}")
    print(f"{'MoE E=8 k=2':14}{nparams(moe):>10,}{k:>12}     # params x{nparams(moe) / nparams(dense):.1f}, FLOPs x{k}")

    # router collapse: same toy task with and without the aux loss. Inputs share a
    # common component (+1), so without pressure all tokens rush to the same experts.
    results = {}
    for aux_coef in (0.0, 0.01):
        torch.manual_seed(0)
        moe = MoE(d, E, k)
        target = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, d))
        opt = torch.optim.AdamW(moe.parameters(), lr=1e-2)
        for step in range(300):
            x = torch.randn(8, 16, d) + 1.0
            out, aux, idx = moe(x)
            loss = F.mse_loss(out, target(x).detach()) + aux_coef * aux
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            _, _, idx = moe(torch.randn(256, 16, d) + 1.0)   # big fresh batch
        results[aux_coef] = usage(idx, E)

    print(f"\nexpert usage after 300 steps (uniform = {1 / E:.1%})")
    print(f"{'expert':>22}" + "".join(f"{e:>7}" for e in range(E)))
    for coef, u in results.items():
        name = "no aux loss" if coef == 0 else f"aux loss x{coef}"
        print(f"{name:>22}" + "".join(f"{p:>7.1%}" for p in u))
    assert results[0.0].max() > 0.3                          # collapse: winners take (almost) all
    assert results[0.0].min() < 0.01                         # ...and losers starve
    assert results[0.01].min() > 0.5 / E                     # aux loss: nobody starves
