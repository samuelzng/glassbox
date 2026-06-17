import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rope.rope import apply_rope

class GQA(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.d_k = d // n_heads
        self.wq = nn.Linear(d, n_heads * self.d_k, bias=False)
        self.wk = nn.Linear(d, n_kv_heads * self.d_k, bias=False)
        self.wv = nn.Linear(d, n_kv_heads * self.d_k, bias=False)
        self.wo = nn.Linear(n_heads * self.d_k, d, bias=False)

    def _qkv(self, x):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.d_k).transpose(1, 2)
        return q, k, v

    def _merge(self, o):
        B, _, L, _ = o.shape
        return self.wo(o.transpose(1, 2).reshape(B, L, self.n_heads * self.d_k))

    def _grouped_attn(self, q, k, v, causal):
        Lq = q.size(-2)
        q = q.unflatten(1, (self.n_kv_heads, self.n_rep))
        k, v = k.unsqueeze(2), v.unsqueeze(2)
        scores = q @ k.transpose(-2, -1) / self.d_k ** 0.5
        if causal:
            mask = torch.triu(torch.ones(Lq, Lq, dtype=torch.bool, device=q.device), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        return (scores.softmax(-1) @ v).flatten(1, 2)

    def _sdpa_attn(self, q, k, v, causal):
        try:
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)
        except TypeError:
            return self._grouped_attn(q, k, v, causal)

    def forward(self, x, cos=None, sin=None, causal=True):
        q, k, v = self._qkv(x)
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        return self._merge(self._sdpa_attn(q, k, v, causal))

    def forward_grouped(self, x, causal=True):
        q, k, v = self._qkv(x)
        return self._merge(self._grouped_attn(q, k, v, causal))


def kv_cache_bytes(n_layers, L, n_kv_heads, d_k, bytes_per=2):
    return 2 * n_layers * L * n_kv_heads * d_k * bytes_per


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, d, H, H_kv = 2, 16, 512, 8, 2
    att = GQA(d, H, H_kv).eval()
    x = torch.randn(B, L, d)

    q, k, v = att._qkv(x)
    print(f"q    {tuple(q.shape)}   H={H} heads")
    print(f"k,v  {tuple(k.shape)}   H_kv={H_kv} heads")
    print(f"out  {tuple(att(x).shape)}")
    assert att(x).shape == x.shape

    # the broadcast path (C) and the fused kernel (D) must agree
    assert torch.allclose(att(x), att.forward_grouped(x), atol=1e-4)
    print("\nforward (SDPA enable_gqa) == forward_grouped (broadcast)")

    # H_kv == H is plain MHA: enable_gqa must match vanilla SDPA
    mha = GQA(d, H, H).eval()
    qh, kh, vh = mha._qkv(x)
    ref = mha._merge(F.scaled_dot_product_attention(qh, kh, vh, is_causal=True))
    assert torch.allclose(mha(x), ref, atol=1e-4)
    print("H_kv = H degenerates to plain MHA")

    # KV-cache at 7B scale: 32 layers, L=8192, d_k=128, bf16; shrinks exactly H/H_kv
    mha_c = kv_cache_bytes(32, 8192, 32, 128)
    print(f"\n{'':6}{'H_kv':>6}{'cache':>12}{'vs MHA':>10}")
    for name, hkv in [("MHA", 32), ("GQA", 8), ("MQA", 1)]:
        c = kv_cache_bytes(32, 8192, hkv, 128)
        print(f"{name:6}{hkv:>6}{c / 2**30:>9.2f} GiB{f'{mha_c // c}x':>9}")
        assert mha_c // c == 32 // hkv
