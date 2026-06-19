import torch
from torch import nn
from torch.nn import ModuleList

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.network(x)
    
class Attn(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        # assert dim == dim_head * heads
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        
        self.scores = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)        
        b, n, _ = x.shape

        q, k, v = map(lambda t: t.view(b, n, self.heads, -1).transpose(1, 2), qkv)
        score = self.scores(torch.matmul(q, k.transpose(-1, -2)) * self.scale)

        attn = self.dropout(score)

        out = torch.matmul(attn, v) # B, h, L, dim_heads
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return self.out(out)

class Model(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, ffn_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(
                ModuleList([
                    Attn(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(dim, ffn_dim, dropout= dropout)
                ])
            )
        
    def forward(self, x):
        for att, ffn in self.layers:
            x = att(x) + x
            x = ffn(x) + x
        
        return self.norm(x)
        
class Patchify(nn.Module):
    def __init__(self, p1, p2):
        super().__init__()
        self.p1, self.p2 = p1, p2

    def forward(self, img):
        B, C, H, W = img.shape
        p1, p2 = self.p1, self.p2

        assert H % p1 == 0 and W % p2 == 0, f'image {H}*{W} not divisible by patch {p1}*{p2}'

        h, w = H // p1, W // p2

        x = img.view(B, C, h, p1, w, p2)
        x = x.permute(0, 2, 4, 3, 5, 1) # B, h, w, p1, p2, c

        return x.reshape(B, h*w, p1*p2*C)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, 
                 ffn_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size *patch_size

        self.to_patch = nn.Sequential(
            Patchify(patch_size, patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        num_cls_tokens = 1 if pool == 'cls' else 0
        self.cls_token = nn.Parameter(torch.randn(num_cls_tokens, dim))

        self.pos_embedding = nn.Parameter(torch.randn(num_cls_tokens+num_patches, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.model = Model(dim, depth, heads, dim_head, ffn_dim, dropout=dropout)

        self.pool = pool

        # Transferability
        self.to_latent = nn.Identity()

        self.classifier = nn.Linear(dim, num_classes) if num_classes > 0 else None


    def forward(self, img):
        x = self.to_patch(img)      # (B, n, dim)
        b = x.shape[0]

        cls_tokens = self.cls_token.unsqueeze(0).expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim = 1) 
        x = x + self.pos_embedding[: x.shape[1]]
        x = self.dropout(x)

        x = self.model(x)                               

        if self.classifier is None:
            return x                                          

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]  
        return self.classifier(self.to_latent(x))

if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = dict(image_size=32, patch_size=4, dim=128, depth=4,
               heads=4, dim_head=32, ffn_dim=256, pool="cls")
    B = 2
    n_tok = (cfg["image_size"] // cfg["patch_size"]) ** 2          # 64 patches

    def nparams(m): return sum(p.numel() for p in m.parameters())

    net = ViT(num_classes=10, **cfg).eval()
    x = torch.randn(B, 3, cfg["image_size"], cfg["image_size"])

    # 1) contract: an image becomes class logits
    logits = net(x)
    print(f"tokens   {n_tok} patches + 1 cls = {n_tok + 1}")
    print(f"shape    {tuple(x.shape)} --ViT--> {tuple(logits.shape)}    params {nparams(net):,}")
    assert logits.shape == (B, 10)

    # 2) duality: num_classes=0 turns ViT into a vision tower -> visual tokens (the VLM hook)
    tower = ViT(num_classes=0, **cfg).eval()
    tokens = tower(x)
    print(f"encoder  {tuple(x.shape)} --ViT(classes=0)--> {tuple(tokens.shape)}")
    assert tokens.shape == (B, n_tok + 1, cfg["dim"])

    # 3) wiring: one backward must reach EVERY weight -- patchify, cls, pos, blocks, head
    net.zero_grad(); net(x).sum().backward()
    params = list(net.named_parameters())
    dead = [n for n, p in params if p.grad is None or p.grad.abs().sum().item() == 0]
    print(f"grad     {len(params) - len(dead)}/{len(params)} weights receive gradient")
    assert not dead, f"no gradient reached: {dead}"

    # 4) signature: kill the position embedding -> the answer must change (patch order is real)
    with torch.no_grad():
        base = net(x).clone()
        net.pos_embedding.zero_()
        assert not torch.allclose(base, net(x)), "position embedding does nothing?"
    print("pos-emb  zeroing positions changes logits  ->  order matters")


