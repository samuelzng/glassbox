import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "vit"))
from vit import ViT, Model

class CLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, d_v, d_t, d=512):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.image_proj = nn.Linear(d_v, d, bias=False)
        self.text_proj = nn.Linear(d_t, d, bias=False)

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))

    def forward(self, images, texts):
        e_img = F.normalize(self.image_proj(self.image_encoder(images)), dim= -1) #[B, d]
        e_text = F.normalize(self.text_proj(self.text_encoder(texts)), dim=-1)    #[B, d]

        logits = self.logit_scale.exp() * e_img @ e_text.t()

        labels = torch.arange(len(images), device=images.device)


        loss_image = F.cross_entropy(logits, labels)
        loss_text = F.cross_entropy(logits.t(), labels)

        return (loss_image + loss_text)/2


if __name__ == "__main__":
    torch.manual_seed(0)
    DIM = 128
    COLORS = {"red": (.9, .2, .2), "green": (.2, .8, .3), "blue": (.3, .4, .95),
              "yellow": (.9, .85, .2), "purple": (.6, .25, .8), "orange": (.95, .55, .15)}
    SHAPES = ["square", "circle", "triangle"]
    CLASSES = [(c, s) for c in COLORS for s in SHAPES]                  # 18 distinct (color, shape)
    VOCAB = {w: i for i, w in enumerate(sorted({"a", *COLORS, *SHAPES}))}   # word-level lookup, no BPE
    CAP_LEN = 3
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")

    def caption(color, shape):
        return [VOCAB[w] for w in ("a", color, shape)]

    def render(color, shape, r=8):                # a flat colored shape, position-jittered + noisy
        cy, cx = (16 + torch.randint(-2, 3, (2,))).tolist()
        dy, dx = yy - cy, xx - cx
        if   shape == "square":   mask = (dx.abs() <= r) & (dy.abs() <= r)
        elif shape == "circle":   mask = dx**2 + dy**2 <= r*r
        else:                     mask = (dy >= -r) & (dy <= r) & (dx.abs().float() <= (dy + r).clamp(min=0)*.5)
        col = torch.tensor(COLORS[color]).view(3, 1, 1)
        img = torch.where(mask.unsqueeze(0), col, torch.full((3, 32, 32), .08))
        return (img + .03 * torch.randn(3, 32, 32)).clamp(0, 1)

    def batch(n):                                 # n DISTINCT classes -> no duplicate captions, clean diagonal
        idx = torch.randperm(len(CLASSES))[:n]
        imgs = torch.stack([render(*CLASSES[i]) for i in idx])
        caps = torch.tensor([caption(*CLASSES[i]) for i in idx])
        return imgs, caps

    class ImageTower(nn.Module):                  # the hand-written ViT, head dropped (num_classes=0)
        def __init__(self):
            super().__init__()
            self.vit = ViT(num_classes=0, image_size=32, patch_size=4, dim=DIM,
                           depth=4, heads=4, dim_head=32, ffn_dim=4*DIM, pool="cls")
        def forward(self, img):
            return self.vit(img)[:, 0]            # CLS token = pooled image vector

    class TextTower(nn.Module):                   # nn.Embedding + the SAME Transformer block + mean-pool
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(VOCAB), DIM)
            self.pos = nn.Parameter(torch.randn(CAP_LEN, DIM))
            self.model = Model(DIM, depth=2, heads=4, dim_head=32, ffn_dim=4*DIM)
        def forward(self, tok):
            return self.model(self.emb(tok) + self.pos[:tok.shape[1]]).mean(dim=1)

    def make_clip():
        return CLIP(ImageTower(), TextTower(), d_v=DIM, d_t=DIM, d=DIM)

    @torch.no_grad()
    def embed(m, imgs, caps):
        zi = F.normalize(m.image_proj(m.image_encoder(imgs)), dim=-1)
        zt = F.normalize(m.text_proj(m.text_encoder(caps)), dim=-1)
        return zi, zt

    @torch.no_grad()
    def diag_acc(m, n=len(CLASSES), reps=4):       # fraction whose row-argmax is the diagonal
        hit = 0.
        for _ in range(reps):
            zi, zt = embed(m, *batch(n))
            hit += ((zi @ zt.t()).argmax(1) == torch.arange(n)).float().mean().item()
        return hit / reps

    @torch.no_grad()
    def zeroshot_acc(m, trials=12):                # no classifier head: classify fresh images by nearest caption
        caps = torch.tensor([caption(*c) for c in CLASSES])
        _, zt = embed(m, torch.zeros(len(CLASSES), 3, 32, 32), caps)
        hit = tot = 0
        for _ in range(trials):
            zi, _ = embed(m, torch.stack([render(*c) for c in CLASSES]), caps)
            hit += ((zi @ zt.t()).argmax(1) == torch.arange(len(CLASSES))).sum().item(); tot += len(CLASSES)
        return hit / tot

    def train(m, B, steps, lr=3e-3):
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        first = last = None
        for t in range(steps):
            loss = m(*batch(B))
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                m.logit_scale.clamp_(max=math.log(100))    # the one CLIP-specific gotcha
            if t == 0: first = loss.item()
            last = loss.item()
        return first, last

    B = len(CLASSES)
    print(f"world    {len(COLORS)} colors x {len(SHAPES)} shapes = {len(CLASSES)} classes | "
          f"vocab {len(VOCAB)} words | image 32x32, 64 patches")

    # 1) contract: two towers -> unit vectors on a shared sphere; at random init the diagonal is not special
    m = make_clip()
    imgs, caps = batch(B)
    zi, zt = embed(m, imgs, caps)
    assert zi.shape == zt.shape == (B, DIM)
    assert torch.allclose(zi.norm(dim=-1), torch.ones(B), atol=1e-5), "image vectors not unit length"
    assert torch.allclose(zt.norm(dim=-1), torch.ones(B), atol=1e-5), "text vectors not unit length"
    loss0 = m(imgs, caps).item()
    unscaled = F.cross_entropy(zi @ zt.t(), torch.arange(B)).item()     # same logits without the 14.3 scale
    print(f"contract zi,zt unit on R^{DIM} sphere, shape {tuple(zi.shape)}   "
          f"init diag-acc {diag_acc(m, B):.0%} (chance {1/B:.0%})")
    print(f"         init loss  scaled {loss0:.2f}   |   unscaled {unscaled:.2f}  (ln B = {math.log(B):.2f})"
          f"   <- no signal at init")
    assert diag_acc(m, B) < 0.30, "diagonal already aligned before training?"

    # 2) it learns: symmetric InfoNCE pulls matched pairs together, pushes the rest apart
    torch.manual_seed(0); m = make_clip()
    first, last = train(m, B=B, steps=1200)
    acc = diag_acc(m, B)
    print(f"learns   loss {first:.2f} -> {last:.2f}   diagonal-match accuracy {acc:.0%}   logit_scale "
          f"{m.logit_scale.exp().item():.1f}")
    assert last < first - 1.0 and acc > 0.90, "contrastive training did not align the towers"

    # 3) zero-shot: classify a fresh image by ranking text prompts -- the classifier was never trained
    zs = zeroshot_acc(m)
    print(f"zeroshot fresh images vs {len(CLASSES)} class captions, NO classifier head:  top-1 {zs:.0%}")
    assert zs > 0.85, "alignment did not transfer to zero-shot"

    # 4) batch size: more in-batch negatives = stronger contrast, judged by a batch-free metric (zero-shot)
    print("batch    same steps & lr, vary B (=> #negatives):")
    for b in (4, 9, 18):
        torch.manual_seed(0); mb = make_clip()
        train(mb, B=b, steps=600)
        print(f"           B={b:>2}  ({b-1:>2} negatives/anchor)   zero-shot {zeroshot_acc(mb):.0%}")

    print("\nthrough-line: the only label is arange(B) -- the diagonal. Bigger batches hand the loss "
          "more negatives per anchor, so contrast is stronger and alignment is faster; once the two "
          "towers share a sphere, zero-shot classification falls out for free, no classifier trained.")
