import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "vit"))
from vit import ViT, Model

class SigLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, d_v, d_t, d=512, bias_init=-10., learn_bias=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.image_proj = nn.Linear(d_v, d, bias=False)
        self.text_proj = nn.Linear(d_t, d, bias=False)

        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.)))                
        self.bias = nn.Parameter(torch.tensor(float(bias_init)), requires_grad=learn_bias)  

    def forward(self, images, texts):
        e_img = F.normalize(self.image_proj(self.image_encoder(images)), dim=-1)  #[B, d]
        e_text = F.normalize(self.text_proj(self.text_encoder(texts)), dim=-1)    #[B, d]

        logits = self.logit_scale.exp() * e_img @ e_text.t() + self.bias          #[B, B]  (+ bias)

        n = len(images)
        labels = 2 * torch.eye(n, device=images.device) - 1                       # +1 on the diagonal, -1 everywhere else

        return -F.logsigmoid(labels * logits).sum() / n                          # every pair its own sigmoid, averaged over the B anchors


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

    def make_siglip(bias_init=-10., learn_bias=True):
        return SigLIP(ImageTower(), TextTower(), d_v=DIM, d_t=DIM, d=DIM,
                      bias_init=bias_init, learn_bias=learn_bias)

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

    def train(m, B, steps, lr=3e-3, log=None):     # log: set of steps to snapshot (step, loss, temp, diag-acc)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        first = last = None; hist = []
        for t in range(steps):
            loss = m(*batch(B))
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                m.logit_scale.clamp_(max=math.log(100))    # CLIP's one runaway guard, kept verbatim
            if t == 0: first = loss.item()
            last = loss.item()
            if log is not None and t in log:
                hist.append((t, loss.item(), m.logit_scale.exp().item(), diag_acc(m, B)))
        return first, last, hist

    B = len(CLASSES)
    print(f"world    {len(COLORS)} colors x {len(SHAPES)} shapes = {len(CLASSES)} classes | "
          f"vocab {len(VOCAB)} words | image 32x32, 64 patches")

    # 1) the bias is load-bearing: b=-10 starts the model at "nothing matches", so only the B
    #    positives carry loss at init; with b=0 the B^2-B negatives flood in (loss ~ ln2 * B).
    torch.manual_seed(0); m = make_siglip(bias_init=-10.)
    imgs, caps = batch(B); zi, zt = embed(m, imgs, caps)
    assert zi.shape == zt.shape == (B, DIM)
    assert torch.allclose(zi.norm(dim=-1), torch.ones(B), atol=1e-5), "image vectors not unit length"
    with torch.no_grad():
        p_init = (m.logit_scale.exp() * zi @ zt.t() + m.bias).sigmoid().mean().item()
    loss_b10 = m(imgs, caps).item()
    torch.manual_seed(0); loss_b0 = make_siglip(bias_init=0.)(imgs, caps).item()   # same towers, only b differs
    print(f"bias     init mean P(match) {p_init:.4f}  = sigmoid(b), ~0 -> 'nothing matches' at init")
    print(f"         init loss  b=-10 {loss_b10:5.2f} (~|b|, only {B} positives speak)   |   "
          f"b=0 {loss_b0:5.2f} (>= ln2*B = {math.log(2)*B:.1f} floor, all {B*B-B} negatives flood in)")
    assert abs(loss_b10 - 10.) < 1.5 and loss_b0 > loss_b10, "b=-10 should isolate the positives at init"

    # 2) it learns: the per-pair sigmoid pulls matched pairs up and pushes the rest down -- no
    #    softmax, no batch-coupled denominator -- yet diagonal alignment + zero-shot fall out as in CLIP.
    torch.manual_seed(0); m = make_siglip(bias_init=-10.)
    snaps = {0, 5, 50, 300, 1199}
    first, last, hist = train(m, B=B, steps=1200, log=snaps)
    acc = diag_acc(m, B)
    print(f"learns   loss {first:.2f} -> {last:.2f}   diagonal-match {acc:.0%}   "
          f"temp {m.logit_scale.exp().item():.1f}   bias {m.bias.item():.1f}")
    assert last < first - 1.0 and acc > 0.90, "sigmoid contrastive did not align the towers"
    zs = zeroshot_acc(m)
    print(f"zeroshot fresh images vs {len(CLASSES)} captions, NO classifier head:  top-1 {zs:.0%}")
    assert zs > 0.85, "alignment did not transfer to zero-shot"

    # 3) temperature is a learned confidence dial.  Its gradient ~ sim, so it sits still while the
    #    embeddings are random, then climbs once they separate (raising t sharpens a now-correct margin).
    print("temp     step ->  temperature  (flat at init, climbs as the towers separate)")
    for (t, l, temp, a) in sorted(hist):
        print(f"           {t:>4}      {temp:6.1f}      diag {a:.0%}")
    assert hist[-1][2] > hist[0][2] + 2.0, "temperature did not climb during training"
    assert abs(hist[0][2] - 10.) < 3.0, "temperature should start near its init (10)"

    # 4) negative drowning, isolated: FREEZE the bias so it can't self-heal.  b=-10 starts the
    #    negatives already satisfied (clean signal); b=0 lets the B^2-B negatives dominate the early
    #    gradient, so alignment starts slower.  (a LEARNABLE bias mostly closes the gap -- it learns
    #    to go negative on its own -- which is why this is a head-start, widening at SigLIP's real 32k batches.)
    print("drown    frozen bias, B=6, equal steps & lr -- the negative flood made visible:")
    for binit in (-10., 0.):
        torch.manual_seed(0); mb = make_siglip(bias_init=binit, learn_bias=False)
        train(mb, B=6, steps=300)
        print(f"           b={binit:>5.0f} (frozen)   diagonal-match {diag_acc(mb, 6):.0%}")

    print("\nthrough-line: only the loss changed -- arange + softmax-CE became a per-pair sigmoid on "
          "2*eye-1. bias=-10 hands the model the right prior (most pairs DON'T match) so the B "
          "positives aren't drowned; temperature is a confidence dial that climbs only as fast as the "
          "embeddings earn it. same towers, same toy world as clip.py -- the seam is the objective alone.")
