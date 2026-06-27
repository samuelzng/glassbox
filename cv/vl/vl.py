import torch
import torch.nn as nn
from config import VLMConfig

# --- below: backbones for the VisionLanguageModel (yours to write) ---
import sys, pathlib
import torch.nn.functional as F
ROOT = pathlib.Path(__file__).resolve().parents[2]   # glassbox/
sys.path.insert(0, str(ROOT / "cv" / "vit"))         # ViT
sys.path.insert(0, str(ROOT / "llm" / "nano"))       # NanoLLM (self-bootstraps rope/rmsnorm/gqa/swiglu)
from vit import ViT
from nano import NanoLLM

class ModalityProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.vit_hidden_dim * (cfg.mp_pixel_shuffle_factor**2)
        self.output_dim = cfg.lm_hidden_dim

        self.scale_factor = cfg.mp_pixel_shuffle_factor

        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim, bias=False)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def pixel_shuffle(self, x):
        b, seq, dim = x.size()
        seq_root = int(seq**0.5)
        assert seq_root**2 == seq
        assert seq_root % self.scale_factor == 0

        h = w = seq_root

        x = x.view(b, h, w, dim)
        h_out = h // self.scale_factor
        w_out = w // self.scale_factor

        x = x.reshape(b, h_out, self.scale_factor, w_out, self.scale_factor, dim)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(b, h_out*w_out, dim * self.scale_factor**2)

        return x
    
    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.proj(x)
        return x
    
class VLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.vision_encoder = ViT(
            num_classes=0,
            pool='cls' if cfg.vit_cls_flag else 'mean',
            image_size=cfg.vit_img_size,
            patch_size=cfg.vit_patch_size,
            dim=cfg.vit_hidden_dim,
            depth=cfg.vit_n_blocks,
            heads=cfg.vit_n_heads,
            dim_head=cfg.vit_hidden_dim // cfg.vit_n_heads,
            ffn_dim=cfg.vit_inter_dim,
        )

        self.MP = ModalityProjector(cfg)

        self.decoder = NanoLLM(
            vocab=cfg.lm_vocab_size,
            d=cfg.lm_hidden_dim,
            n_layers=cfg.lm_n_blocks,
            n_heads=cfg.lm_n_heads,
            n_kv_heads=cfg.lm_n_kv_heads,
            max_len=cfg.lm_max_position_embeddings,
        )

        self.image_token_id = cfg.image_token_id

    def _splice(self, input_ids, text_embd, image_embd):
        out = text_embd.clone()
        mask = (input_ids == self.image_token_id)             
        flat = image_embd.reshape(-1, image_embd.size(-1))    
        
        assert mask.sum() == flat.size(0), \
            f"placeholders {int(mask.sum())} != visual tokens {flat.size(0)}"
        
        out[mask] = flat.to(out.dtype)
        return out

    def forward(self, input_ids, images=None, targets=None):
        text_embd = self.decoder.embed(input_ids)              # [B, T, D]
        if images is not None:
            patches = self.vision_encoder(images)              # [N_img, n_patch, d_vit]
            image_embd = self.MP(patches)                      # [N_img, L_img, D]
            text_embd = self._splice(input_ids, text_embd, image_embd)
        logits = self.decoder(inputs_embeds=text_embd)         # [B, T, vocab]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
        return logits, loss


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # a tiny world: one colored shape on a near-black field. the VLM must NAME the color --
    # which only the spliced visual tokens carry. shape varies as a distractor (target is color).
    COLORS = {"red": (.9, .2, .2), "green": (.2, .8, .3), "blue": (.3, .4, .95),
              "yellow": (.9, .85, .2), "purple": (.6, .25, .8), "orange": (.95, .55, .15)}
    SHAPES = ["square", "circle", "triangle"]
    COLOR_LIST = list(COLORS)
    QUERY_ID, IMAGE_ID, N_COLOR = 6, 7, len(COLORS)          # query marker, <image> placeholder, 6 colors = ids 0..5
    VOCAB = 8
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")

    def render(color, shape, r=8):
        cy, cx = (16 + torch.randint(-2, 3, (2,))).tolist()
        dy, dx = yy - cy, xx - cx
        if   shape == "square":   m = (dx.abs() <= r) & (dy.abs() <= r)
        elif shape == "circle":   m = dx**2 + dy**2 <= r*r
        else:                     m = (dy >= -r) & (dy <= r) & (dx.abs().float() <= (dy + r).clamp(min=0) * .5)
        col = torch.tensor(COLORS[color]).view(3, 1, 1)
        img = torch.where(m.unsqueeze(0), col, torch.full((3, 32, 32), .08))
        return (img + .03 * torch.randn(3, 32, 32)).clamp(0, 1)

    def batch(n, L_img):                                     # sequence = [<image> x L_img, "color?"], answer at the last slot
        ci = torch.randint(0, N_COLOR, (n,))
        si = torch.randint(0, len(SHAPES), (n,))
        imgs = torch.stack([render(COLOR_LIST[c], SHAPES[s]) for c, s in zip(ci.tolist(), si.tolist())])
        ids  = torch.cat([torch.full((n, L_img), IMAGE_ID), torch.full((n, 1), QUERY_ID)], dim=1)
        tgt  = torch.full((n, L_img + 1), -100)
        tgt[:, -1] = ci                                      # supervise ONLY the query slot -> predict the color id
        return imgs.to(dev), ids.to(dev), tgt.to(dev)

    def make_cfg(shuffle):
        side = 32 // 4                                       # 8 patches per side -> 64 patches
        return VLMConfig(vit_img_size=32, vit_patch_size=4, vit_hidden_dim=128, vit_n_heads=4, vit_n_blocks=2,
                         lm_hidden_dim=128, lm_n_heads=4, lm_n_kv_heads=2, lm_n_blocks=2,
                         lm_vocab_size=VOCAB, image_token_id=IMAGE_ID,
                         mp_pixel_shuffle_factor=shuffle, mp_image_token_length=side**2 // shuffle**2)

    @torch.no_grad()
    def color_acc(m, L_img, blind=False, n=400):
        imgs, ids, tgt = batch(n, L_img)
        logits, _ = m(ids, images=None if blind else imgs)
        pred = logits[:, -1, :N_COLOR].argmax(-1)           # read the query slot, restricted to color tokens
        return (pred == tgt[:, -1]).float().mean().item()

    def train(m, L_img, steps=400, blind=False, lr=1e-3):
        opt = torch.optim.AdamW(m.parameters(), lr=lr)
        first = last = None
        for t in range(steps):
            imgs, ids, tgt = batch(32, L_img)
            _, loss = m(ids, images=None if blind else imgs, targets=tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            if t == 0: first = loss.item()
            last = loss.item()
        return first, last

    chance = 1 / N_COLOR
    print(f"world    {N_COLOR} colors x {len(SHAPES)} shapes | image 32x32 -> 64 patches | device {dev.type} | chance {chance:.0%}")

    # 1) the bridge works: with the image spliced in, the LM learns to read the color off the visual
    #    tokens through ordinary causal attention -- loss falls, color accuracy goes far past chance.
    torch.manual_seed(0); cfg = make_cfg(2); L = cfg.mp_image_token_length      # s=2 -> 16 visual tokens
    m = VLM(cfg).to(dev)
    first, last = train(m, L)
    acc = color_acc(m, L)
    print(f"learns   s=2  {L:>2} img-tokens  loss {first:.2f} -> {last:.2f}   color-acc {acc:.0%}")
    assert last < first - 0.5 and acc > 0.9, "the spliced vision did not teach the LM to name colors"

    # 2) control: feed the SAME placeholders but no image (splice skipped -> placeholder embedding is a
    #    constant, color-free). the only thing removed is the visual signal, and accuracy falls to chance.
    torch.manual_seed(0); mb = VLM(make_cfg(2)).to(dev)
    train(mb, L, blind=True)
    acc_blind = color_acc(mb, L, blind=True)
    print(f"blind    same prompt, image removed              color-acc {acc_blind:.0%}  (~chance)")
    assert acc_blind < chance + 0.10, "a blind model should not beat chance -- the signal must come from the image"

    # 3) the budget knob: pixel-shuffle factor s sets how many visual tokens one image costs. a low-detail
    #    answer (color) survives heavy compression, so the LM still nails it at 16x fewer tokens, while the
    #    sequence length -- and the O(T^2) attention cost -- collapse. (the slack adaptive allocation exploits.)
    print("budget   s   img-tokens   seq-len   rel-attn-cost   color-acc")
    for s in (1, 2, 4):
        torch.manual_seed(0); c = make_cfg(s); Ls = c.mp_image_token_length
        ms = VLM(c).to(dev); train(ms, Ls)
        a = color_acc(ms, Ls); T = Ls + 1; rel = (T / 65) ** 2     # 65 = seq-len at s=1 (no compression)
        print(f"          {s}      {Ls:>3}        {T:>4}      {rel:>6.2f}x         {a:.0%}")
        assert a > 0.9, f"s={s} ({Ls} tokens) failed to learn the color"

    print("\nthrough-line: splice is a dumb scatter -- expand <image> into L placeholders, overwrite their "
          "embeddings with the projected visual tokens, hand the sequence to an unmodified causal LM. vision "
          "becomes 'just more tokens'; attention does the rest. removing the image drops it to chance, and the "
          "color survives a 16x token cut (64->4) at no accuracy cost while attention cost falls ~170x -- on a "
          "low-detail question the visual budget is almost all slack.")
