# cv/vl — Vision-Language Model (the splice)

A VLM is a language model that can see. The whole trick is one move: turn an image into a short
list of vectors that live in the LM's embedding space, then **drop those vectors into the token
sequence where a few `<image>` placeholder tokens sit**, and hand the spliced sequence to an
ordinary causal LM. Nothing about the LM changes — no cross-attention, no adapters, no gating. The
image becomes *just more tokens*, and the LM's own self-attention does all the cross-modal work
(subject to the causal mask, so text after the image can attend to it, text before cannot). This is
the LLaVA-family "decoder-only / early-fusion" design, and it is almost embarrassingly mechanical:
the splice itself is a boolean-mask scatter with zero learnable parts. The intelligence is upstream
(a vision tower that encodes the image, a projector that lands it in the right space) and downstream
(the LM that reads it) — the bridge in the middle is dumb on purpose. Three pieces meet here: a ViT
vision tower (reused from [`../vit`](../vit), encoder mode, no `[CLS]`), a **modality projector**
(pixel shuffle + 2-layer MLP), and a causal LM (reused from [`../../llm/nano`](../../llm/nano),
which already accepts `inputs_embeds`).

> **References** — Liu et al., *Visual Instruction Tuning* (LLaVA), NeurIPS 2023
> ([arXiv:2304.08485](https://arxiv.org/abs/2304.08485)) — the splice / early-fusion paradigm ·
> HuggingFace [nanoVLM](https://github.com/huggingface/nanoVLM) — the reference implementation this
> mirrors · pixel-shuffle visual-token compression from the Idefics3 / SmolVLM lineage
> ([arXiv:2504.05299](https://arxiv.org/abs/2504.05299)).

```
image [B,3,H,W]                              text ids [B,T]   (L_img <image> placeholders per image)
   │  ViT encoder (num_classes=0, no CLS)        │  decoder.embed
   ▼                                             ▼
patches [N_img, 64, d_vit]                    embeds [B, T, D]      (placeholder rows hold garbage)
   │  pixel shuffle: 64 → 64/s² tokens,          │
   │  d_vit → d_vit·s²  (space → channel)         │
   │  MLP: Linear(d_vit·s² → D) · GELU · Linear(D → D)
   ▼                                             │
visual [N_img, L_img, D] ───── splice ─────────▶ embeds[ids == <image>] = visual.reshape(-1, D)
                                                 │   boolean scatter · #placeholders == #visual tokens
                                                 ▼
                                           sequence [B, T, D]
                                                 │  unmodified causal LM (NanoLLM, inputs_embeds)
                                                 ▼
                                           logits [B, T, vocab]
```

**Pixel shuffle** is the only non-obvious operation, and it carries the whole token-budget story. It
is `view`+`reshape`+`permute`+`reshape` — no learnable parameters — that merges each `s×s` block of
spatial patches into one fat token: token count ÷ s², channel dim × s². It is lossless (a pure
re-layout, space moved into channels); the actual compression happens in the MLP that follows, whose
first layer is sized `d_vit·s² → D`. So `s` (`mp_pixel_shuffle_factor`) is the budget dial — one
image costs `(img/patch)² / s²` tokens in the LM sequence. **The splice**
(`_replace_img_tokens_with_embd`) is the heart: `mask = input_ids == image_token_id` finds every
placeholder across the batch, and `out[mask] = visual.reshape(-1, D)` overwrites them in one
vectorized assignment. Boolean indexing flattens row-major and the visual tokens are stacked in the
same order, so the only requirement is a **counting contract** — total placeholders must equal total
visual tokens — asserted in place. This is why data prep inserts exactly `mp_image_token_length`
placeholders per image. The loss masks every non-answer position with `ignore_index=-100`, so the
model is never trained to "predict" image tokens.

## Experiment (`python vl.py`)

A real but tiny training run, not a forward-only assertion: a synthetic world of one colored shape on
a near-black field (6 colors × 3 shapes, where shape is a distractor), prompt `[<image>×L, "color?"]`,
supervised only at the query slot. Runs in under a minute on MPS.

```
world    6 colors x 3 shapes | image 32x32 -> 64 patches | device mps | chance 17%
learns   s=2  16 img-tokens  loss 2.10 -> 0.00   color-acc 100%
blind    same prompt, image removed              color-acc 23%  (~chance)
budget   s   img-tokens   seq-len   rel-attn-cost   color-acc
          1       64          65        1.00x         100%
          2       16          17        0.07x         100%
          4        4           5        0.01x         100%
```

- **learns** — with the image spliced in, the LM learns to name the color off the visual tokens
  through ordinary causal attention: loss collapses, accuracy hits 100% from a 17% floor. The bridge
  works end-to-end, trained from scratch (no pretrained backbones — so unlike real LLaVA there is
  nothing to freeze; the projector and both towers learn together).
- **blind** — feed the *same* prompt with no image (splice skipped, so the placeholder rows keep a
  single constant, color-free embedding) and accuracy falls back to chance. The one thing removed is
  the visual signal, so this pins the accuracy *to the splice* — it is not the prompt leaking the
  answer.
- **budget** — `s` sets how many tokens one image costs. A low-detail answer (color) survives heavy
  compression: at `s=4` the image is 4 tokens instead of 64 (16× fewer), the O(T²) attention cost
  drops ~170×, and accuracy stays 100%. On a question that needs no spatial detail, the visual budget
  is almost all slack — the opening that adaptive token allocation exploits.

This is the **mechanism** sanity — that vision flows through the splice and the budget dial behaves —
not a capability result. A harder, detail-dependent question (e.g. *which* shape, or counting) is
where the budget sweep would start to bite, and where a learned, per-image token allocation would
have to earn its keep.

> Next: the visual-token-budget line this rides on — [`../tokbudget`](../tokbudget) and
> [`../tokmerge`](../tokmerge).
