from dataclasses import dataclass


@dataclass
class VLMConfig:
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 4 * vit_hidden_dim
    vit_patch_size: int = 16
    vit_img_size: int = 512
    vit_n_heads: int = 12
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False          # no CLS -> patch count stays a perfect square, so pixel shuffle works

    lm_hidden_dim: int = 960            
    lm_inter_dim: int = 2560
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_base_vocab_size: int = 49152
    extra_token_amount: int = 66        # image / tiling tokens appended to the vocab (incl. <|image|>)
    lm_vocab_size: int = lm_base_vocab_size + extra_token_amount
    image_token_id: int = lm_base_vocab_size  # <|image|> placeholder id; splice targets this
    lm_n_heads: int = 15
    lm_n_kv_heads: int = 5             
    lm_n_blocks: int = 32
    lm_use_tokens: bool = False       
    lm_tie_weights: bool = True         # tie LM head to the token-embedding matrix

    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = 64     # = (vit_img_size/vit_patch_size)^2 / shuffle^2 = 1024 / 16

    @classmethod
    def toy(cls):                       # small, CPU-runnable preset for the __main__ experiment
        return cls(vit_img_size=32, vit_patch_size=4, vit_hidden_dim=128,
                   vit_n_heads=4, vit_n_blocks=2,
                   lm_hidden_dim=128, lm_n_heads=4, lm_n_kv_heads=2, lm_n_blocks=2,
                   mp_pixel_shuffle_factor=2, mp_image_token_length=16)
