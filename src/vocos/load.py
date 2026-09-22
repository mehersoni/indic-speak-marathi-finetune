"""
Vocos decoder checkpoint loader.
Loads weights into VocosDecoder architecture from checkpoint dictionary.
"""

import torch
from src.vocos.model import VocosDecoder

LATENT_DIM = 768


def load_vocos(ckpt_path: str = "best.pt", device: str = "cpu") -> VocosDecoder:
    """Builds the decoder from the checkpoint config and loads trained weights."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = dict(ck["config"]["model"])
    # latent_dim is passed explicitly; remove it from config dict if present to avoid duplicate-kwarg TypeError
    model_cfg.pop("latent_dim", None)
    dec = VocosDecoder(latent_dim=LATENT_DIM, **model_cfg)
    dec.load_state_dict(ck.get("ema") or ck["vocos"])
    return dec.eval().to(device)
