"""
Vocos neural vocoder decoder package.
Reconstructs 24 kHz audio waveforms from SNAC quantized latent vectors.
"""

from src.vocos.load import load_vocos
from src.vocos.model import VocosDecoder

__all__ = ["load_vocos", "VocosDecoder"]
