"""
Vocos decoder architecture for Indic-Speak.
Transforms SNAC quantized latent tensors into 24 kHz audio waveforms using ConvNeXt and iSTFT.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, layer_scale_init: float):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = self.gamma * x
        return residual + x.transpose(1, 2)


class NoiseInject(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gain = nn.Linear(dim, dim)
        nn.init.zeros_(self.gain.weight)
        nn.init.zeros_(self.gain.bias)
        self.noise_scale = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_scale == 0.0:
            return x
        g = self.gain(x.transpose(1, 2)).transpose(1, 2)
        return x + g * torch.randn_like(x) * self.noise_scale


class ISTFTHead(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop: int):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.out = nn.Linear(dim, n_fft + 2)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1), mode="replicate")
        h = self.out(x.transpose(1, 2)).transpose(1, 2)
        mag, phase = h.chunk(2, dim=1)
        mag = torch.exp(mag.clamp(max=6.0))
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = mag.float() * (torch.cos(phase.float()) + 1j * torch.sin(phase.float()))
            audio = torch.istft(
                spec,
                self.n_fft,
                hop_length=self.hop,
                win_length=self.n_fft,
                window=self.window.float(),
                center=True,
            )
        self.last_spec = spec
        return audio.unsqueeze(1)


class TimeDomainHead(nn.Module):
    def __init__(self, dim: int, rates=(8, 4, 4), channels: int = 768, noise: bool = True):
        super().__init__()
        from snac.layers import DecoderBlock, Snake1d, WNConv1d
        assert int(torch.tensor(rates).prod()) == 128
        layers = [WNConv1d(dim, channels, kernel_size=7, padding=3)]
        for i, r in enumerate(rates):
            layers.append(DecoderBlock(channels // 2 ** i, channels // 2 ** (i + 1), r, noise))
        out_dim = channels // 2 ** len(rates)
        layers += [Snake1d(out_dim), WNConv1d(out_dim, 1, kernel_size=7, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class VocosDecoder(nn.Module):
    """Maps SNAC latents z_q [B, latent_dim, L] to waveform [B, 1, 512*L]."""

    def __init__(
        self,
        latent_dim: int = 768,
        dim: int = 512,
        intermediate_dim: int = 1536,
        num_blocks_pre: int = 2,
        num_blocks_post: int = 8,
        upsample: int = 2,
        n_fft: int = 1024,
        hop: int = 256,
        noise_inject: bool = False,
        head_type: str = "istft",
    ):
        super().__init__()
        assert upsample * hop == 512, "Must preserve SNAC 512 samples per latent step"
        self.upsample = upsample
        n_total = num_blocks_pre + num_blocks_post
        ls = 1.0 / n_total
        self.stem = nn.Conv1d(latent_dim, dim, kernel_size=7, padding=3)
        self.pre = nn.ModuleList(
            [ConvNeXtBlock(dim, intermediate_dim, ls) for _ in range(num_blocks_pre)]
        )
        self.up_conv = nn.Conv1d(dim, dim, kernel_size=7, padding=3)
        self.noise_post = NoiseInject(dim) if noise_inject else None
        self.noise_head = NoiseInject(dim) if noise_inject else None
        self.post = nn.ModuleList(
            [ConvNeXtBlock(dim, intermediate_dim, ls) for _ in range(num_blocks_post)]
        )
        self.final_norm = nn.LayerNorm(dim, eps=1e-6)
        if head_type == "timedomain":
            assert upsample == 4
            self.head = TimeDomainHead(dim)
        else:
            self.head = ISTFTHead(dim, n_fft=n_fft, hop=hop)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        x = self.stem(z_q)
        for blk in self.pre:
            x = blk(x)
        x = x.repeat_interleave(self.upsample, dim=-1)
        x = self.up_conv(x)
        if self.noise_post is not None:
            x = self.noise_post(x)
        for blk in self.post:
            x = blk(x)
        x = self.final_norm(x.transpose(1, 2)).transpose(1, 2)
        if self.noise_head is not None:
            x = self.noise_head(x)
        return self.head(x)
