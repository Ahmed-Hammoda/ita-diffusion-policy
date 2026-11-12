"""The full language-conditioned diffusion policy.   [Owner: Model + Diffusion — week 2]

Encoder input tokens (each projected to d_model with its own nn.Linear, plus a learned
"token type" embedding and positional encoding):
    1 language token          from lang       (B, 512)
    T_o scene-image tokens    from img_scene  (B, T_o, 384)
    T_o wrist-image tokens    from img_wrist  (B, T_o, 384)
    T_o state tokens          from state      (B, T_o, 8)
  -> Encoder -> memory (B, 1 + 3*T_o, d_model)

Decoder input:
    noisy action chunk (B, H, 7) -> Linear -> (B, H, d_model) + positional encoding
    + timestep embedding of k (added to every action token, or prepended as an extra memory token)
  -> Decoder(x, memory) -> Linear(d_model, 7) -> predicted noise eps (B, H, 7)

Training loss (in compute_loss):
    k ~ Uniform{0..K-1}, noise ~ N(0, I), x_k = scheduler.add_noise(actions, noise, k)
    loss = MSE(model(obs, x_k, k), noise)        (masked by action_mask for padded steps)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiffusionPolicy(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        raise NotImplementedError("TODO: build projections, Encoder, Decoder, output head from cfg['model']")

    def encode(self, obs: dict) -> torch.Tensor:
        """obs dict (see ita/data/dataset.py) -> memory (B, S, d_model)."""
        raise NotImplementedError

    def forward(self, obs: dict, noisy_actions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the noise: (B, H, 7)."""
        raise NotImplementedError

    def compute_loss(self, batch: dict, scheduler) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def predict_action_chunk(self, obs: dict, scheduler, num_inference_steps: int = 10) -> torch.Tensor:
        """Start from pure noise and run DDIM -> normalised actions (B, H, 7). Encode obs ONCE, reuse memory."""
        raise NotImplementedError
