"""DDPM noise schedule (training) + DDIM sampler (inference), written from scratch.   [Owner: Diffusion — week 1]

Notation: K = num_train_steps, beta_k = noise variance at step k, alpha_k = 1 - beta_k,
          abar_k = prod_{i<=k} alpha_i  (alphas_cumprod).

Forward process (training):   x_k = sqrt(abar_k) * x_0 + sqrt(1 - abar_k) * eps
DDIM step (eta = 0, deterministic), from step t to t_prev:
    x0_pred = (x_t - sqrt(1 - abar_t) * eps_pred) / sqrt(abar_t)       (clip to [-1, 1])
    x_prev  = sqrt(abar_prev) * x0_pred + sqrt(1 - abar_prev) * eps_pred
    with abar_prev = 1.0 when t_prev < 0 (final step).

"squaredcos" schedule (Nichol & Dhariwal 2021):
    f(t) = cos^2(((t/K) + s) / (1 + s) * pi/2), s = 0.008
    beta_k = min(1 - f(k+1)/f(k), 0.999)

The specification is in tests/test_scheduler.py.
You may compare your numbers against `diffusers.DDIMScheduler` for debugging, but do not import it here.
"""

from __future__ import annotations

from typing import Callable

import torch


class NoiseScheduler:
    def __init__(self, num_train_steps: int = 100, schedule: str = "squaredcos"):
        self.num_train_steps = num_train_steps
        # TODO: self.betas, self.alphas, self.alphas_cumprod  — float32 tensors of shape (K,)
        raise NotImplementedError("TODO (Diffusion)")

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x0, noise: (B, H, A); t: (B,) long -> x_t (B, H, A)."""
        raise NotImplementedError

    def ddim_timesteps(self, num_inference_steps: int) -> list[int]:
        """Evenly spaced, descending, e.g. K=100, n=10 -> [90, 80, ..., 0]."""
        raise NotImplementedError

    def ddim_step(self, eps_pred: torch.Tensor, t: int, t_prev: int, x_t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, eps_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], shape: tuple,
               num_inference_steps: int = 10, device: str | torch.device = "cpu",
               generator: torch.Generator | None = None) -> torch.Tensor:
        """Start from x ~ N(0, I) of `shape`, call eps_fn(x_t, t_batch) at each DDIM step, return x_0."""
        raise NotImplementedError
