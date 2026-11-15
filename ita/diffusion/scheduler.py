"""DDPM noise schedule (training) + DDIM sampler (inference), written from scratch.

Notation: K = num_train_steps, beta_k = noise variance at step k, alpha_k = 1 - beta_k,
          abar_k = prod_{i<=k} alpha_i  (alphas_cumprod).

Forward process (training):   x_k = sqrt(abar_k) * x_0 + sqrt(1 - abar_k) * eps
DDIM step (eta = 0, deterministic), from step t to t_prev:
    x0_pred = (x_t - sqrt(1 - abar_t) * eps_pred) / sqrt(abar_t)       (clip to [-1, 1])
    x_prev  = sqrt(abar_prev) * x0_pred + sqrt(1 - abar_prev) * eps_pred
    with abar_prev = 1.0 when t_prev < 0 (final step).
DDIM timesteps: n values evenly spaced from K-1 down to 0 ("trailing" spacing).

"squaredcos" schedule (Nichol & Dhariwal 2021):
    f(t) = cos^2(((t/K) + s) / (1 + s) * pi/2), s = 0.008
    beta_k = min(1 - f(k+1)/f(k), 0.999)

"linear" schedule (Ho et al. 2020), kept for comparison:
    beta_k evenly spaced from 1e-4 to 0.02 (rescaled for K != 1000).

The specification is in tests/test_scheduler.py; notebooks/step5_scheduler.ipynb explains and
checks every method on real LIBERO action chunks. `diffusers` is not used anywhere.
"""

from __future__ import annotations

import math
from typing import Callable

import torch


def _squaredcos_betas(K: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    t = torch.arange(K + 1, dtype=torch.float64) / K                     # 0, 1/K, ..., 1
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2                    # f(0) .. f(K), f(t) ~ abar
    betas = 1 - f[1:] / f[:-1]                                             # (K,)
    return betas.clamp(max=max_beta)


def _linear_betas(K: int) -> torch.Tensor:
    scale = 1000 / K                                                       # keeps abar_K close to 0 for small K
    return torch.linspace(scale * 1e-4, scale * 0.02, K, dtype=torch.float64)


class NoiseScheduler:
    def __init__(self, num_train_steps: int = 100, schedule: str = "squaredcos"):
        self.num_train_steps = K = num_train_steps
        if schedule == "squaredcos":
            betas = _squaredcos_betas(K)
        elif schedule == "linear":
            betas = _linear_betas(K)
        else:
            raise ValueError(f"unknown schedule {schedule!r} (use 'squaredcos' or 'linear')")

        # Computed in float64 for an accurate cumulative product, stored as float32.
        alphas = 1 - betas
        self.betas = betas.float()                                         # (K,)
        self.alphas = alphas.float()                                       # (K,)
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).float()         # (K,), abar_k

    def _abar(self, t: torch.Tensor, device: torch.device) -> torch.Tensor:
        """abar for a batch of integer steps, on the right device."""
        return self.alphas_cumprod.to(device)[t.to(device).long()]

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x0, noise: (B, H, A); t: (B,) long -> x_t (B, H, A)."""
        ab = self._abar(t, x0.device).view(-1, *([1] * (x0.dim() - 1)))  # (B, 1, 1) broadcasts over H, A
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def ddim_timesteps(self, num_inference_steps: int) -> list[int]:
        """Evenly spaced from K-1 down to 0, e.g. K=100, n=10 -> [99, 88, 77, ..., 11, 0].

        Starting at K-1 (abar ~ 0) matches the pure noise we start from. The "leading" spacing
        [90, 80, ..., 0] starts at a step that still contains ~14% signal. On a trained toy model
        this spacing is better for 3-10 steps; with 1-2 steps the epsilon -> x0 inversion at
        k = K-1 is ill-conditioned (divides by sqrt(abar) ~ 5e-4), so use n >= 3.
        See notebooks/step5_scheduler.ipynb, Parts 4.3 and 6.2.
        """
        K, n = self.num_train_steps, num_inference_steps
        assert 1 <= n <= K, f"num_inference_steps must be in [1, {K}]"
        if n == 1:
            return [K - 1]
        return [round((K - 1) * (n - 1 - i) / (n - 1)) for i in range(n)]

    def ddim_step(self, eps_pred: torch.Tensor, t: int, t_prev: int, x_t: torch.Tensor) -> torch.Tensor:
        """One deterministic DDIM update (eta = 0) from step t to step t_prev (t_prev < 0 = final step)."""
        ab_t = self.alphas_cumprod[t].item()
        ab_prev = self.alphas_cumprod[t_prev].item() if t_prev >= 0 else 1.0

        # 1) Guess the clean chunk from the noisy one and the predicted noise. Actions live in [-1, 1].
        x0_pred = (x_t - math.sqrt(1 - ab_t) * eps_pred) / math.sqrt(ab_t)
        x0_pred = x0_pred.clamp(-1, 1)

        # 2) Re-noise that guess to the (smaller) noise level of t_prev, reusing the same noise direction.
        return math.sqrt(ab_prev) * x0_pred + math.sqrt(1 - ab_prev) * eps_pred

    @torch.no_grad()
    def sample(self, eps_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], shape: tuple,
               num_inference_steps: int = 10, device: str | torch.device = "cpu",
               generator: torch.Generator | None = None) -> torch.Tensor:
        """Start from x ~ N(0, I) of `shape`, call eps_fn(x_t, t_batch) at each DDIM step, return x_0."""
        x = torch.randn(shape, generator=generator, device=device)
        timesteps = self.ddim_timesteps(num_inference_steps)
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
            eps = eps_fn(x, t_batch)
            x = self.ddim_step(eps, t, t_prev, x)
        return x
