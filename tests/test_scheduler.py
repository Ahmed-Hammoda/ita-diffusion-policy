"""Specification for ita/diffusion/scheduler.py — these pass once the scheduler is implemented."""

import torch

from ita.diffusion.scheduler import NoiseScheduler

K = 100


def test_alphas_cumprod_decreasing_in_unit_interval():
    s = NoiseScheduler(K, "squaredcos")
    a = s.alphas_cumprod
    assert a.shape == (K,)
    assert torch.all(a > 0) and torch.all(a < 1)
    assert torch.all(a[1:] < a[:-1])


def test_add_noise_formula():
    s = NoiseScheduler(K)
    x0, noise = torch.randn(3, 16, 7), torch.randn(3, 16, 7)
    t = torch.tensor([0, 50, 99])
    xt = s.add_noise(x0, noise, t)
    ab = s.alphas_cumprod[t].view(-1, 1, 1)
    assert torch.allclose(xt, ab.sqrt() * x0 + (1 - ab).sqrt() * noise, atol=1e-5)


def test_ddim_timesteps():
    ts = NoiseScheduler(K).ddim_timesteps(10)
    assert len(ts) == 10 and ts == sorted(ts, reverse=True) and ts[-1] == 0


def test_ddim_with_perfect_noise_predictor_recovers_x0():
    """If the model predicted the true noise exactly, DDIM must return the clean actions."""
    torch.manual_seed(0)
    s = NoiseScheduler(K)
    x0 = torch.rand(2, 16, 7) * 2 - 1  # inside [-1, 1]

    def oracle_eps(x_t, t):
        ab = s.alphas_cumprod[t].view(-1, 1, 1)
        return (x_t - ab.sqrt() * x0) / (1 - ab).sqrt()

    out = s.sample(oracle_eps, x0.shape, num_inference_steps=10)
    assert torch.allclose(out, x0, atol=1e-3)
