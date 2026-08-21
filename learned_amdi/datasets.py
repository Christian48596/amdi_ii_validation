from __future__ import annotations

import numpy as np

from amdi.synthetic import add_gaussian_noise, four_region_image


def random_multiscale_image(n: int, seed: int) -> np.ndarray:
    """Deterministic synthetic family for train/validation/test separation."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n]
    X = (x + 0.5) / n
    Y = (y + 0.5) / n
    img = 0.15 + 0.20 * X + 0.10 * Y
    # Smooth Gaussian structures.
    for _ in range(3):
        cx, cy = rng.uniform(0.15, 0.85, 2)
        width = rng.uniform(0.025, 0.12)
        amp = rng.uniform(0.15, 0.45)
        img += amp * np.exp(-((X-cx)**2 + (Y-cy)**2) / width)
    # One sharp rectangular inclusion.
    x0, y0 = rng.uniform(0.1, 0.6, 2)
    w, h = rng.uniform(0.12, 0.30, 2)
    mask = (X >= x0) & (X <= min(x0+w, 0.95)) & (Y >= y0) & (Y <= min(y0+h, 0.95))
    img[mask] += rng.uniform(0.15, 0.35)
    # Local oscillatory patch.
    px, py = rng.uniform(0.45, 0.75, 2)
    patch = (np.abs(X-px) < 0.20) & (np.abs(Y-py) < 0.20)
    freq = int(rng.integers(4, 10))
    img[patch] += 0.10 * np.sin(2*np.pi*freq*X[patch]) * np.sin(2*np.pi*freq*Y[patch])
    img -= img.min()
    img /= max(img.max(), 1e-15)
    return np.clip(img, 0.0, 1.0)


def truth_image(kind: str, n: int, seed: int = 0) -> np.ndarray:
    if kind == "four_region":
        return four_region_image(n)
    if kind == "random_multiscale":
        return random_multiscale_image(n, seed)
    raise ValueError(kind)


def noisy_case(kind: str, n: int, image_seed: int, noise_sigma: float, noise_seed: int) -> tuple[np.ndarray, np.ndarray]:
    truth = truth_image(kind, n, image_seed)
    noisy = add_gaussian_noise(truth, noise_sigma, seed=noise_seed)
    return truth, noisy
