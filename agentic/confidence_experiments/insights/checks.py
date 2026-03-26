from pathlib import Path

import numpy as np
from scipy import stats

X_GRID = np.linspace(0, 100, 1000)

N_BOOT = 10_000

def bootstrap_mean_test(
    x: np.ndarray,
    mu0: float = 50.0,
    n_boot: int = N_BOOT,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, np.ndarray, tuple[float, float]]:
    """One-sided bootstrap test for H1: mean(x) > mu0.

    Returns (observed_mean, p_value, boot_means, ci_95).
    """
    if rng is None:
        rng = np.random.default_rng()

    obs_mean = float(np.mean(x))
    n = len(x)

    # resample with replacement
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(x[idx], axis=1)

    # one-sided p-value: proportion of bootstrap means <= mu0
    p_value = ((boot_means <= mu0).sum() + 1) / (n_boot + 1)

    ci_95 = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))
    return obs_mean, p_value, boot_means, ci_95

def overlap_coefficient(
    a: np.ndarray, b: np.ndarray, x_grid: np.ndarray = X_GRID, bw: float | None = None
) -> float:
    """Area of overlap between two KDE-estimated distributions (0 to 1).

    If bw is None, uses Scott's rule (scipy default). Otherwise uses the given
    fixed bandwidth for both KDEs.
    """
    kde_a = stats.gaussian_kde(a)
    kde_b = stats.gaussian_kde(b)
    if bw is not None:
        kde_a.set_bandwidth(bw_method=bw / kde_a.dataset.std(ddof=1))
        kde_b.set_bandwidth(bw_method=bw / kde_b.dataset.std(ddof=1))
    return float(np.trapz(np.minimum(kde_a(x_grid), kde_b(x_grid)), x_grid))