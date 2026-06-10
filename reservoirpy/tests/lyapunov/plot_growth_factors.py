"""Plot per-cycle expansion/contraction factors for a known dynamical system.

Usage::

    python -m reservoirpy.tests.lyapunov.plot_growth_factors           # lorenz
    python -m reservoirpy.tests.lyapunov.plot_growth_factors lorenz96  # any system
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from reservoirpy.tests.lyapunov.lyapunov_test import known_system_test

_RTOLS = [0.5, 0.2, 0.1, 0.05]
_RTOL_COLORS = ["#555555", "#888888", "#aaaaaa", "#cccccc"]


def _running_stats(log_growths: np.ndarray, cycle_length: int, t_step: float):
    """Return running stats for the growth-factor plot.

    NaN entries in log_growths (collapsed directions) are excluded from sums
    so the running mean / CI are based only on valid cycles.

    Returns
    -------
    running_mean : (n, k) — exp of running mean of log-growths (factor space)
    lower : (n, k) — lower 95 % CI bound in factor space
    upper : (n, k) — upper 95 % CI bound in factor space
    running_ci_half_rate : (n, k) — 95 % CI half-width in rate space (1/time)
    running_spectrum : (n, k) — running mean Lyapunov estimates in rate space
    """
    n, k = log_growths.shape
    rate_scale = 1.0 / (cycle_length * t_step)

    valid = np.isfinite(log_growths).astype(float)           # (n, k)
    lg_clean = np.where(np.isfinite(log_growths), log_growths, 0.0)

    cum_valid = np.cumsum(valid, axis=0)                      # valid-sample count
    cum_sum = np.cumsum(lg_clean, axis=0)
    cum_sum_sq = np.cumsum(lg_clean ** 2, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        running_log_mean = np.where(cum_valid > 0, cum_sum / cum_valid, np.nan)
        running_mean_sq = np.where(cum_valid > 0, cum_sum_sq / cum_valid, np.nan)
        running_var = np.maximum(running_mean_sq - running_log_mean ** 2, 0.0)
        running_std = np.where(
            cum_valid > 1,
            np.sqrt(running_var * cum_valid / (cum_valid - 1)),
            np.inf,
        )
        ci_half = 1.96 * running_std / np.sqrt(np.maximum(cum_valid, 1))

    running_mean = np.exp(running_log_mean)
    lower = np.exp(running_log_mean - ci_half)
    upper = np.exp(running_log_mean + ci_half)
    lower[0] = np.nan
    upper[0] = np.nan

    running_ci_half_rate = ci_half * rate_scale
    running_spectrum = running_log_mean * rate_scale

    return running_mean, lower, upper, running_ci_half_rate, running_spectrum


def _convergence_cycle(running_ci_half_rate: np.ndarray, running_spectrum: np.ndarray,
                       rtol: float) -> int | None:
    """First cycle where max CI half-width < rtol * |lambda_1(n)|."""
    for n in range(1, running_ci_half_rate.shape[0]):
        lam1 = running_spectrum[n, 0]
        if lam1 != 0 and np.max(running_ci_half_rate[n]) < rtol * abs(lam1):
            return n
    return None


def plot_growth_factors(system_name: str = "lorenz", **kwargs) -> str:
    """Run known_system_test and save a log-scale growth-factor plot.

    Parameters
    ----------
    system_name : str
        One of the systems supported by ``known_system_test``.
    **kwargs
        Forwarded to ``known_system_test`` (e.g. ``N=36`` for lorenz96).

    Returns
    -------
    str
        Path to the saved PNG.
    """
    print(f"Running {system_name} Lyapunov test …")
    result = known_system_test(system_name, display=False, **kwargs)

    log_growths = result["log_growths"]              # (n_cycles, k)
    spectrum = result["spectrum"]                     # (k,) — rates (1/time)
    collapsed = result["collapsed_directions"]        # (k,) bool
    cycle_length = result["cycle_length"]
    t_step = result["h"]
    n_cycles, k = log_growths.shape

    factors = np.exp(np.abs(log_growths))
    running_mean, lower, upper, running_ci_half_rate, running_spectrum = _running_stats(
        log_growths, cycle_length, t_step
    )
    running_mean = np.abs(running_mean)
    lower = np.abs(lower)
    upper = np.abs(upper)

    cycles = np.arange(n_cycles)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i in range(k):
        if collapsed[i]:
            ax.plot(cycles, factors[:, i], color="0.75", alpha=0.20, lw=0.5,
                    label=f"$\\lambda_{{{i+1}}}$ [collapsed]")
        else:
            color = f"C{i % 10}"
            lam_str = f"{spectrum[i]:+.4f}"
            label = f"$\\lambda_{{{i+1}}}$ = {lam_str}"
            ax.plot(cycles, factors[:, i], color=color, alpha=0.25, lw=0.6)
            ax.plot(cycles, running_mean[:, i], color=color, lw=1.5, label=label)
            ax.fill_between(cycles, lower[:, i], upper[:, i], color=color, alpha=0.18)

    # convergence markers — CI evaluated over non-collapsed directions only
    good = ~collapsed
    rci_good = running_ci_half_rate[:, good] if good.any() else running_ci_half_rate
    for rtol, vcolor in zip(_RTOLS, _RTOL_COLORS):
        cyc = _convergence_cycle(rci_good, running_spectrum, rtol)
        if cyc is not None:
            ax.axvline(cyc, color=vcolor, lw=1.2, ls="--",
                       label=f"rtol={rtol} → cycle {cyc}")

    ax.set_yscale("log")
    ax.set_xlabel("Post-breakin cycle")
    ax.set_ylabel("Per-cycle factor  |diag(R)| / pert_scale")
    ax.set_title(
        f"{system_name.replace('_', ' ').title()} expansion/contraction factors"
        f" (k={k}, {n_cycles} cycles)"
    )
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"{system_name}_growth_factors.png",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    spec_str = "  ".join(
        (f"{l:+.4f} ± {c:.4f}" if np.isfinite(l) else f"[collapsed] ± {c:.4f}")
        for l, c in zip(spectrum, result["spectrum_ci"])
    )
    ky = result["ky_dim"]
    ky_ci = result["ky_dim_ci"]
    ky_str = f"{ky:.3f} ± {ky_ci:.3f}" if ky is not None else "undefined"
    print(f"Spectrum: {spec_str}")
    print(f"KY dim:   {ky_str}")
    print(f"Cycles:   {n_cycles}  converged={result['converged']}")

    return out_path


if __name__ == "__main__":
    sys_name = sys.argv[1] if len(sys.argv) > 1 else "lorenz"
    extra = {}
    if sys_name == "lorenz96" and len(sys.argv) > 2:
        extra["N"] = int(sys.argv[2])
    plot_growth_factors(sys_name, **extra)
