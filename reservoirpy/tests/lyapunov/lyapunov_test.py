"""Lyapunov spectrum computation for known dynamical systems.

This module validates ``_lyapunov()`` against reference dynamical systems from
``reservoirpy.datasets._chaos``.  It is designed to be run as a script (item 3
of the Lyapunov implementation plan) rather than as a pytest test suite.

Usage::

    # single system (with display)
    python -m reservoirpy.tests.lyapunov.lyapunov_test lorenz
    # all item-3 systems → writes true_values.csv
    python -m reservoirpy.tests.lyapunov.lyapunov_test
"""

import os
import pickle
import sys
from typing import Optional

# Ensure the local reservoirpy fork takes precedence over any installed copy.
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))


import numpy as np
import scipy.fft

from reservoirpy.datasets._chaos import _kuramoto_sivashinsky_etdrk4
from reservoirpy.observables import _lyapunov, valid_time_multitest


# ---------------------------------------------------------------------------
# KS spectral setup  (mirrors _kuramoto_sivashinsky in _chaos.py)
# ---------------------------------------------------------------------------


def _ks_etdrk4_setup(N: int, M: int, h: float, d: float) -> dict:
    """Precompute ETDRK4 scalars for Kuramoto-Sivashinsky integration.

    N: number of spatial grid points.
    M: number of Cauchy quadrature points (16 is standard).
    h: time step.
    d: domain length L.
    """
    k = np.conj(np.r_[np.arange(0, N / 2), [0], np.arange(-N / 2 + 1, 0)]) * (2 * np.pi / d)
    L = k**2 - k**4
    E = np.exp(h * L)
    E2 = np.exp(h * L / 2)
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = h * np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], N, axis=0)
    Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = (-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3
    f1 = h * np.real(np.mean(f1, axis=1))
    f2 = (2 + LR + np.exp(LR) * (-2 + LR)) / LR**3
    f2 = h * np.real(np.mean(f2, axis=1))
    f3 = (-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3
    f3 = h * np.real(np.mean(f3, axis=1))
    g = -0.5j * k
    return {"g": g, "E": E, "E2": E2, "Q": Q, "f1": f1, "f2": f2, "f3": f3}


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class _KnownSystem:
    """Per-system constants, stepper, and IC; presents the ``_lyapunov`` API.

    Subclasses must implement :meth:`step` and :meth:`initial_state`.
    After calling ``initial_state()``, assign its return values to
    ``self.state`` and ``self.stdev`` before passing the instance to
    ``_lyapunov``.
    """

    name: str = ""
    k_default: int = 3
    min_cycles_default: int = 500
    t_step: float = 1.0
    n_warmup: int = 2000
    n_sample: int = 2000

    @property
    def cache_key(self) -> str:
        return self.name

    def evolve(self, n_steps: int) -> None:
        self.state = self.step(self.state, n_steps)

    def step(self, state: np.ndarray, n_steps: int) -> np.ndarray:
        raise NotImplementedError

    def initial_state(self):
        raise NotImplementedError


class _RK4System(_KnownSystem):
    """Autonomous ODE systems integrated with fixed-step RK4.

    Subclasses set ``t_step`` and ``x0`` and implement :meth:`rhs`.
    """

    def rhs(self, s: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def step(self, state: np.ndarray, n_steps: int) -> np.ndarray:
        h = self.t_step
        h2, h6 = 0.5 * h, h / 6.0
        s = state.copy()
        for _ in range(n_steps):
            k1 = self.rhs(s)
            k2 = self.rhs(s + h2 * k1)
            k3 = self.rhs(s + h2 * k2)
            k4 = self.rhs(s + h * k3)
            s += h6 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return s

    def initial_state(self):
        state = self.step(np.asarray(self.x0, dtype=float), self.n_warmup)
        traj = np.empty((self.n_sample, state.size))
        for i in range(self.n_sample):
            state = self.step(state, 1)
            traj[i] = state
        return state, float(np.std(traj))


# ---------------------------------------------------------------------------
# Concrete systems
# ---------------------------------------------------------------------------


class Lorenz(_RK4System):
    name = "lorenz"
    k_default = 3
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    t_step = 0.02
    x0 = [10.0, 10.0, 1.0]

    def rhs(self, s):
        x, y, z = s
        return np.array(
            [self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z]
        )


class Lorenz96(_RK4System):
    name = "lorenz96"
    k_default = 10
    F = 8.0
    t_step = 0.02

    def __init__(self, N: int = 10):
        self.N = N
        x0 = np.ones(N)
        x0[0] += 0.01
        self.x0 = x0

    @property
    def cache_key(self) -> str:
        return f"{self.name}_N{self.N}"

    def rhs(self, s):
        return (np.roll(s, -1) - np.roll(s, 2)) * np.roll(s, 1) - s + self.F


class RabinovichFabrikant(_RK4System):
    name = "rabinovich_fabrikant"
    k_default = 3
    min_cycles_default = 300
    alpha, gamma = 1.1, 0.87
    t_step = 0.05
    x0 = [-1.0, 0.0, 0.5]

    def rhs(self, s):
        x, y, z = s
        return np.array(
            [
                y * (z - 1.0 + x**2) + self.gamma * x,
                x * (3.0 * z + 1.0 - x**2) + self.gamma * y,
                -2.0 * z * (self.alpha + x * y),
            ]
        )


class MackeyGlass(_KnownSystem):
    name = "mackey_glass"
    k_default = 5
    tau, a, b, n_mg = 20.0, 0.2, 0.1, 10
    t_step = 0.1

    @property
    def n_delay(self):
        return round(self.tau / self.t_step)

    def step(self, state: np.ndarray, n_steps: int) -> np.ndarray:
        h = self.t_step
        h2, h6 = 0.5 * h, h / 6.0
        a, b, n = self.a, self.b, self.n_mg

        def _rhs(x_now, x_del):
            return a * x_del / (1.0 + x_del**n) - b * x_now

        buf = state.copy()
        for _ in range(n_steps):
            x_now, x_del = buf[-1], buf[0]
            k1 = _rhs(x_now, x_del)
            k2 = _rhs(x_now + h2 * k1, x_del)
            k3 = _rhs(x_now + h2 * k2, x_del)
            k4 = _rhs(x_now + h * k3, x_del)
            x_new = x_now + h6 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            buf = np.roll(buf, -1)
            buf[-1] = x_new
        return buf

    def initial_state(self):
        state = self.step(0.9 * np.ones(self.n_delay), self.n_warmup)
        traj = np.empty(self.n_sample)
        for i in range(self.n_sample):
            state = self.step(state, 1)
            traj[i] = state[-1]
        return state, float(np.std(traj))


class KuramotoSivashinsky(_KnownSystem):
    name = "kuramoto_sivashinsky"
    k_default = 10
    N, M = 128, 16
    L = 22.0
    t_step = 0.25

    def __init__(self):
        self._scalars = _ks_etdrk4_setup(self.N, self.M, self.t_step, d=self.L)

    def step(self, state: np.ndarray, n_steps: int) -> np.ndarray:
        v = scipy.fft.fft(state)
        for _ in range(n_steps):
            v = _kuramoto_sivashinsky_etdrk4(v, **self._scalars)
        return np.real(scipy.fft.ifft(v))

    def initial_state(self):
        phi = 2.0 * np.pi * np.arange(1, self.N + 1) / self.N
        state = np.cos(phi) * (1.0 + np.sin(phi))
        state = self.step(state, self.n_warmup)
        traj = np.empty((self.n_sample, self.N))
        for i in range(self.n_sample):
            state = self.step(state, 1)
            traj[i] = state
        return state, float(np.std(traj))


# ---------------------------------------------------------------------------
# known_system_test
# ---------------------------------------------------------------------------

_SYSTEMS = {
    "lorenz":               Lorenz,
    "lorenz96":             lambda: Lorenz96(N=10),
    "rabinovich_fabrikant": RabinovichFabrikant,
    "mackey_glass":         MackeyGlass,
    "kuramoto_sivashinsky": KuramotoSivashinsky,
}


def known_system_test(
    system_name: str,
    k: Optional[int] = None,
    cycle_length: Optional[int] = None,
    breakin_cycles: Optional[int] = None,
    min_cycles: Optional[int] = None,
    max_cycles: Optional[int] = None,
    rtol: float = 0.0,
    epsilon: float = 1e-5,
    convergence: str = "ky_dim",
    space: str = "real",
    display: bool = True,
) -> dict:
    """Compute the Lyapunov spectrum of a known dynamical system.

    Parameters
    ----------
    system_name : str
        One of ``"lorenz"``, ``"lorenz96"``, ``"rabinovich_fabrikant"``,
        ``"mackey_glass"``, ``"kuramoto_sivashinsky"``.
    k : int, optional
        Number of Lyapunov exponents.  Defaults per system if None.
    cycle_length : int or None, optional
        Steps between QR reorthonormalizations.  ``None`` (default) lets the
        algorithm estimate it from a short λ₁ probe.
    breakin_cycles : int or None, optional
        Breakin cycles before accumulation.  ``None`` (default) uses adaptive
        breakin (200-cycle chunks, up to 1000 total).
    min_cycles : int, optional
        Minimum accumulation cycles between convergence checks.
    max_cycles : int, optional
        Hard cap on accumulation cycles.  Default 5000.
    rtol : float, optional
        Relative convergence tolerance (0 = run to max_cycles).
    epsilon : float, optional
        Relative perturbation magnitude.
    convergence : str, optional
        Convergence criterion: ``"ky_dim"`` (default), ``"lambda_1"``, or
        ``"all"``.
    space : str, optional
        Ignored (KS always uses real-space representation).
    display : bool, optional
        Show convergence plots.

    Returns
    -------
    dict
        Result from ``_lyapunov`` plus ``"system"`` and ``"h"`` keys.
    """
    factory = _SYSTEMS.get(system_name)
    if factory is None:
        raise ValueError(f"Unknown system: {system_name!r}")

    model = factory()
    cache = _load_cache()
    row = cache.get(model.cache_key)
    if row is not None and row["ic"] is not None:
        model.state = row["ic"].copy()
        model.stdev = row["stdev"]
    else:
        model.state, model.stdev = model.initial_state()
        cache[model.cache_key] = {
            "stdev": model.stdev,
            "ic": model.state.copy(),
            "spectrum": None,
            "spectrum_ci": None,
        }
        _save_cache(cache)

    k = k if k is not None else model.k_default
    min_cycles = min_cycles if min_cycles is not None else model.min_cycles_default
    max_cycles = max_cycles if max_cycles is not None else 2000
    # breakin_cycles = breakin_cycles if breakin_cycles is not None else 4000

    result = _lyapunov(
        model,
        k=k,
        cycle_length=cycle_length,
        breakin_cycles=breakin_cycles,
        min_cycles=min_cycles,
        max_cycles=max_cycles,
        rtol=rtol,
        epsilon=epsilon,
        convergence=convergence,
        display=display,
    )
    result["system"] = system_name
    result["h"] = model.t_step

    entry = cache.setdefault(model.cache_key, {
        "stdev": model.stdev, "ic": model.state.copy(),
        "spectrum": None, "spectrum_ci": None,
    })
    entry["spectrum"] = np.asarray(result["spectrum"])
    entry["spectrum_ci"] = np.asarray(result["spectrum_ci"])
    _save_cache(cache)

    return result


# ---------------------------------------------------------------------------
# Item-3 driver: run all systems, write true_values.csv, print comparison
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", "..", ".."))
_SUMMARY_STATS = os.path.join(
    _REPO_ROOT, "dataset_builder", "summary_stats_new_standards.csv"
)
_TRUE_VALUES_CSV = os.path.join(_THIS_DIR, "true_values.csv")
_CACHE_CSV = os.path.join(_THIS_DIR, "attractor_cache.csv")


def _load_cache() -> dict:
    """Load attractor_cache.csv; return {cache_key: {stdev, ic, spectrum, spectrum_ci}}."""
    if not os.path.exists(_CACHE_CSV):
        return {}
    import csv

    def _parse(s):
        s = (s or "").strip()
        return np.array([float(x) for x in s.split()]) if s else None

    out = {}
    with open(_CACHE_CSV, newline="") as f:
        for row in csv.DictReader(f):
            out[row["system"]] = {
                "stdev": float(row["stdev"]),
                "ic": _parse(row["ic"]),
                "spectrum": _parse(row.get("spectrum", "")),
                "spectrum_ci": _parse(row.get("spectrum_ci", "")),
            }
    return out


def _save_cache(cache: dict) -> None:
    """Write cache dict back to attractor_cache.csv."""
    import csv

    def _fmt(arr):
        return "" if arr is None else " ".join(repr(float(x)) for x in arr)

    fieldnames = ["system", "stdev", "ic", "spectrum", "spectrum_ci"]
    with open(_CACHE_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for key, v in cache.items():
            w.writerow({
                "system": key,
                "stdev": repr(float(v["stdev"])),
                "ic": _fmt(v["ic"]),
                "spectrum": _fmt(v.get("spectrum")),
                "spectrum_ci": _fmt(v.get("spectrum_ci")),
            })



def _load_summary_stats():
    if not os.path.exists(_SUMMARY_STATS):
        return None
    import csv

    rows = {}
    with open(_SUMMARY_STATS) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["System"].lower().replace(" ", "_")] = row
    return rows


def run_item3(verbose: bool = True):
    """Run known_system_test for all item-3 systems and write true_values.csv."""
    systems = [
        # ("lorenz", {}),
        # ("lorenz96", {}),
        # ("rabinovich_fabrikant", {}),
        # ("mackey_glass", {}),
        ("kuramoto_sivashinsky", {}),
    ]

    summary = _load_summary_stats()
    ss_key_map = {
        "lorenz": "lorenz",
        "lorenz96": "lorenz96_10",
        "rabinovich_fabrikant": "rabfab",
        "mackey_glass": "mg",
    }

    rows = []
    all_results = {}

    for sys_name, kwargs in systems:
        space_suffix = kwargs.get("space", "")
        label = f"{sys_name}_{space_suffix}" if space_suffix else sys_name
        if verbose:
            sep = "=" * 60
            print(f"\n{sep}")
            print(f"  {label}")
            print(sep)

        result = known_system_test(sys_name, **kwargs)
        all_results[label] = result
        spec = result["spectrum"]
        ci = result["spectrum_ci"]

        if verbose:
            pairs = "  ".join(
                (f"{l:+.4f}±{c:.4f}" if np.isfinite(l) else f"[collapsed]±{c:.4f}")
                for l, c in zip(spec, ci)
            )
            ky_dim = result["ky_dim"]
            ky_dim_ci = result["ky_dim_ci"]
            n_cycles = result["n_cycles"]
            converged = result["converged"]
            ky_str = (f"{ky_dim:.3f} ± {ky_dim_ci:.3f}"
                      if ky_dim is not None else "undefined")
            probe = result["lambda_1_probe"]
            probe_str = f"{probe:.4f}" if probe is not None else "n/a (manual)"
            print(f"  Spectrum:  {pairs}")
            print(f"  KY dim:    {ky_str}")
            print(f"  Cycles:    {n_cycles}  converged={converged}")
            print(f"  λ₁ probe:  {probe_str}  "
                  f"cycle_length={result['cycle_length']}  "
                  f"breakin={result['breakin_cycles']}")

            ss_key = ss_key_map.get(label)
            if summary and ss_key and ss_key in summary:
                row_ss = summary[ss_key]
                ref = []
                for i in range(1, 11):
                    v = row_ss.get(f"l{i}", "").strip()
                    if v:
                        try:
                            ref.append(float(v))
                        except ValueError:
                            break
                if ref:
                    ref_str = [f"{x:+.4f}" for x in ref]
                    print(f"  summary_stats ref: {ref_str}")
                    print("  (parameters differ — see note below)")

        ky = result["ky_dim"]
        row_out = {
            "system": label,
            "h": result["h"],
            "ky_dim": round(ky, 4) if ky is not None else None,
            "n_cycles": result["n_cycles"],
        }
        for i, (lam, c) in enumerate(zip(spec, ci)):
            row_out[f"l{i+1}"] = round(float(lam), 6)
            row_out[f"l{i+1}_ci"] = round(float(c), 6)
        rows.append(row_out)

    import csv

    if rows:
        max_k = max(
            sum(1 for key in r if key.startswith("l") and "_" not in key) for r in rows
        )
        fieldnames = ["system", "h", "ky_dim", "n_cycles"]
        for i in range(1, max_k + 1):
            fieldnames += [f"l{i}", f"l{i}_ci"]

        with open(_TRUE_VALUES_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        if verbose:
            print(f"\nWritten: {_TRUE_VALUES_CSV}")

    return all_results


# ---------------------------------------------------------------------------
# Item-5 infrastructure: ESN builder, caching, valid-prediction-time check
# ---------------------------------------------------------------------------

# Parent project root: Reservoir/  (4 levels up from tests/lyapunov/)
_REPO_ROOT_5 = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
_FULL_LOSSES_DIR = os.path.join(
    _REPO_ROOT_5, "reservoir", "results",
    "4_24_26_a_esn2221_multireg_main", "full_losses",
)
_MODEL_CACHE_DIR = os.path.join(_THIS_DIR, "model_cache")
_DATA_CACHE_DIR = os.path.join(_THIS_DIR, "data_cache")

# Per-system configuration for data generation and HP lookup.
# subsample_t       : keep every nth *sampled* step (matches the /results CSV's
#                     "subsample t" column, i.e. the effective downsampling after
#                     the IC generator)
# stepper_substeps  : extra integration steps per sampled point, used when the
#                     integrator runs at a finer dt than the standard sampling
#                     interval.  Omit (defaults to 1) when stepper dt already
#                     equals the standard sampling dt.  The total integrator steps
#                     per collected sample = subsample_t × stepper_substeps, and
#                     the effective physical t_step = stepper.t_step × that product.
#                     Only Mackey-Glass is non-trivial: integrates at dt=0.1 but
#                     samples at dt=1.0, so stepper_substeps=10.
# select_dims       : list of state indices to use as the RC input/output; None = all
# losses_file       : filename under _FULL_LOSSES_DIR
_SYSTEM_CONFIGS_5 = {
    "lorenz_x": {
        "class": "Lorenz",
        "subsample_t": 3,
        "select_dims": [0],    # x component only
        "losses_file": "lor_x_2221_cr.csv",
        "reg_method":  "ridge",
    },
    "lorenz_f": {
        "class": "Lorenz",
        "subsample_t": 1,
        "select_dims": None,   # all 3 dims
        "losses_file": "lor_f_2221_cr.csv",
        "reg_method":  "ridge",
    },
    "lorenz96": {
        "class": "Lorenz96",
        "subsample_t": 3,
        "select_dims": None,   # all 10 dims
        # cn (noise regularization) yields far better stable results than cr (ridge)
        "losses_file": "lor96_2221_cn_an0.csv",
        "reg_method":  "noise",
    },
    "kuramoto_sivashinsky": {
        "class": "KuramotoSivashinsky",
        "subsample_t": 1,
        "select_dims": None,   # full 128-dim real-space field
        # cn (noise regularization) yields far better stable results than cr (ridge)
        "losses_file": "ks22_2221_cn.csv",
        "reg_method":  "noise",
    },
    "mackey_glass": {
        "class": "MackeyGlass",
        "subsample_t": 1,
        "stepper_substeps": 10,   # integrate at dt=0.1, sample at dt=1.0  (1.0/0.1=10)
        "select_dims": [-1],      # last element of DDE history = observable x[t]
        "losses_file": "mg_2221_cr_an0.csv",
        "reg_method":  "ridge",
    },
}

# ---------------------------------------------------------------------------
# Item-5 constants — shared by run_item5 and (via import) esn_test
# ---------------------------------------------------------------------------

_N_TRAIN   = 20000  # training trajectory length
_N_VAL     = 10000  # validation block length
_N_WINDOWS = 50     # validation windows (sampled with replacement)
_KS        = [4, 16, 64, 256]  # RMSE horizons
_PRED_LEN  = 256    # prediction horizon steps (minimum / plotting default)
_SPINUP    = 200    # open-loop warmup steps (fixed fallback)
#: Ridge β candidates for ridge-regularised systems (mirrors solve_square defaults).
_RIDGE_BETAS = [0.0, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-18]

# Map item-5 system names → item-3 known-system names (for true spectrum lookup).
_ITEM5_TO_ITEM3 = {
    "lorenz_x":             "lorenz",
    "lorenz_f":             "lorenz",
    "lorenz96":             "lorenz96",
    "mackey_glass":         "mackey_glass",
    "kuramoto_sivashinsky": "kuramoto_sivashinsky",
}


def _pred_len_for(hp: dict, t_step: float) -> int:
    """Return a prediction-window length sufficient to measure the expected VT.

    Uses 1.3× the expected valid-time (in steps) as the window, floored at
    ``_PRED_LEN`` so that systems with short VT are unaffected.

    For Mackey-Glass (expected VT = 715 steps at t_step = 1.0) this gives
    ``ceil(1.3 × 715) = 930``, so windows are ``_SPINUP + 930 = 1130`` steps
    — short enough to sample easily from ``_N_VAL = 10000``.

    Parameters
    ----------
    hp : dict
        Hyperparameter dict with key ``"vt_0_2_std_med"`` (expected VT in
        physical time units) and ``"time_step"`` (not used here; ``t_step``
        is passed explicitly).
    t_step : float
        Physical time per step.
    """
    vt_steps = hp["vt_0_2_std_med"] / t_step  # expected VT in steps
    return max(_PRED_LEN, int(np.ceil(1.3 * vt_steps)))


def _analytic_spinup(hp: dict, train_data: np.ndarray,
                     arch_components: bool = False) -> dict:
    """Estimate analytic synchronisation time and recommended spinup for *hp*.

    Mirrors ``architectures.py:1683–1708``: runs the reservoir on *train_data*,
    estimates the leading per-step contraction eigenvalue via
    ``architectures.analytic_synchrony``, then converts to a recommended spinup:

    .. code-block::

        rec_spin = sync_time × (log2(√size) − log2(ε))   [mach_eps_power=0]
        spinup   = min(max(100, rec_spin), 1000)

    Parameters
    ----------
    hp : dict
        Hyperparameter dict (as returned by ``_load_best_hp``).
    train_data : np.ndarray, shape (N, d)
        Raw (un-normalised) training trajectory.  Used to normalise inputs and
        warm up the reservoir.
    arch_components : bool, default False
        When True, use ``architectures.basic_res`` / ``basic_in`` weight
        matrices (matching the framework's exact matrices).  When False, use
        ``build_esn``'s uniform[−1,1] matrices.

    Returns
    -------
    dict with keys: ``contraction``, ``sync_time``, ``rec_spin``, ``spinup``.
    """
    from scipy import sparse as _sparse  # noqa: PLC0415
    _reservoir_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../reservoir")
    )
    if _reservoir_dir not in sys.path:
        sys.path.insert(0, _reservoir_dir)
    import architectures as _arch  # noqa: PLC0415

    d_in = train_data.shape[1]
    x_mean = train_data[:-1].mean(axis=0)
    x_std  = np.maximum(train_data[:-1].std(axis=0), 1e-8)
    x_norm = (train_data[:-1] - x_mean) / x_std
    y_norm = (train_data[1:]  - x_mean) / x_std

    # Build and initialise reservoir (fit() also initialises node shapes).
    model = build_esn(hp, d_in, len(x_norm))
    if arch_components:
        W_arr, Win_arr, bias_arr = _arch_components(hp, d_in)
        model.reservoir.W    = W_arr
        model.reservoir.Win  = Win_arr
        model.reservoir.bias = bias_arr
    model.fit(x_norm, y_norm)          # readout result discarded; just initialises

    # Collect reservoir states on a clean run.
    model.reservoir.reset()
    R = model.reservoir.run(x_norm)    # (T, size)

    leak    = float(model.reservoir.lr)
    size    = int(hp["size"])
    w_res_sp = _sparse.csr_matrix(model.reservoir.W)
    rng_sync = np.random.default_rng(42)
    contraction = _arch.analytic_synchrony(
        R[-100:], leak, w_res_sp, samples=20, rng=rng_sync)

    if np.isfinite(contraction) and contraction < 1.0:
        sync_time = -1.0 / np.log2(contraction)
        eps_term  = -np.log2(np.finfo(float).eps)   # ≈ 52.0  (mach_eps_power=0)
        size_term =  np.log2(np.sqrt(size))
        rec_spin  = min(int(sync_time * (size_term + eps_term)) + 1, 1000)
    else:
        sync_time = np.inf
        rec_spin  = 1000
    spinup = max(100, rec_spin)

    return {"contraction": contraction, "sync_time": sync_time,
            "rec_spin": rec_spin, "spinup": spinup}


def _spinup_for(hp: dict, arch_components: bool = False) -> int:
    """Return the saved analytic spinup matching the reservoir builder in use.

    Reads ``hp["spinup_arch"]`` when *arch_components* is True (architectures
    basic_res/basic_in matrices), ``hp["spinup"]`` otherwise (default
    reservoirpy uniform matrices).  Falls back to ``_SPINUP`` if the key is
    absent (old CSV without ``spinup`` key or key = 200).
    """
    if arch_components and "spinup_arch" in hp:
        return int(hp["spinup_arch"])
    return int(hp.get("spinup", _SPINUP))


def _system_factory(cfg: dict):
    """Instantiate the _KnownSystem described by *cfg*."""
    cls_name = cfg["class"]
    cls_map = {
        "Lorenz": Lorenz,
        "Lorenz96": lambda: Lorenz96(N=10),
        "KuramotoSivashinsky": KuramotoSivashinsky,
        "MackeyGlass": MackeyGlass,
    }
    return cls_map[cls_name]()


def generate_train_val(
    system_name: str,
    n_train: int = 20000,
    n_val: int = 10000,
) -> tuple:
    """Generate (or load from cache) training data and a contiguous validation block.

    Returns
    -------
    (train_data, val_block, t_step)
    train_data : np.ndarray of shape (n_train, n_dims)
    val_block  : np.ndarray of shape (n_val, n_dims)
    t_step     : effective physical timestep (system t_step × subsample_t)
    """
    tag = f"{system_name}_nt{n_train}_nv{n_val}"
    os.makedirs(_DATA_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_DATA_CACHE_DIR, f"{tag}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        train_data = data["train_data"]
        val_block = data["val_block"]
        t_step = float(data["t_step"])
        print(f"  [data cache] loaded {os.path.basename(cache_path)}")
        return train_data, val_block, t_step

    cfg = _SYSTEM_CONFIGS_5[system_name]
    sub_t    = cfg["subsample_t"]
    substeps = cfg.get("stepper_substeps", 1)   # extra integration steps per sample
    gen_sub  = sub_t * substeps                  # total integrator steps per collected pt
    sel      = cfg["select_dims"]

    sys = _system_factory(cfg)
    ic_cache = _load_cache()
    row = ic_cache.get(sys.cache_key)
    if row is not None and row["ic"] is not None:
        sys.state = row["ic"].copy()
    else:
        sys.state, sys.stdev = sys.initial_state()

    def _step_and_collect(n_subsampled):
        out = []
        for _ in range(n_subsampled):
            sys.evolve(gen_sub)
            raw = np.asarray(sys.state, float).ravel()
            if sel is not None:
                raw = raw[sel]
            out.append(raw)
        return np.stack(out, axis=0)

    train_data = _step_and_collect(n_train)
    val_block = _step_and_collect(n_val)

    t_step = sys.t_step * gen_sub
    np.savez(cache_path, train_data=train_data, val_block=val_block, t_step=t_step)
    print(f"  [data cache] saved  {os.path.basename(cache_path)}")
    return train_data, val_block, t_step


def sample_val_windows(
    val_block: np.ndarray,
    n: int = 50,
    spinup: int = 1000,
    pred_len: int = 256,
    seed: Optional[int] = None,
) -> list:
    """Sample *n* windows (with replacement) from *val_block*.

    Each window has shape ``(spinup + pred_len, n_dims)``.  Start indices are
    drawn uniformly at random (with replacement) from the set of valid starts
    ``[0, len(val_block) - (spinup + pred_len)]``.

    Parameters
    ----------
    val_block : np.ndarray of shape (T, d)
        Contiguous validation block.
    n : int
        Number of windows to sample.
    spinup : int
        Open-loop warmup steps (prepended to each window).
    pred_len : int
        Prediction horizon steps (follow the spinup in each window).
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    list of np.ndarray, each of shape (spinup + pred_len, d)
    """
    win_len = spinup + pred_len
    n_valid = len(val_block) - win_len
    if n_valid <= 0:
        raise ValueError(
            f"val_block length {len(val_block)} is too short for "
            f"spinup={spinup} + pred_len={pred_len}."
        )
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_valid + 1, size=n)
    return [val_block[s: s + win_len] for s in starts]


def _load_best_hp(system_name: str) -> dict:
    """Read the best-VT row from the *_cr* full_losses CSV.

    Selects the row with the highest ``prediction valid time 0.2 std med``
    (median VPT at the 0.2×std threshold), which is the primary comparison
    target for item-5 verification.

    Returns a dict with keys: size, seed, leak, radius, in_scale, bias_scale,
    sparse, subsample_t, ridge, vt_0_2_std_med, l2_gmean_4, l2_gmean_16,
    l2_gmean_64, l2_gmean_256.
    """
    import pandas as pd

    cfg = _SYSTEM_CONFIGS_5[system_name]
    path = os.path.join(_FULL_LOSSES_DIR, cfg["losses_file"])
    df = pd.read_csv(path)
    _l2_cols = [
        "prediction 4 step L2 error gmean",
        "prediction 16 step L2 error gmean",
        "prediction 64 step L2 error gmean",
        "prediction 256 step L2 error gmean",
    ]
    df = df[(df[_l2_cols] < 2).all(axis=1)]
    if df.empty:
        raise ValueError(f"No rows pass the l2<2 stability filter for {system_name!r}.")
    best = df.loc[df["prediction valid time 0.2 std med"].idxmax()]

    # Parse reg key — format is "<method>=<value>", e.g. "ridge=1e-12" or "noise=0.01".
    reg_name, reg_str = best["reg key"].split("=")
    reg_val = float(reg_str)
    method = cfg["reg_method"]
    if reg_name != method:
        raise ValueError(
            f"reg key method {reg_name!r} does not match reg_method {method!r} "
            f"for {system_name!r}."
        )
    ridge     = reg_val if method == "ridge" else 0.0
    noise_reg = reg_val if method == "noise" else 0.0

    # Verify subsample_t matches the system config
    csv_sub_t = int(best["subsample t"])
    expected_sub_t = cfg["subsample_t"]
    if csv_sub_t != expected_sub_t:
        raise ValueError(
            f"subsample t mismatch for {system_name!r}: "
            f"CSV has {csv_sub_t}, _SYSTEM_CONFIGS_5 has {expected_sub_t}."
        )

    return {
        "size": int(best["size"]),
        "seed": int(best.get("seed", 0)),
        "leak": float(best["leak"]),
        "radius": float(best["radius"]),
        "in_scale": float(best["in scale"]),
        "bias_scale": float(best["bias scale"]),
        "sparse": float(best["sparse"]),
        "subsample_t": csv_sub_t,
        "ridge":     ridge,
        "noise_reg": noise_reg,
        "vt_0_2_std_med": float(best["prediction valid time 0.2 std med"]),
        "l2_gmean_4":  float(best["prediction 4 step L2 error gmean"]),
        "l2_gmean_16": float(best["prediction 16 step L2 error gmean"]),
        "l2_gmean_64": float(best["prediction 64 step L2 error gmean"]),
        "l2_gmean_256": float(best["prediction 256 step L2 error gmean"]),
    }


def _load_reg_sweep(system_name: str) -> list:
    """Return one entry per reg_key row that matches the best-HP config.

    The best HP row is identified by the same max-VPT + l2<2 criterion as
    ``_load_best_hp``.  All rows sharing the same (size, seed, leak, radius,
    in_scale, bias_scale, sparse) — across **all** reg values in the CSV, not
    just the l2-filtered ones — are collected so the full regularization sweep
    is visible in task_e.

    Returns
    -------
    list of dict, sorted by reg_value, each with keys:
        reg_value, vt_0_2_std_med, l2_gmean_4, l2_gmean_16, l2_gmean_64, l2_gmean_256
    """
    import pandas as pd

    cfg = _SYSTEM_CONFIGS_5[system_name]
    path = os.path.join(_FULL_LOSSES_DIR, cfg["losses_file"])
    df_full = pd.read_csv(path)

    # Identify the best row using the l2<2 stability filter (same as _load_best_hp).
    _l2_cols = [
        "prediction 4 step L2 error gmean",
        "prediction 16 step L2 error gmean",
        "prediction 64 step L2 error gmean",
        "prediction 256 step L2 error gmean",
    ]
    df_stable = df_full[(df_full[_l2_cols] < 2).all(axis=1)]
    if df_stable.empty:
        raise ValueError(f"No rows pass the l2<2 stability filter for {system_name!r}.")
    best = df_stable.loc[df_stable["prediction valid time 0.2 std med"].idxmax()]

    # Collect all reg-sweep rows for that HP config from the *full* (unfiltered) CSV
    # so the sweep covers every reg value, not just the stable ones.
    match_cols = ["size", "seed", "leak", "radius", "in scale", "bias scale", "sparse"]
    mask = np.ones(len(df_full), dtype=bool)
    for col in match_cols:
        mask &= df_full[col].values == best[col]
    sweep_rows = df_full[mask].copy()

    results = []
    for _, row in sweep_rows.iterrows():
        reg_val = float(row["reg key"].split("=")[1])
        results.append({
            "reg_value":     reg_val,
            "vt_0_2_std_med": float(row["prediction valid time 0.2 std med"]),
            "l2_gmean_4":  float(row["prediction 4 step L2 error gmean"]),
            "l2_gmean_16": float(row["prediction 16 step L2 error gmean"]),
            "l2_gmean_64": float(row["prediction 64 step L2 error gmean"]),
            "l2_gmean_256": float(row["prediction 256 step L2 error gmean"]),
        })
    results.sort(key=lambda r: r["reg_value"])
    return results


def _hp_csv_path(system_name: str) -> str:
    """Path to the per-system best-HP CSV (written by run_ridge_sweep / task_b_hp)."""
    return os.path.join(_THIS_DIR, f"{system_name}_best_hp.csv")


def _load_hp_csv(system_name: str) -> dict:
    """Load the HP dict written by ``run_ridge_sweep`` / ``task_b_hp``.

    Raises ``FileNotFoundError`` if the CSV is absent (run
    ``esn_test.run_ridge_sweep`` first).
    """
    import csv  # noqa: PLC0415
    path = _hp_csv_path(system_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"HP CSV not found at {path!r}. Run esn_test.run_ridge_sweep() first."
        )
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    int_keys = {"size", "seed", "spinup", "spinup_arch", "subsample_t"}
    float_keys = {
        "time_step", "leak", "radius", "in_scale", "bias_scale",
        "sparse", "ridge", "noise_reg", "vt_0_2_std_med",
        "l2_gmean_4", "l2_gmean_16", "l2_gmean_64", "l2_gmean_256",
    }
    hp = {}
    for k, v in row.items():
        if k in int_keys:
            hp[k] = int(v)
        elif k in float_keys:
            hp[k] = float(v)
        else:
            hp[k] = v
    return hp


def _arch_components(hp: dict, input_dim: int):
    """Return (W, Win, bias) built by ``architectures.basic_res`` / ``basic_in``.

    Lazy-imports ``architectures`` from ``../../../../reservoir`` (the framework
    directory) so the dependency is only paid when this function is called.

    Parameters
    ----------
    hp : dict
        Hyperparameter dict with keys: ``size``, ``seed``, ``radius``, ``sparse``,
        ``in_scale``, ``bias_scale``.
    input_dim : int
        Number of input dimensions (columns in *x*).

    Returns
    -------
    W : ndarray, shape (size, size)
        Dense reservoir matrix, spectral radius scaled to ``hp['radius']``.
    Win : ndarray, shape (size, input_dim)
        Dense input weight matrix (in_scale applied).
    bias : ndarray, shape (size,)
        Dense bias vector (bias_scale applied) — the last column of ``basic_in``.
    """
    _reservoir_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../reservoir")
    )
    if _reservoir_dir not in sys.path:
        sys.path.insert(0, _reservoir_dir)

    import architectures as _arch  # noqa: PLC0415 (lazy import by design)

    p = {
        "size":       hp["size"],
        "seed":       hp["seed"],
        "radius":     hp["radius"],
        "sparse":     hp["sparse"],
        "in scale":   hp["in_scale"],
        "bias scale": hp["bias_scale"],
        "d in":       input_dim,
        "w in bias":  True,
        "sparse in":  True,
    }
    W = _arch.basic_res(p).toarray()
    M = _arch.basic_in(p).toarray()
    return W, M[:, :input_dim], M[:, input_dim]


def build_esn(hp: dict, input_dim: int, n_train_samples: int):
    """Construct (but do not train) a reservoirpy ESN matching *hp*.

    Uses input_connectivity = 1/input_dim, reservoir connectivity from
    hp["sparse"], and Uniform[−1,1] distributions for Win, W, and bias
    (matching architectures.basic_in / basic_res).
    input_to_readout=True so that the readout receives [reservoir_state,
    input] as features — matching the TradRC setup.

    Ridge convention: the framework's classic_ridge (4/24/26) pre-scaled the
    training data by 1/√T before solving, i.e. it regularised the *averaged*
    Gram XᵀX/T with β·I — equivalent to penalty β·T on the *summed* Gram.
    reservoirpy's Ridge regularises the summed Gram XᵀX directly, so β is
    scaled by ``n_train_samples`` (the training length T) here to reproduce the
    same effective regularisation, letting the framework's CSV ridge
    hyperparameters translate directly.

    When using noise regularisation (hp["noise_reg"] > 0), the noise acts as
    an implicit regulariser on XXT, so hp["ridge"] is set to 0.  The edge case
    of both ridge=0 and noise_reg=0 (the noise=0 sweep point) is handled by a
    tiny ridge floor to avoid a singular solve.
    """
    from reservoirpy import ESN
    from reservoirpy.nodes import Reservoir, Ridge
    from reservoirpy.mat_gen import uniform

    ridge_beta = float(hp.get("ridge", 0.0))
    noise_reg  = float(hp.get("noise_reg", 0.0))
    # Ridge floor: prevent singular solve when both regularisers are zero.
    # if ridge_beta == 0.0 and noise_reg == 0.0:
    #     ridge_beta = 1e-10

    # In the framework, p['sparse'] is connections per row, so connectivity = sparse/size.
    rc_conn = hp["sparse"] / hp["size"]
    reservoir = Reservoir(
        units=hp["size"],
        sr=hp["radius"],
        lr=hp["leak"],
        input_scaling=hp["in_scale"],
        input_connectivity=1.0 / input_dim,
        rc_connectivity=rc_conn,
        W=uniform,
        Win=uniform,
        bias=uniform(input_scaling=hp["bias_scale"]),
        seed=hp["seed"],
    )
    # β·T to match classic_ridge's averaged-Gram convention (see docstring).
    readout = Ridge(ridge=ridge_beta * n_train_samples, fit_bias=True) #
    return ESN(reservoir=reservoir, readout=readout, input_to_readout=True)


def _esn_cache_path(system_name: str, hp: dict) -> str:
    """Return pickle path for a cached (model, norm) tuple.

    Suffix ``_n1r1z1u1``:
      ``n1`` — std-normalised training data (mean 0 std 1)
      ``r1`` — Ridge β scaled by training length (β·T), matching classic_ridge's
               averaged-Gram convention; supersedes the old ``r0`` (raw β) tag
      ``z1`` — noise_reg included in tag (distinguishes noise vs ridge variants)
      ``u1`` — uniform[−1,1] Win/W/bias distributions
    """
    noise_reg = float(hp.get("noise_reg", 0.0))
    tag = (f"{system_name}_sz{hp['size']}_sr{hp['radius']:.4f}"
           f"_lr{hp['leak']:.4f}_is{hp['in_scale']:.4f}"
           f"_bs{hp['bias_scale']:.4f}_rg{hp['ridge']:.2e}"
           f"_nz{noise_reg:.2e}_sd{hp['seed']}_n1r1z1u1")
    os.makedirs(_MODEL_CACHE_DIR, exist_ok=True)
    return os.path.join(_MODEL_CACHE_DIR, f"{tag}.pkl")


def load_or_train_esn(
    system_name: str,
    hp: dict,
    train_data: np.ndarray,
    spinup: Optional[int] = None,
    cache: bool = True,
):
    """Return ``(model, norm)`` — a trained ESN and its normalization stats.

    In ``architectures.py`` the readout receives std-normalised input
    (``(u - input_means) / input_stds``) alongside raw reservoir states, and
    the target is also std-normalised.  We replicate this by normalising all
    training (and later validation) data with the training mean/std before
    feeding the model, which preserves the effective ridge regularisation
    scale relative to the feature magnitudes.

    ``norm`` is a dict ``{"mean": x_mean, "std": x_std}`` where the arrays
    have shape ``(input_dim,)``.  Apply to any new data as
    ``(x - norm["mean"]) / norm["std"]`` before passing to the model.

    Parameters
    ----------
    spinup : int or None
        Number of warmup steps passed to ``model.fit(..., warmup=spinup)``.
        When None or 0 no warmup is applied.  Mirrors the *spinup* argument of
        ``_setup_esn`` in ``esn_test.py``.
    cache : bool, default True
        When False, skip both cache-load and cache-save — always train fresh.
        Useful for sweep loops that vary hp across calls (same cache key would
        alias to the same pickle file).
    """
    cache_path = _esn_cache_path(system_name, hp)
    if cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        print(f"  [cache] loaded {os.path.basename(cache_path)}")
        return cached["model"], cached["norm"]

    # Compute normalization from clean training inputs; clip std away from zero.
    x_mean = train_data[:-1].mean(axis=0)
    x_std  = np.maximum(train_data[:-1].std(axis=0), 1e-8)
    norm = {"mean": x_mean, "std": x_std}

    # Normalize the full trajectory, then optionally add input noise.
    # Noise is added AFTER normalization (std=noise_reg in normalized units),
    # matching the framework's 'noise' regression which perturbs the already-
    # normalized reservoir inputs and targets consistently.
    # Validation data is never noised — norm stats are from the clean trajectory.
    traj_norm = (train_data - x_mean) / x_std
    noise_reg = float(hp.get("noise_reg", 0.0))
    if noise_reg > 0.0:
        rng = np.random.default_rng(int(hp["seed"]))
        traj_norm = traj_norm + noise_reg * rng.standard_normal(traj_norm.shape)

    x_norm = traj_norm[:-1]
    y_norm = traj_norm[1:]

    input_dim = train_data.shape[1]
    model = build_esn(hp, input_dim, len(x_norm))

    # Explicit reset: guarantee the reservoir starts training from r=0 instead of
    # relying on initialize()'s implicit zeroing.  Model.fit() does NOT reset node
    # state (reservoirpy/model.py:519), so this makes the training pass robust to
    # any reuse / re-fit of the model object.  initialize() must run first to create
    # the state dict; fit() will skip re-initialization via its own guard.
    if not model.initialized:
        model.initialize(x_norm, y_norm)
    model.reset()

    model.fit(x_norm, y_norm, warmup=(spinup or 0))

    if cache:
        with open(cache_path, "wb") as f:
            pickle.dump({"model": model, "norm": norm}, f)
        print(f"  [cache] saved  {os.path.basename(cache_path)}")
    return model, norm


def verify_valid_time(
    system_name: str,
    n_train: int = 20000,
    n_val: int = 10000,
    n_windows: int = 50,
    spinup: int = 1000,
    pred_len: int = 256,
    verbose: bool = True,
) -> dict:
    """Train an ESN with best-*_cr* HPs and measure valid prediction time.

    Compares the reproduced median VPT at 0.2×std against the saved value
    (``prediction valid time 0.2 std med``) from the original framework.
    All training and validation data are std-normalised before the model sees
    them, matching the per-feature normalisation in ``architectures.py``.
    """
    hp = _load_best_hp(system_name)
    if verbose:
        print(f"\n{'='*60}")
        print(f"  verify_valid_time: {system_name}")
        print(f"  size={hp['size']}  sr={hp['radius']:.4f}  lr={hp['leak']:.4f}")
        print(f"  in_scale={hp['in_scale']:.4f}  bias_scale={hp['bias_scale']:.4f}")
        print(f"  sparse={hp['sparse']:.4f}  ridge={hp['ridge']:.2e}  seed={hp['seed']}")
        print(f"  saved vt_0_2_std_med={hp['vt_0_2_std_med']:.4f}")
        print(f"{'='*60}")

    train_data, val_block, t_step = generate_train_val(
        system_name, n_train=n_train, n_val=n_val,
    )
    if verbose:
        print(f"  data: train={train_data.shape}  val_block={val_block.shape}"
              f"  t_step={t_step:.4f}")

    model, norm = load_or_train_esn(system_name, hp, train_data)

    # Sample 50 windows with replacement and normalise with training statistics.
    windows = sample_val_windows(val_block, n=n_windows, spinup=spinup, pred_len=pred_len)
    windows_norm = [(w - norm["mean"]) / norm["std"] for w in windows]

    results = valid_time_multitest(
        model, windows_norm, n_segments=n_windows,
        spinup=spinup, eps=[0.2, 0.4], ks=[4, 16, 64, 256],
        block=pred_len, t_step=t_step,
    )

    reproduced = results.get("valid_time_0.2_median", float("nan"))
    saved = hp["vt_0_2_std_med"]
    if verbose:
        print(f"\n  Results:")
        print(f"    valid_time_0.2_median  = {reproduced:.4f}  (saved: {saved:.4f})")
        print(f"    valid_time_0.4_median  = {results.get('valid_time_0.4_median', float('nan')):.4f}")
        print(f"    valid_time_fitness     = {results.get('valid_time_fitness', float('nan')):.4f}")
        for k in [4, 16, 64, 256]:
            obs = results.get(f"rmse_{k}_gmean", float("nan"))
            exp = hp.get(f"l2_gmean_{k}", float("nan"))
            print(f"    rmse_{k:3d}_gmean = {obs:.4f}  (saved: {exp:.4f})")
        ratio = reproduced / saved if saved > 0 else float("nan")
        print(f"    ratio (reprod/saved)   = {ratio:.3f}")

    return {
        "system": system_name,
        "hp": hp,
        "norm": norm,
        "results": results,
        "vt_0_2_median_reproduced": reproduced,
        "vt_0_2_median_saved": saved,
        "t_step": t_step,
    }


def run_item5(systems=None, verbose: bool = True) -> list:
    """Verify VPT for the swept-best HPs loaded from each system's HP CSV.

    Requires that ``esn_test.run_ridge_sweep`` has already been run and
    written ``{system}_best_hp.csv`` for each system.  Reads those HPs
    (including the swept-best regularisation value and analytic spinup),
    trains a single ESN per system with ``warmup=spinup``, and reports VPT
    vs. the baseline.

    Results match the best-beta row of ``esn_test.run_ridge_sweep`` because
    the two paths use identical HPs, spinup, window seed, and pred_len.

    Parameters
    ----------
    systems : list of str or None
        System names from ``_SYSTEM_CONFIGS_5``.  ``None`` runs all five.
    verbose : bool
        When True, print per-system details and the final summary table.

    Returns
    -------
    list of dict, one per system, with keys:
        ``system``, ``reg_method``, ``best_reg``, ``best_vt``,
        ``baseline_vt``, ``ratio``.
    """
    if systems is None:
        systems = list(_SYSTEM_CONFIGS_5.keys())

    all_rows = []

    for sys_name in systems:
        reg_method = _SYSTEM_CONFIGS_5[sys_name]["reg_method"]

        if verbose:
            print(f"\n{'='*60}")
            print(f"  run_item5: {sys_name}  (reg_method={reg_method})")
            print(f"{'='*60}")

        # 1. Swept HPs from HP CSV (written by run_ridge_sweep / task_b_hp).
        hp = _load_hp_csv(sys_name)
        best_reg = hp["noise_reg"] if reg_method == "noise" else hp["ridge"]
        spinup = _spinup_for(hp, arch_components=False)

        if verbose:
            print(f"  spinup={spinup}  ridge={hp['ridge']:.2e}  "
                  f"noise_reg={hp['noise_reg']:.2e}  "
                  f"(from {sys_name}_best_hp.csv)")

        # 2. Train + val data and physical t_step.
        train_data, val_block, t_step = generate_train_val(
            sys_name, n_train=_N_TRAIN, n_val=_N_VAL,
        )

        # 3. Prediction window length.
        pred_len = _pred_len_for(hp, t_step)

        # 4. Validation windows — seed=1 matches run_ridge_sweep.
        windows = sample_val_windows(
            val_block, n=_N_WINDOWS, spinup=spinup, pred_len=pred_len, seed=1,
        )

        # 5. Train once (no cache) and score.
        model, norm = load_or_train_esn(
            sys_name, hp, train_data, spinup=spinup, cache=False,
        )
        windows_norm = [(w - norm["mean"]) / norm["std"] for w in windows]
        res = valid_time_multitest(
            model, windows_norm, n_segments=_N_WINDOWS,
            spinup=spinup, eps=[0.2, 0.4], ks=_KS,
            block=pred_len, t_step=t_step,
        )
        best_vt = res.get("valid_time_0.2_median", float("nan"))

        # 6. Ratio vs baseline.
        baseline_vt = hp["vt_0_2_std_med"]
        ratio = (best_vt / baseline_vt
                 if baseline_vt > 0 and np.isfinite(best_vt) else float("nan"))

        if verbose:
            print(f"  best_vt={best_vt:.4f}  baseline_vt={baseline_vt:.4f}  "
                  f"ratio={ratio:.3f}")

        all_rows.append({
            "system":      sys_name,
            "reg_method":  reg_method,
            "best_reg":    best_reg,
            "best_vt":     best_vt,
            "baseline_vt": baseline_vt,
            "ratio":       ratio,
        })

    if verbose:
        sep = "=" * 70
        print(f"\n{sep}")
        print("  Item-5 summary")
        print(sep)
        print(f"  {'System':<28} {'Reg':>6} {'Best reg':>12} {'VT':>10} {'Baseline':>10} {'Ratio':>7}")
        print(f"  {'-'*28} {'-'*6} {'-'*12} {'-'*10} {'-'*10} {'-'*7}")
        for r in all_rows:
            br_str = f"{r['best_reg']:.2e}" if r["best_reg"] > 0 else "0"
            bvt_str = f"{r['best_vt']:.4f}" if np.isfinite(r["best_vt"]) else "nan"
            print(f"  {r['system']:<28} {r['reg_method']:>6} {br_str:>12} "
                  f"{bvt_str:>10} {r['baseline_vt']:>10.4f} {r['ratio']:>7.3f}")
        print(sep)

    return all_rows


def assess_lyapunov(
    system_name: str,
    k: Optional[int] = None,
    min_cycles: int = 500,
    max_cycles: int = 2000,
    rtol: float = 0.01,
    display: bool = False,
    progress_bar: bool = True,
    verbose: bool = True,
) -> dict:
    """Train an ESN on *system_name* and compare its Lyapunov spectrum to the true system.

    Workflow:
    1. Load HPs from ``{system_name}_best_hp.csv`` (written by
       ``esn_test.run_ridge_sweep``).
    2. Generate / load train data; train ESN via ``load_or_train_esn``.
    3. Compute the true spectrum via ``known_system_test`` (result is cached in
       ``attractor_cache.csv`` after the first run).
    4. Compute the ESN spectrum via ``reservoirpy.observables.lyapunov``.
       Internally ``_RCAdapter.t_step = 1.0`` so raw exponents are in per-step
       units; divide by ``t_step`` to convert to physical-time units matching
       the true spectrum.
    5. Print a per-exponent comparison table and KY dimension.

    Parameters
    ----------
    system_name : str
        Key into ``_SYSTEM_CONFIGS_5`` (e.g. ``"lorenz_x"``).
    k : int or None
        Number of Lyapunov exponents for the ESN.  Defaults to
        ``min(6, true_sys.k_default)``: 3 (Lorenz), 5 (Mackey-Glass),
        6 (Lorenz96 / KS).  The true-system count is additionally capped at
        ``true_sys.k_default`` (``true_k`` in the return dict).
    min_cycles, max_cycles, rtol
        Forwarded to ``_lyapunov`` for both the true system and the ESN.
    display : bool
        Show convergence plots (blocks until window closed).
    progress_bar : bool, default True
        Show a plain-stdout progress bar during breakin and accumulation.
        When ``verbose=True`` (the default) each mark is a status line;
        otherwise a ruler ``____`` / mark ``||||`` pair is printed.
    verbose : bool
        Print per-exponent table and KY dimension.

    Returns
    -------
    dict with keys:
        ``system``, ``k``, ``t_step``,
        ``true_spectrum``, ``true_ci``, ``true_ky_dim``,
        ``esn_spectrum``, ``esn_ci``, ``esn_ky_dim``.
    """
    from reservoirpy.observables import lyapunov as _lyap_esn  # noqa: PLC0415

    item3_name = _ITEM5_TO_ITEM3.get(system_name)
    if item3_name is None:
        raise ValueError(
            f"Unknown system {system_name!r}. Known: {sorted(_ITEM5_TO_ITEM3)}"
        )

    true_sys = _SYSTEMS[item3_name]()
    if k is None:
        k = min(6, true_sys.k_default)
    true_k = min(k, true_sys.k_default)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  assess_lyapunov: {system_name}  "
              f"(item3={item3_name}  k_esn={k}  k_true={true_k})")
        print(f"{'='*60}")

    # --- Step 1: load HP and train ESN ---
    hp     = _load_hp_csv(system_name)
    spinup = _spinup_for(hp, arch_components=False)

    if verbose:
        print(f"  spinup={spinup}  ridge={hp['ridge']:.2e}  "
              f"noise_reg={hp['noise_reg']:.2e}")

    train_data, _val, t_step = generate_train_val(
        system_name, n_train=_N_TRAIN, n_val=_N_VAL,
    )
    model, norm = load_or_train_esn(system_name, hp, train_data, spinup=spinup)
    x_norm = (train_data - norm["mean"]) / norm["std"]

    # --- Step 2: true system Lyapunov spectrum ---
    if verbose:
        print(f"\n  [true] known_system_test({item3_name!r}, k={true_k}) ...")
    true_result = known_system_test(
        item3_name, k=true_k,
        min_cycles=min_cycles, max_cycles=max_cycles,
        rtol=rtol, display=display,
    )
    true_spec = np.asarray(true_result["spectrum"])
    true_ci   = np.asarray(true_result["spectrum_ci"])
    true_ky   = true_result["ky_dim"]

    # --- Step 3: ESN Lyapunov spectrum ---
    # lyapunov() uses _RCAdapter.t_step=1.0, giving exponents in per-step units.
    # Dividing by t_step converts to physical-time units (same scale as true_spec).
    if verbose:
        print(f"  [ESN]  lyapunov(k={k}, spinup={spinup}) ...")
    esn_raw = _lyap_esn(
        model,
        init=x_norm,            # n_spinup=1 default ensures m=1 regardless of array length
        spinup=spinup,
        k=k,
        min_cycles=min_cycles,
        max_cycles=max_cycles,
        rtol=rtol,
        display=display,
        progress_bar=progress_bar,
        progress_verbose=verbose,
        probe_sub_cycle_cap=50,   # ESN state is not a ring buffer; cap the slow probe
    )
    esn_spec = np.asarray(esn_raw["spectrum"]) / t_step
    esn_ci   = np.asarray(esn_raw["spectrum_ci"]) / t_step
    esn_ky   = esn_raw["ky_dim"]   # KY dim is invariant to t_step rescaling

    # --- Step 4: print comparison ---
    if verbose:
        print(f"\n  Spectrum  [physical time, t_step={t_step:.5f}]")
        print(f"  {'':>3}  {'True λ':>10}  {'±CI':>8}  "
              f"{'ESN λ':>10}  {'±CI':>8}  {'|Δλ|':>8}  {'ratio':>7}")
        print(f"  {'─'*3}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*7}")
        for i, (tl, tc, el, ec) in enumerate(zip(true_spec, true_ci, esn_spec, esn_ci)):
            fin = np.isfinite(tl) and np.isfinite(el)
            delta = abs(el - tl) if fin else float("nan")
            ratio = (el / tl) if (fin and tl != 0) else float("nan")
            tl_s = f"{tl:+.4f}" if np.isfinite(tl) else "  [nan]"
            el_s = f"{el:+.4f}" if np.isfinite(el) else "  [nan]"
            print(f"  λ{i+1:<2}  {tl_s:>10}  {tc:>8.4f}  "
                  f"{el_s:>10}  {ec:>8.4f}  {delta:>8.4f}  {ratio:>7.3f}")
        ky_t = f"{true_ky:.3f}" if true_ky is not None else "None"
        ky_e = f"{esn_ky:.3f}"  if esn_ky  is not None else "None"
        print(f"\n  KY dim:  true={ky_t}  ESN={ky_e}")

    return {
        "system":        system_name,
        "k":             k,
        "true_k":        true_k,
        "t_step":        t_step,
        "true_spectrum": true_spec,
        "true_ci":       true_ci,
        "true_ky_dim":   true_ky,
        "esn_spectrum":  esn_spec,
        "esn_ci":        esn_ci,
        "esn_ky_dim":    esn_ky,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
#
# Usage:
#   python lyapunov_test.py                       → run_item3 (Lyapunov, all systems)
#   python lyapunov_test.py lorenz                → known_system_test("lorenz")
#   python lyapunov_test.py item5                 → run_item5 (VPT, all systems)
#   python lyapunov_test.py item5 lorenz_x mg     → run_item5 for named systems

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "item5":
            systems = list(sys.argv[2:]) if len(sys.argv) > 2 else None
            run_item5(systems, verbose=True)
        else:
            result = known_system_test(cmd, display=True)
            spec = result["spectrum"]
            ci = result["spectrum_ci"]
            ky = result["ky_dim"]
            ky_ci = result["ky_dim_ci"]
            nc = result["n_cycles"]
            conv = result["converged"]
            probe = result["lambda_1_probe"]
            ky_str = f"{ky:.4f} ± {ky_ci:.4f}" if ky is not None else "undefined"
            probe_str = f"{probe:.4f}" if probe is not None else "n/a (manual)"
            print(f"\nSpectrum: {spec}")
            print(f"CI:       {ci}")
            print(f"KY dim:   {ky_str}")
            print(f"Cycles:   {nc}  converged={conv}")
            print(f"λ₁ probe: {probe_str}  cycle_length={result['cycle_length']}  breakin={result['breakin_cycles']}")
    else:
        # run_item5(verbose=True)
        assess_lyapunov("lorenz_f")
