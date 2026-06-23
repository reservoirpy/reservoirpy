"""Lyapunov spectrum demonstration suite for known dynamical systems and trained ESNs.

See `LYAPUNOV_README.md` for details.
"""

import ast
import csv
import os
import pickle
import sys
import time
from typing import Optional

# Ensure the local reservoirpy fork takes precedence over any installed copy.
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")))

import numpy as np
import scipy.fft

from reservoirpy import ESN
from reservoirpy.datasets._chaos import _kuramoto_sivashinsky_etdrk4
from reservoirpy.mat_gen import uniform
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import _lyapunov, ky_dim, lyapunov

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_IC_DIR = os.path.join(_THIS_DIR, "initial_conditions")
_KNOWN_SPECS_CSV = os.path.join(_THIS_DIR, "literature_lyapunov_spectra.csv")
_MODEL_CACHE_DIR = os.path.join(_THIS_DIR, "model_cache")
_TRAIN_DATA_CACHE_DIR = os.path.join(_THIS_DIR, "training_data_cache")
_HP_CSV = os.path.join(_THIS_DIR, "hyperparameters.csv")

_N_TRAIN = 20000
_SPINUP = 200


# ---------------------------------------------------------------------------
# KS spectral setup  (mirrors _kuramoto_sivashinsky in _chaos.py)
# ---------------------------------------------------------------------------


def _ks_etdrk4_setup(N: int, M: int, h: float, d: float) -> dict:
    """Precompute ETDRK4 scalars for Kuramoto-Sivashinsky integration.

    Parameters
    ----------
    N : int
        Number of spatial grid points.
    M : int
        Number of Cauchy quadrature points (16 is standard).
    h : float
        Time step.
    d : float
        Domain length L.

    Returns
    -------
    dict
        ETDRK4 scalars ``g``, ``E``, ``E2``, ``Q``, ``f1``, ``f2``, ``f3``.
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
    """Per-system constants and integrator for a known dynamical system.

    Subclasses must implement :meth:`step` and :meth:`initial_state`.
    To pass an instance to :func:`_lyapunov`, set ``self.state = self.x0.copy()``,
    then wrap with :class:`_SystemStepper`.

    Attributes
    ----------
    x0 : np.ndarray
        On-attractor initial condition (post-warmup).  Used as the starting
        state for all normal-path calls; ``initial_state()`` is a regenerator
        utility only.
    stdev : float
        Trajectory standard deviation on the attractor.
    """

    name: str = ""
    formal_name: str = ""
    method: str = ""
    k_default: int = 3
    min_cycles_default: int = 500
    t_step: float = 1.0
    n_warmup: int = 2000
    n_sample: int = 2000
    stdev: float = 1.0
    x0: np.ndarray = np.zeros(1)

    @property
    def cache_key(self) -> str:
        return self.name

    def step(self, state: np.ndarray, n_steps: int) -> np.ndarray:
        raise NotImplementedError

    def initial_state(self):
        raise NotImplementedError


class _SystemStepper:
    """Adapt a :class:`_KnownSystem` to the stepper contract for :func:`_lyapunov`.

    Mirrors the interface of :class:`reservoirpy.observables.ReservoirStepper`
    for baseline ODE/DDE/PDE systems.
    """

    def __init__(self, system: _KnownSystem):
        self._system = system
        self.state = system.state
        self.stdev = system.stdev
        self.t_step = system.t_step

    def run(self, n_steps: int) -> None:
        self.state = self._system.step(self.state, n_steps)


class _ODESystem(_KnownSystem):
    """Autonomous ODE systems integrated with fixed-step RK4."""

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


class Lorenz(_ODESystem):
    name = "lorenz"
    formal_name = "Lorenz63"
    method = "ODE RK4"
    k_default = 3
    stdev = 14.282866422209104
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    t_step = 0.02
    x0 = np.array([-14.534641402422949, -9.213263792189714, 39.81952304782705])

    def rhs(self, s):
        x, y, z = s
        return np.array([self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z])


class Lorenz96(_ODESystem):
    name = "lorenz96"
    formal_name = "Lorenz96 10d"
    method = "ODE RK4"
    k_default = 10
    stdev = 3.622654737513749
    F = 8.0
    t_step = 0.02
    N = 10
    x0 = np.array(
        [
            1.538644901557219,
            2.6612631824750546,
            8.92197688363573,
            -1.4017919582755187,
            -1.5123381827246805,
            0.13260794908701265,
            2.900132447269597,
            5.747863245706213,
            1.7267896821355198,
            -2.798819138645696,
        ]
    )

    def rhs(self, s):
        return (np.roll(s, -1) - np.roll(s, 2)) * np.roll(s, 1) - s + self.F


class MackeyGlass(_KnownSystem):
    name = "mackey_glass"
    formal_name = "Mackey-Glass"
    method = "DDE RK4"
    k_default = 10
    stdev = 0.2252707719024293
    tau, beta, gamma, n_mg = 20.0, 0.2, 0.1, 10
    t_step = 0.1

    def __init__(self):
        self.x0 = np.load(os.path.join(_IC_DIR, "mackey_glass.npy"))

    @property
    def n_delay(self):
        return round(self.tau / self.t_step)

    def step(self, state: np.ndarray, n_steps: int) -> np.ndarray:
        h = self.t_step
        h2, h6 = 0.5 * h, h / 6.0
        beta, gamma, n = self.beta, self.gamma, self.n_mg

        def _rhs(x_now, x_del):
            return beta * x_del / (1.0 + x_del**n) - gamma * x_now

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
        """Regenerator: starts from 0.9·ones(n_delay), not from self.x0."""
        state = self.step(0.9 * np.ones(self.n_delay), self.n_warmup)
        traj = np.empty(self.n_sample)
        for i in range(self.n_sample):
            state = self.step(state, 1)
            traj[i] = state[-1]
        return state, float(np.std(traj))


class KuramotoSivashinsky(_KnownSystem):
    name = "kuramoto_sivashinsky"
    formal_name = "Kuramoto-Sivashinsky"
    method = "PDE ETDRK4"
    k_default = 10
    stdev = 1.1297214457875504
    N, M = 128, 16
    L = 22.0
    t_step = 0.25

    def __init__(self):
        self._scalars = _ks_etdrk4_setup(self.N, self.M, self.t_step, d=self.L)
        self.x0 = np.load(os.path.join(_IC_DIR, "kuramoto_sivashinsky.npy"))

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
# System registry
# ---------------------------------------------------------------------------

_SYSTEMS = {
    "lorenz": Lorenz,
    "lorenz96": Lorenz96,
    "mackey_glass": MackeyGlass,
    "kuramoto_sivashinsky": KuramotoSivashinsky,
}


# ---------------------------------------------------------------------------
# Lyapunov-spectra CSV helpers  (shared by both CSV files)
# ---------------------------------------------------------------------------


def _load_spectra(path: str) -> dict:
    if not os.path.exists(path):
        return {}

    out: dict = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sys_name = row.get("system", "").strip()
            if not sys_name:
                continue
            vals = []
            i = 1
            while f"l{i}" in row and row[f"l{i}"].strip() != "":
                vals.append(float(row[f"l{i}"]))
                i += 1
            entry: dict = {"spectrum": np.array(vals) if vals else np.array([])}
            for col, raw in row.items():
                if col == "system" or (col.startswith("l") and col[1:].isdigit()):
                    continue
                cell = raw.strip()
                if not cell:
                    continue
                try:
                    entry[col] = float(cell)
                except ValueError:
                    entry[col] = cell
            if sys_name not in out:
                out[sys_name] = entry
    return out


# ---------------------------------------------------------------------------
# known_system_test
# ---------------------------------------------------------------------------


def known_system_test(
    system_name: str,
    k: Optional[int] = None,
    cycle_length: Optional[int] = None,
    breakin_cycles: int = 500,
    breakin_groups: int = 3,
    ordering_threshold: float = 0.8,
    min_cycles: Optional[int] = None,
    max_cycles: Optional[int] = None,
    rtol: float = 0.0,
    atol: float = 0.0,
    epsilon: float = 1e-5,
    convergence: str = "ky_dim",
    space: str = "real",
    display: bool = True,
    verbose: bool = True,
) -> dict:
    """Compute the Lyapunov spectrum of a known ODE, PDE, or DDE dynamical
    system, using :func: `observables._lyapunov`
    """
    factory = _SYSTEMS.get(system_name)
    if factory is None:
        raise ValueError(f"Unknown system: {system_name!r}.  Known: {sorted(_SYSTEMS)}")

    model = factory()
    model.state = model.x0.copy()

    k = k if k is not None else model.k_default
    min_cycles = min_cycles if min_cycles is not None else model.min_cycles_default
    max_cycles = max_cycles if max_cycles is not None else 2000

    if verbose:
        _fn = _SYSTEMS[system_name].formal_name or system_name
        _mt = _SYSTEMS[system_name].method or "direct"
        print(f"  [true system lyapunov spectrum] computing {_fn} {_mt}" f" lyapunov spectrum  (k={k})...")

    stepper = _SystemStepper(model)
    result = _lyapunov(
        stepper,
        k=k,
        cycle_length=cycle_length,
        breakin_cycles=breakin_cycles,
        breakin_groups=breakin_groups,
        ordering_threshold=ordering_threshold,
        min_cycles=min_cycles,
        max_cycles=max_cycles,
        rtol=rtol,
        atol=atol,
        epsilon=epsilon,
        convergence=convergence,
        progress_bar=display,
    )
    result["system"] = system_name
    result["h"] = model.t_step

    return result


# ---------------------------------------------------------------------------
# Hyperparameter CSV  (hyperparameters.csv)
# ---------------------------------------------------------------------------

_HP_INT_KEYS = {"size", "seed", "spinup", "subsample_t", "stepper_substeps"}
_HP_FLOAT_KEYS = {"leak", "radius", "in_scale", "bias_scale", "sparse", "ridge", "noise_reg"}


def _load_hp_csv(system_name: str) -> dict:
    path = _HP_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(f"hyperparameters.csv not found at {path!r}.")
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["system"] == system_name:
                hp: dict = {}
                for col, val in row.items():
                    if col == "system":
                        continue
                    elif col == "select_dims":
                        hp[col] = ast.literal_eval(val) if val.strip() else None
                    elif col in _HP_INT_KEYS:
                        hp[col] = int(val)
                    elif col in _HP_FLOAT_KEYS:
                        hp[col] = float(val)
                    else:
                        hp[col] = val
                return hp
    raise KeyError(f"No row for system {system_name!r} in {path!r}.")


# ---------------------------------------------------------------------------
# System factory and data generation
# ---------------------------------------------------------------------------

_CLASS_MAP = {
    "Lorenz": Lorenz,
    "Lorenz96": Lorenz96,
    "MackeyGlass": MackeyGlass,
    "KuramotoSivashinsky": KuramotoSivashinsky,
}


def _system_factory(hp: dict) -> _KnownSystem:
    """Instantiate the ``_KnownSystem`` described by *hp*'s ``class`` field."""
    cls_name = hp["class"]
    cls = _CLASS_MAP.get(cls_name)
    if cls is None:
        raise ValueError(f"Unknown system class {cls_name!r}. Known: {sorted(_CLASS_MAP)}")
    return cls()


def _input_label(hp: dict) -> str:
    cls_name = hp.get("class", "")
    sel = hp.get("select_dims")
    if cls_name == "Lorenz":
        if sel is not None and len(sel) == 1 and sel[0] == 0:
            return "x"
        return "(x, y, z)"
    if cls_name == "Lorenz96":
        return "(x_1, x_2, ..., x_10)"
    if cls_name == "MackeyGlass":
        return "P"
    if cls_name == "KuramotoSivashinsky":
        return "64 point discretization of [0, L)"
    return ""


def generate_train(
    system_name: str,
    n_train: int = _N_TRAIN,
) -> tuple:
    """Generate (or load from cache) training data for *system_name*."""
    tag = f"{system_name}_nt{n_train}"
    os.makedirs(_TRAIN_DATA_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_TRAIN_DATA_CACHE_DIR, f"{tag}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        train_data = data["train_data"]
        t_step = float(data["t_step"])
        print(f"  [training data cache] loaded {os.path.basename(cache_path)}")
        return train_data, t_step

    hp = _load_hp_csv(system_name)
    sub_t = hp["subsample_t"]
    substeps = hp.get("stepper_substeps", 1)
    gen_sub = sub_t * substeps
    sel = hp["select_dims"]

    sys = _system_factory(hp)
    sys.state = sys.x0.copy()

    def _step_and_collect(n_subsampled):
        out = []
        for _ in range(n_subsampled):
            sys.state = sys.step(sys.state, gen_sub)
            raw = np.asarray(sys.state, float).ravel()
            if sel is not None:
                raw = raw[sel]
            out.append(raw)
        return np.stack(out, axis=0)

    print(f"  [training data cache] generating {os.path.basename(cache_path)}...")
    train_data = _step_and_collect(n_train)

    t_step = sys.t_step * gen_sub
    np.savez(cache_path, train_data=train_data, t_step=t_step)
    print(f"  [training data cache] saved    {os.path.basename(cache_path)}")
    return train_data, t_step


# ---------------------------------------------------------------------------
# ESN builders
# ---------------------------------------------------------------------------


_2RES_SEED_OFFSET = 1
_2RES_COMBINER_FLOOR = 1e-6


def _build_reservoir(hp: dict, input_dim: int, seed: int, name: Optional[str] = None):
    """Construct a single Reservoir node from *hp*."""
    rc_conn = hp["sparse"] / hp["size"]
    kwargs = dict(
        units=hp["size"],
        sr=hp["radius"],
        lr=hp["leak"],
        input_scaling=hp["in_scale"],
        input_connectivity=1.0 / input_dim,
        rc_connectivity=rc_conn,
        W=uniform,
        Win=uniform,
        bias=uniform(input_scaling=hp["bias_scale"]),
        seed=seed,
    )
    if name is not None:
        kwargs["name"] = name
    return Reservoir(**kwargs)


def build_esn(hp: dict, input_dim: int, n_train_samples: int):
    """Construct (but do not train) a reservoirpy ESN matching *hp*."""
    ridge_beta = float(hp.get("ridge", 0.0))
    reservoir = _build_reservoir(hp, input_dim, hp["seed"])
    readout = Ridge(ridge=ridge_beta * n_train_samples, fit_bias=True)
    return ESN(reservoir=reservoir, readout=readout, input_to_readout=True)


def build_two_res_esn(
    hp: dict,
    input_dim: int,
    n_train_samples: int,
    seed_offset: int = _2RES_SEED_OFFSET,
):
    """Construct (but do not train) a two-reservoir reservoir computer."""
    ridge_beta = float(hp.get("ridge", 0.0))
    res1 = _build_reservoir(hp, input_dim, int(hp["seed"]), "2res_res1")
    res2 = _build_reservoir(hp, input_dim, int(hp["seed"]) + seed_offset, "2res_res2")
    ro1 = Ridge(ridge=ridge_beta * n_train_samples, fit_bias=True, name="2res_readout1")
    ro2 = Ridge(ridge=ridge_beta * n_train_samples, fit_bias=True, name="2res_readout2")
    comb = Ridge(ridge=_2RES_COMBINER_FLOOR, fit_bias=True, name="2res_combiner")
    return [res1 >> ro1, res2 >> ro2] >> comb


def load_or_train_esn(
    system_name: str,
    hp: dict,
    train_data: np.ndarray,
    spinup: Optional[int] = None,
    cache: bool = True,
):
    os.makedirs(_MODEL_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_MODEL_CACHE_DIR, f"{system_name}.pkl")
    if cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        print(f"  [trained rc cache] loaded {os.path.basename(cache_path)}")
        return cached["model"], cached["norm"]

    x_mean = train_data[:-1].mean(axis=0)
    x_std = np.maximum(train_data[:-1].std(axis=0), 1e-8)
    norm = {"mean": x_mean, "std": x_std}

    traj_norm = (train_data - x_mean) / x_std
    noise_reg = float(hp.get("noise_reg", 0.0))
    if noise_reg > 0.0:
        rng = np.random.default_rng(int(hp["seed"]))
        traj_norm = traj_norm + noise_reg * rng.standard_normal(traj_norm.shape)

    x_norm = traj_norm[:-1]
    y_norm = traj_norm[1:]

    model = build_esn(hp, train_data.shape[1], len(x_norm))
    if not model.initialized:
        model.initialize(x_norm, y_norm)
    model.reset()
    model.fit(x_norm, y_norm, warmup=(spinup or 0))

    if cache:
        with open(cache_path, "wb") as f:
            pickle.dump({"model": model, "norm": norm}, f)
        print(f"  [trained rc cache] saved  {os.path.basename(cache_path)}")
    return model, norm


def load_or_train_two_res_esn(
    base_system_name: str,
    hp: dict,
    train_data: np.ndarray,
    spinup: Optional[int] = None,
    seed_offset: int = _2RES_SEED_OFFSET,
    cache: bool = True,
):
    os.makedirs(_MODEL_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_MODEL_CACHE_DIR, f"{base_system_name}_2res_off{seed_offset}.pkl")
    if cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        print(f"  [trained rc cache] loaded {os.path.basename(cache_path)}")
        return cached["model"], cached["norm"]

    x_mean = train_data[:-1].mean(axis=0)
    x_std = np.maximum(train_data[:-1].std(axis=0), 1e-8)
    norm = {"mean": x_mean, "std": x_std}

    traj_norm = (train_data - x_mean) / x_std
    noise_reg = float(hp.get("noise_reg", 0.0))
    if noise_reg > 0.0:
        rng = np.random.default_rng(int(hp["seed"]))
        traj_norm = traj_norm + noise_reg * rng.standard_normal(traj_norm.shape)

    x_norm = traj_norm[:-1]
    y_norm = traj_norm[1:]

    model = build_two_res_esn(hp, train_data.shape[1], len(x_norm), seed_offset=seed_offset)
    if not model.initialized:
        model.initialize(x_norm, y_norm)
    model.reset()
    model.fit(x_norm, y_norm, warmup=(spinup or 0))

    if cache:
        with open(cache_path, "wb") as f:
            pickle.dump({"model": model, "norm": norm}, f)
        print(f"  [trained rc cache] saved  {os.path.basename(cache_path)}")
    return model, norm


# ---------------------------------------------------------------------------
# _find_rc_lyap_spec  (single ESN vs. true system)
# ---------------------------------------------------------------------------


def _find_rc_lyap_spec(
    esn_name: str,
    true_result: dict,
    *,
    min_cycles: int = 500,
    max_cycles: int = 2000,
    rtol: float = 0.01,
    progress_bar: bool = True,
    verbose: bool = True,
) -> dict:
    """Train an ESN find its Lyapunov spectrum"""
    _t0 = time.perf_counter()

    is_2res = esn_name.endswith("_2res")
    base_name = esn_name[: -len("_2res")] if is_2res else esn_name

    hp = _load_hp_csv(base_name)
    spinup = int(hp.get("spinup", _SPINUP))
    sys_inst = _system_factory(hp)
    underlying = sys_inst.name

    true_spec = np.asarray(true_result["spectrum"])
    true_ky = true_result["ky_dim"]
    true_ky_ci = true_result.get("ky_dim_ci")
    k = len(true_spec)

    if verbose:
        input_lbl = _input_label(hp)
        size = int(hp.get("size", 0))
        if is_2res:
            model_desc = f"2-reservoir model, size={size} each, input={input_lbl}"
        else:
            model_desc = f"Basic ESN, size={size}, input={input_lbl}"
        print(f"\n  [{esn_name}]  {model_desc}")

    train_data, t_step = generate_train(base_name)

    if is_2res:
        model, norm = load_or_train_two_res_esn(base_name, hp, train_data, spinup=spinup)
    else:
        model, norm = load_or_train_esn(base_name, hp, train_data, spinup=spinup)
    x_norm = (train_data - norm["mean"]) / norm["std"]

    if verbose:
        print(f"  [{esn_name}]  Computing RC Lyapunov spectrum...")
    esn_raw = lyapunov(
        model,
        init_traj=x_norm[:spinup],
        k=k,
        min_cycles=min_cycles,
        max_cycles=max_cycles,
        rtol=rtol,
        progress_bar=progress_bar,
        probe_sub_cycle_cap=50,
    )
    esn_spec = np.asarray(esn_raw["spectrum"]) / t_step
    esn_ky = esn_raw["ky_dim"]
    esn_ky_ci = esn_raw.get("ky_dim_ci")

    nz_thresh = 1e-2 * abs(float(true_spec[0])) if len(true_spec) else 0.0
    keep_true = np.abs(true_spec) >= nz_thresh
    keep_esn = np.abs(esn_spec) >= nz_thresh
    true_spec_red = true_spec[keep_true]
    esn_spec_red = esn_spec[keep_esn]
    true_ky_red = ky_dim(true_spec_red)
    esn_ky_red = ky_dim(esn_spec_red)

    elapsed = time.perf_counter() - _t0
    if verbose:
        print(f"  [{esn_name}]  DONE  ({elapsed:.1f}s)")

    return {
        "esn_name": esn_name,
        "underlying_sys": underlying,
        "t_step": t_step,
        "true_spectrum": true_spec,
        "true_ky": true_ky,
        "true_ky_ci": true_ky_ci,
        "esn_spectrum": esn_spec,
        "esn_ky": esn_ky,
        "esn_ky_ci": esn_ky_ci,
        "true_spectrum_reduced": true_spec_red,
        "esn_spectrum_reduced": esn_spec_red,
        "true_ky_reduced": true_ky_red,
        "esn_ky_reduced": esn_ky_red,
        "is_2res": is_2res,
        "input_label": _input_label(hp),
        "size": int(hp.get("size", 0)),
    }


# ---------------------------------------------------------------------------
# Per-system ASCII results table
# ---------------------------------------------------------------------------


def _lam_str(val: float) -> str:
    """Format a single Lyapunov exponent value for the table."""
    if not np.isfinite(val):
        return "[nan]"
    return f"{val:+.4f}"


def _ky_str(ky, marked: bool = False, ci=None) -> str:
    if ky is None:
        return ""
    base = f"{ky:.3f} + n" if marked else f"{ky:.3f}"
    if ci is not None and np.isfinite(ci):
        return f"{base} ±{ci:.3f}"
    return base


def _rc_row_label(r: dict, multi_rc: bool) -> str:
    rc_type = "2 reservoirs" if r.get("is_2res") else "basic ESN"
    if multi_rc:
        return f"RC, {rc_type}, {r['input_label']} input"
    return f"RC, {rc_type}, size={r['size']}"


def _print_system_table(
    sys_name: str,
    group: list,
    known_row,
    *,
    reduced: bool = False,
) -> None:
    spec_key = "true_spectrum_reduced" if reduced else "true_spectrum"
    espec_key = "esn_spectrum_reduced" if reduced else "esn_spectrum"
    tky_key = "true_ky_reduced" if reduced else "true_ky"
    eky_key = "esn_ky_reduced" if reduced else "esn_ky"

    sys_cls = _SYSTEMS.get(sys_name)
    formal_name = sys_cls.formal_name if sys_cls else sys_name
    method = sys_cls.method if sys_cls else "direct"

    if known_row is not None and known_row.get("spectrum") is not None:
        lit_spec_full = np.asarray(known_row["spectrum"])
        if reduced and sys_name == "kuramoto_sivashinsky":
            keep = [i for i in range(len(lit_spec_full)) if i not in (2, 3)]
            lit_spec_disp = lit_spec_full[keep]
        elif reduced:
            nz_thresh_lit = 1e-2 * abs(float(lit_spec_full[0])) if len(lit_spec_full) else 0.0
            lit_spec_disp = lit_spec_full[np.abs(lit_spec_full) >= nz_thresh_lit]
        else:
            lit_spec_disp = lit_spec_full
    else:
        lit_spec_disp = None

    max_k = max(max(len(r[spec_key]), len(r[espec_key])) for r in group)
    if lit_spec_disp is not None:
        max_k = max(max_k, len(lit_spec_disp))

    multi_rc = len(group) > 1
    source_str = known_row.get("source", "") if known_row else ""
    candidate_labels = [
        f"Literature: {source_str}" if source_str else "",
        method,
    ] + [_rc_row_label(r, multi_rc) for r in group]
    lbl_w = max(28, max(len(lbl) for lbl in candidate_labels) + 2)

    lam_w = 9
    ky_w = 12 if reduced else 18
    suffix = "*" if reduced else ""

    hdr = f"  {'System / kind':<{lbl_w}}"
    for i in range(max_k):
        hdr += f"  {('λ' + str(i + 1) + suffix):>{lam_w}}"
    hdr += f"  {'KY dim':>{ky_w}}"
    total_w = len(hdr) - 2
    sep = "  " + "─" * total_w
    title = f"  ── {formal_name} " + "─" * max(0, total_w - len(formal_name) - 5)

    print(f"\n{title}")
    print(hdr)
    print(sep)

    if lit_spec_disp is not None and len(lit_spec_disp) > 0:
        lit_label = f"Literature: {source_str}" if source_str else "Literature"
        row_s = f"  {lit_label:<{lbl_w}}"
        for i in range(max_k):
            row_s += f"  {_lam_str(lit_spec_disp[i]):>{lam_w}}" if i < len(lit_spec_disp) else f"  {'':>{lam_w}}"
        if reduced:
            ky_lit = ky_dim(lit_spec_disp)
            row_s += f"  {_ky_str(ky_lit, marked=True):>{ky_w}}"
        else:
            ky_lit = known_row.get("ky_dim") if known_row else None
            row_s += f"  {_ky_str(ky_lit):>{ky_w}}"
        print(row_s)

    r0 = group[0]
    t_spec = r0[spec_key]
    row_s = f"  {method:<{lbl_w}}"
    for i in range(max_k):
        row_s += f"  {_lam_str(t_spec[i]):>{lam_w}}" if i < len(t_spec) else f"  {'':>{lam_w}}"
    dropped_true = len(r0["true_spectrum_reduced"]) < len(r0["true_spectrum"])
    true_ci = None if reduced else r0.get("true_ky_ci")
    row_s += f"  {_ky_str(r0[tky_key], reduced and dropped_true, ci=true_ci):>{ky_w}}"
    print(row_s)

    for r in group:
        e_spec = r[espec_key]
        rc_label = _rc_row_label(r, multi_rc)
        row_s = f"  {rc_label:<{lbl_w}}"
        for i in range(max_k):
            row_s += f"  {_lam_str(e_spec[i]):>{lam_w}}" if i < len(e_spec) else f"  {'':>{lam_w}}"
        dropped_esn = len(r["esn_spectrum_reduced"]) < len(r["esn_spectrum"])
        esn_ci = None if reduced else r.get("esn_ky_ci")
        row_s += f"  {_ky_str(r[eky_key], reduced and dropped_esn, ci=esn_ci):>{ky_w}}"
        print(row_s)

    print(sep)


def main(
    esn_names=["lorenz_f", "lorenz_x_2res", "lorenz_x", "lorenz96", "mackey_glass", "kuramoto_sivashinsky"],
    *,
    min_cycles: int = 500,
    max_cycles: int = 2000,
    rtol: float = 0.00,
    progress_bar: bool = True,
    verbose: bool = True,
) -> list:
    """Compare trained-RC Lyapunov spectra to true-system spectra.

    For each unique underlying dynamical system represented in *esn_names*,
    the true spectrum is computed fresh, then each RC that maps to that system
    is trained and assessed.  A combined ASCII comparison table is printed at
    the end.  For Kuramoto-Sivashinsky, an additional reduced table (near-zero
    exponents removed) is always printed.

    Several RC names may share one underlying system: ``lorenz_x``,
    ``lorenz_f``, and ``lorenz_x_2res`` all map to ``"lorenz"``, so they are
    grouped under a single true-spectrum computation.

    Parameters
    ----------
    esn_names : list of str or None
        System names from ``hyperparameters.csv``.  ``None`` runs all rows.
        The ``_2res`` suffix triggers the two-reservoir ensemble path.
    min_cycles, max_cycles, rtol
        Forwarded to ``_lyapunov`` for both true-system and ESN computations.
    progress_bar : bool
        Show tqdm progress bars.
    verbose : bool
        Print status messages and the results table.

    Returns
    -------
    list of dict, one per RC, as returned by :func:`_find_rc_lyap_spec`.
    """
    _t0 = time.perf_counter()

    if esn_names is None:
        hp_path = _HP_CSV
        if not os.path.exists(hp_path):
            raise FileNotFoundError(f"hyperparameters.csv not found at {hp_path!r}.")
        with open(hp_path, newline="") as f:
            esn_names = [row["system"] for row in csv.DictReader(f)]

    underlying_groups: dict = {}
    for esn_name in esn_names:
        base = esn_name[: -len("_2res")] if esn_name.endswith("_2res") else esn_name
        hp = _load_hp_csv(base)
        sys_inst = _system_factory(hp)
        underlying_groups.setdefault(sys_inst.name, []).append(esn_name)

    all_results: list = []
    known = _load_spectra(_KNOWN_SPECS_CSV)

    if verbose:
        print(f"\n{'=' * 64}")
        print(f"{'LYAPUNOV EXPONENT FINDER DEMONSTRATION':^64}")
        print(f"{'=' * 64}")

    for sys_name, esn_list in underlying_groups.items():
        if verbose:
            sys_cls = _SYSTEMS.get(sys_name)
            formal_name = sys_cls.formal_name if sys_cls else sys_name
            rc_list_str = ", ".join(esn_list)
            print(f"\n{'=' * 64}")
            print(f"  {'Underlying system':<21}: {formal_name}")
            print(f"  {'Reservoir Computer(s)':<21}: {rc_list_str}")
            print(f"{'=' * 64}")

        k = _SYSTEMS[sys_name].k_default
        true_result = known_system_test(
            sys_name,
            k=k,
            min_cycles=min_cycles,
            max_cycles=max_cycles,
            rtol=rtol,
            display=progress_bar,
            verbose=verbose,
        )

        for esn_name in esn_list:
            result = _find_rc_lyap_spec(
                esn_name,
                true_result,
                min_cycles=min_cycles,
                max_cycles=max_cycles,
                rtol=rtol,
                progress_bar=progress_bar,
                verbose=verbose,
            )
            all_results.append(result)

        if verbose:
            group = [r for r in all_results if r["underlying_sys"] == sys_name]
            _print_system_table(sys_name, group, known.get(sys_name))

            if sys_name == "kuramoto_sivashinsky":
                print("\nRemoving near-zero exponents:")
                for r in group:
                    _print_system_table(
                        "kuramoto_sivashinsky",
                        [r],
                        known.get("kuramoto_sivashinsky"),
                        reduced=True,
                    )

    elapsed = time.perf_counter() - _t0
    mins, secs = divmod(elapsed, 60)
    print(f"\nOverall Runtime: {int(mins):02d}:{secs:05.2f}", flush=True)

    return all_results


if __name__ == "__main__":
    main()
