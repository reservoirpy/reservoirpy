"""Sequential visual testing of the ESN verification pipeline.

Each ``task_X`` function is independent and relies on cached outputs of the
prior task.  Run from the IDE — results are assessed by visual inspection
(plots + printed stats), not automated thresholds.

Two workflows are available:

1. **run_pass(system_name)** — step-through pipeline for a single system
   (tasks a → e), with interactive prompts and matplotlib figures.  Supported
   systems: ``"lorenz_x"`` and ``"lorenz96"``.

2. **run_benchmark(systems=None, use_new_standards=False)** — quick multi-system
   expected-vs-observed comparison using hyperparameters from /results.  Trains a
   fresh default ESN (no model cache) for each of the five benchmark systems and
   prints tables of valid time and RMSE.  No figures.  Pass
   ``use_new_standards=True`` to train and evaluate on the canonical
   ``/new_standards`` CSV (the exact data the expected metrics came from), rather
   than the locally-generated integrator data.

Typical usage::

    from reservoirpy.tests.lyapunov.esn_test import run_pass, run_benchmark
    run_pass("lorenz_x")
    run_benchmark()
    run_benchmark(["lorenz_x", "lorenz96"])
    run_benchmark(use_new_standards=True)

Or step by step (run_pass workflow)::

    from reservoirpy.tests.lyapunov.esn_test import (
        task_a_data, task_b_hp, task_c_train,
        task_d_valid_time, task_e_reg_sweep,
    )
    task_a_data("lorenz_x")
    task_b_hp("lorenz_x")
    # ... etc.
"""

import csv
import os
import pickle
import sys
from copy import deepcopy

# Ensure the local reservoirpy fork takes precedence.
sys.path.insert(
    0,
    os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")
    ),
)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from reservoirpy.observables import valid_time_multitest, _rc_run_generative
from reservoirpy.tests.lyapunov.lyapunov_test import (
    _SYSTEM_CONFIGS_5,
    _system_factory,
    generate_train_val,
    sample_val_windows,
    _load_best_hp,
    _load_reg_sweep,
    build_esn,
    _arch_components,
    _esn_cache_path,
    _pred_len_for,
    _analytic_spinup,
    _spinup_for,
    _hp_csv_path,
    _load_hp_csv,
    _write_hp_csv,
    _N_TRAIN,
    _N_VAL,
    _N_WINDOWS,
    _KS,
    _PRED_LEN,
    _SPINUP,
    _RIDGE_BETAS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Four levels up from tests/lyapunov/ → Reservoir/
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
_NEW_STANDARDS_DIR = os.path.join(_REPO_ROOT, "reservoir", "datasets", "new_standards")

# Per-pass reference dataset config
PASS_CONFIGS = {
    "lorenz_x":             {"ref_csv": "lorenz_det.csv",      "ref_col": "x"},
    "lorenz_f":             {"ref_csv": "lorenz_det.csv",      "ref_col": "x"},
    "lorenz96":             {"ref_csv": "lorenz96_10_det.csv", "ref_col": "x1"},
    "mackey_glass":         {"ref_csv": "mackeyglass_det.csv", "ref_col": "x"},
    "kuramoto_sivashinsky": {"ref_csv": "ks_22.csv",           "ref_col": "x0"},
}

# Benchmark workflow — five reference systems and display labels
_BENCH_SYSTEMS = [
    "lorenz_x", "lorenz_f", "lorenz96", "mackey_glass", "kuramoto_sivashinsky",
]
_BENCH_LABELS = {
    "lorenz_x":              "Lorenz (x)",
    "lorenz_f":              "Lorenz (full)",
    "lorenz96":              "Lorenz96 (10d)",
    "mackey_glass":          "Mackey-Glass",
    "kuramoto_sivashinsky":  "Kuramoto-Sivashinsky",
}

def _step(msg: str) -> None:
    """Print a consistent progress marker for the current stage."""
    print(f"  -> {msg}", flush=True)


def _show_hint() -> None:
    """Remind the user that the next plot blocks until its window is closed."""
    print("  -> opening figure (close the window to continue) ...", flush=True)




def _load_standard_data(
    system_name: str,
    n_train: int = _N_TRAIN,
    n_val: int = _N_VAL,
) -> tuple:
    """Load train + val data from the canonical ``/new_standards`` CSV.

    Returns the same ``(train_data, val_block, t_step)`` contract as
    ``lyapunov_test.generate_train_val``, but sources data from the
    ``/new_standards`` CSV that the ``/results`` expected metrics were computed
    on, rather than stepping a local integrator.

    Subsampling (``subsample_t``) and column selection (``select_dims``) are
    applied exactly as in ``generate_train_val`` so that the resulting arrays
    are drop-in compatible with ``_setup_esn``.  The CSV is already at the
    standard sampling dt, so ``stepper_substeps`` is **not** applied here.

    Parameters
    ----------
    system_name : str
        Key into ``_SYSTEM_CONFIGS_5`` / ``PASS_CONFIGS``.
    n_train, n_val : int
        Number of (subsampled) timesteps for train / validation blocks.

    Returns
    -------
    train_data : np.ndarray, shape (n_train, n_dims)
    val_block  : np.ndarray, shape (n_val, n_dims)
    t_step     : float  — physical timestep (CSV index spacing × subsample_t)
    """
    cfg     = _SYSTEM_CONFIGS_5[system_name]
    ref_csv = PASS_CONFIGS[system_name]["ref_csv"]
    sub_t   = cfg["subsample_t"]
    sel     = cfg["select_dims"]

    df   = pd.read_csv(os.path.join(_NEW_STANDARDS_DIR, ref_csv), index_col=0)
    dt   = float(df.index[1] - df.index[0])
    vals = df.values[::sub_t]          # subsample rows
    if sel is not None:
        vals = vals[:, sel]            # positional column select (e.g. [0], [-1])

    need = n_train + n_val
    if len(vals) < need:
        raise ValueError(
            f"{ref_csv}: only {len(vals)} subsampled rows available, "
            f"need {need} (n_train={n_train} + n_val={n_val})."
        )

    t_step = dt * sub_t
    print(
        f"  [new_standards] {ref_csv}  "
        f"(dt={dt}, sub_t={sub_t}, t_step={t_step}, dims={vals.shape[1]})"
    )
    return vals[:n_train], vals[n_train:n_train + n_val], t_step


# ---------------------------------------------------------------------------
# ESN setup helpers (edit _setup_esn to experiment with ESN variants)
# ---------------------------------------------------------------------------

def _arch_ridge_solve(model, x_norm: np.ndarray, y_norm: np.ndarray,
                      beta: float, square_method: str = "svd",
                      discard: int = 0) -> None:
    """Re-solve the readout with architectures.solve_square (classic_ridge).

    Mirrors the benchmark's square-solve convention: resets+runs the reservoir
    to collect clean states, assembles the input_to_readout feature matrix
    [R, input, 1], solves via ``architectures.solve_square`` (which regularizes
    the raw summed Gram XᵀX — not the averaged Gram — so β is NOT multiplied by
    T here), and injects the resulting Wout/bias into the already-fit reservoirpy
    readout.  Call AFTER
    ``model.fit(x_norm, y_norm, warmup=discard)`` (i.e. reservoir state is
    already warmed through *discard* steps before this call).

    The *discard* value should come from ``_spinup_for(hp, arch_components)``
    (saved in the HP CSV by ``task_b_hp``); do **not** recompute it here.

    Parameters
    ----------
    model : ESN
        A fitted reservoirpy ESN with ``input_to_readout=True``.
    x_norm : np.ndarray, shape (T, d)
        Normalised training inputs.
    y_norm : np.ndarray, shape (T, d)
        Normalised training targets.
    beta : float
        Ridge regularisation coefficient (the *β* value, NOT β×T).
    square_method : str, default ``"svd"``
        Passed to ``architectures.solve_square`` as ``p["square method"]``.
        Valid values: ``"svd"``, ``"pinv"``, ``"solve"``, ``"cholesky"``.
    discard : int, default 0
        Number of initial transient steps to exclude from the regression
        (from ``_spinup_for``).  Passed by ``_setup_esn``.
    """
    reservoir_dir = os.path.normpath(
        os.path.join(_THIS_DIR, "../../../../reservoir"))
    if reservoir_dir not in sys.path:
        sys.path.insert(0, reservoir_dir)
    import architectures as arch  # noqa: PLC0415 (lazy import by design)

    model.reservoir.reset()               # zero state before collecting
    R = model.reservoir.run(x_norm)       # (T, units)
    print(f"  [arch_ridge_solve] discard={discard}")

    # Skip initial transient — value comes from task_b's analytic spinup estimate.
    R_fit    = R[discard:]
    x_fit    = x_norm[discard:]
    y_fit    = y_norm[discard:]
    T_eff    = len(R_fit)
    X_full = np.hstack([R_fit, x_fit, np.ones((T_eff, 1))])
    ws, _ = arch.solve_square(
        X_full, y_fit,
        regression="ridge", p={"reg ridge": [beta], "square method": square_method},
    )
    W = ws[arch.reg_key(ridge=beta)]      # (d_out, units+d_in+1)
    model.readout.Wout = W[:, :-1].T.copy()   # (units+d_in, d_out)
    model.readout.bias = W[:, -1].copy()      # (d_out,)


def _setup_esn(system_name: str, hp: dict, train_data: np.ndarray,
               overwrite: bool = False, cache: bool = True,
               arch_components: bool = False,
               solve: str = "reservoirpy", square_method: str = "svd"):
    """Build, (optionally) modify, and train an ESN; cache the (model, norm).

    Normalization and noise-injection are identical to
    ``lyapunov_test.load_or_train_esn`` so that training conditions stay
    consistent.  ``norm`` is a dict ``{"mean": x_mean, "std": x_std}``; apply
    to validation data as ``(w - norm["mean"]) / norm["std"]`` before scoring.

    The training transient is discarded using the analytic spinup saved in
    ``hp["spinup"]`` (default reservoirpy matrices) or ``hp["spinup_arch"]``
    (``architectures.basic_res/basic_in`` matrices), selected via
    ``_spinup_for(hp, arch_components)``.  This applies to **both** solve
    paths: ``"reservoirpy"`` uses ``model.fit(..., warmup=spinup)`` and
    ``"arch"`` passes ``discard=spinup`` to ``_arch_ridge_solve``.  If the HP
    CSV predates ``task_b``'s analytic spinup (no ``spinup`` key or key = 200),
    ``_spinup_for`` falls back to ``_SPINUP``.

    The variant kwargs (``arch_components``, ``solve``, ``square_method``) only
    take effect when the model is (re)built, i.e. when ``overwrite=True`` or no
    cache entry exists.

    Parameters
    ----------
    system_name : str
        Key into ``_SYSTEM_CONFIGS_5`` (e.g. ``"lorenz_x"``).
    hp : dict
        Hyperparameter dict (from ``_load_hp_csv``).
    train_data : np.ndarray, shape (N, d)
        Raw (un-normalised) training trajectory.
    overwrite : bool, default False
        When False, a cached model of the same name is loaded if present.
        When True, the model is always rebuilt and the cache overwritten.
    cache : bool, default True
        When False, skip both cache-load and cache-save — always train fresh
        and return without touching any pickle file.  Used by ``run_benchmark``
        to keep the quick default-ESN benchmark independent of the ``task_c``
        model cache (same hp would otherwise alias to the same cache path).
    arch_components : bool, default False
        When True, swap ``W``/``Win``/``bias`` for the exact matrices produced
        by ``architectures.basic_res`` / ``basic_in`` before fitting (mirrors
        ``components.py`` Test 1).
    solve : str, default ``"reservoirpy"``
        Readout solve method.  ``"reservoirpy"`` uses reservoirpy's Ridge;
        ``"arch"`` re-solves via ``architectures.solve_square`` (classic_ridge)
        after fitting (mirrors ``components.py`` Test 2).
    square_method : str, default ``"svd"``
        ``architectures.solve_square`` ``p["square method"]``; used only when
        ``solve="arch"``.  Valid values: ``"svd"``, ``"pinv"``, ``"solve"``,
        ``"cholesky"``.
    """
    if solve not in ("reservoirpy", "arch"):
        raise ValueError(
            f"solve must be 'reservoirpy' or 'arch', got {solve!r}."
        )

    # Select analytic spinup from the HP CSV (computed by task_b).
    spinup = _spinup_for(hp, arch_components)

    cache_path = _esn_cache_path(system_name, hp)
    if cache and os.path.exists(cache_path) and not overwrite:
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        print(f"  [cache] loaded {os.path.basename(cache_path)}")
        return cached["model"], cached["norm"]

    variant_msg = (f"  [setup] arch_components={arch_components}  solve={solve}"
                   f"  spinup={spinup}"
                   + (f"  square_method={square_method!r}" if solve == "arch" else ""))
    print(variant_msg, flush=True)

    # Normalization — clip std away from zero.
    x_mean = train_data[:-1].mean(axis=0)
    x_std  = np.maximum(train_data[:-1].std(axis=0), 1e-8)
    norm = {"mean": x_mean, "std": x_std}

    # Normalize the full trajectory, then optionally add input noise.
    # Noise is added AFTER normalization (std=noise_reg in normalized units),
    # matching the framework's 'noise' regression.  Validation data is never
    # noised — norm stats come from the clean trajectory.
    traj_norm = (train_data - x_mean) / x_std
    noise_reg = float(hp.get("noise_reg", 0.0))
    if noise_reg > 0.0:
        rng = np.random.default_rng(int(hp["seed"]))
        traj_norm = traj_norm + noise_reg * rng.standard_normal(traj_norm.shape)
    x_norm = traj_norm[:-1]
    y_norm = traj_norm[1:]

    input_dim = train_data.shape[1]
    model = build_esn(hp, input_dim, len(x_norm))

    # PRE-FIT: swap reservoir matrices for architectures.basic_res / basic_in.
    # reservoirpy only converts *callable* W/Win/bias during initialize(), so
    # assigning arrays here makes the reservoir use them verbatim at fit time.
    if arch_components:
        W_arr, Win_arr, bias_arr = _arch_components(hp, input_dim)
        model.reservoir.W    = W_arr
        model.reservoir.Win  = Win_arr
        model.reservoir.bias = bias_arr

    # Explicit reset: guarantee the reservoir starts training from r=0 instead of
    # relying on initialize()'s implicit zeroing.  Model.fit() does NOT reset node
    # state (reservoirpy/model.py:519), so this makes the training pass robust to
    # any reuse / re-fit of the model object.  initialize() must run first to create
    # the state dict; fit() will skip re-initialization via its own guard.
    if not model.initialized:
        model.initialize(x_norm, y_norm)
    model.reset()

    if solve == "arch":
        # Initialize reservoir (fit discards readout result; re-solved below).
        model.fit(x_norm, y_norm)
        # POST-FIT: re-solve the readout via architectures.solve_square,
        # discarding the first `spinup` transient steps.
        _arch_ridge_solve(model, x_norm, y_norm,
                          float(hp.get("ridge", 0.0)), square_method=square_method,
                          discard=spinup)
    else:
        # solve="reservoirpy": warmup discards the first `spinup` steps from
        # the readout regression while keeping the reservoir warmed through them.
        model.fit(x_norm, y_norm, warmup=spinup)

    if cache:
        with open(cache_path, "wb") as f:
            pickle.dump({"model": model, "norm": norm}, f)
        print(f"  [cache] saved  {os.path.basename(cache_path)}")
    return model, norm


# ---------------------------------------------------------------------------
# task_a — Dataset generation, caching, and recall
# ---------------------------------------------------------------------------

def task_a_data(system_name: str, use_new_standards: bool = False) -> None:
    """Generate (or load cached) data; compare statistics with new_standards reference.

    Prints a table comparing mean and std of the generated data and its 1-step
    discrete diff against the corresponding new_standards reference CSV (subsampled
    by the system's ``subsample_t`` factor).  Shows a 2-subplot figure with a
    500-step section of each dataset.

    Parameters
    ----------
    use_new_standards : bool, default False
        When True, load data from the ``/new_standards`` CSV instead of the
        local integrator.  The reference subplot is omitted (data is its own
        reference).
    """
    cfg = PASS_CONFIGS[system_name]
    ref_col = cfg["ref_col"]

    print(f"\n{'='*60}")
    print(f"  task_a_data: {system_name}")
    print(f"{'='*60}")

    # --- Primary data ---------------------------------------------------
    if use_new_standards:
        _step(f"loading {_N_TRAIN}-step train + {_N_VAL}-step val data from /new_standards")
        train_data, val_block, t_step = _load_standard_data(
            system_name, n_train=_N_TRAIN, n_val=_N_VAL,
        )
        data_label = f"new_standards ({cfg['ref_csv']}, t_step={t_step:.4f})"
    else:
        _step(f"generating/loading {_N_TRAIN}-step train + {_N_VAL}-step val data (cached)")
        train_data, val_block, t_step = generate_train_val(
            system_name, n_train=_N_TRAIN, n_val=_N_VAL,
        )
        data_label = f"generated (t_step={t_step:.4f})"

    # Use first variable (index 0) for 1-D comparison
    _step("computing mean/std of data and its 1-step diff (first variable)")
    gen_col = train_data[:, 0]
    gen_diff = np.diff(gen_col)
    print(f"\n  {data_label}:")
    print(f"    data  — mean={gen_col.mean():.4f}  std={gen_col.std():.4f}")
    print(f"    1-step diff — mean={gen_diff.mean():.4f}  std={gen_diff.std():.4f}")

    # --- New-standards reference (only when using generated data) --------
    ref_col_data = None
    sub_t = _SYSTEM_CONFIGS_5[system_name]["subsample_t"]
    if not use_new_standards:
        ref_path = os.path.join(_NEW_STANDARDS_DIR, cfg["ref_csv"])
        if not os.path.exists(ref_path):
            print(f"\n  [WARN] reference CSV not found: {ref_path}")
        else:
            _step(f"loading new_standards reference '{cfg['ref_csv']}' and subsampling ×{sub_t}")
            ref_df = pd.read_csv(ref_path, index_col=0)
            ref_df = ref_df.iloc[::sub_t].reset_index(drop=True)
            ref_col_data = ref_df[ref_col].values
            _step("computing reference mean/std and 1-step diff for comparison")
            ref_diff = np.diff(ref_col_data)
            print(f"\n  New-standards ref (subsampled ×{sub_t}, col='{ref_col}'):")
            print(f"    data  — mean={ref_col_data.mean():.4f}  std={ref_col_data.std():.4f}")
            print(f"    1-step diff — mean={ref_diff.mean():.4f}  std={ref_diff.std():.4f}")
            print(f"\n  Ratio generated/reference (std): "
                  f"{gen_col.std() / ref_col_data.std():.4f}")

    # --- Figure -----------------------------------------------------------
    _show_hint()
    n_show = 500
    n_rows = 1 if use_new_standards else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows + 1), sharex=True)
    if n_rows == 1:
        axes = [axes]
    axes[0].plot(gen_col[:n_show], 'b', linewidth=0.7)
    axes[0].set_title(f"{system_name} — {data_label}")
    axes[0].set_ylabel("x[0]")
    if ref_col_data is not None:
        axes[1].plot(ref_col_data[:n_show], 'b', linewidth=0.7)
        axes[1].set_title(
            f"new_standards ref — {cfg['ref_csv']} col '{ref_col}' "
            f"(subsampled ×{sub_t})"
        )
        axes[1].set_ylabel(ref_col)
    axes[-1].set_xlabel("steps")
    fig.suptitle(f"task_a: {system_name} — {n_show}-step sections")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# task_b — Hyperparameter finding, saving, and recall
# ---------------------------------------------------------------------------

def task_b_hp(system_name: str, use_new_standards: bool = False) -> None:
    """Load best HPs from full_losses CSV, compute analytic spinup, write to CSV, display.

    Computes the analytic synchronisation spinup for **both** reservoir builders
    (default reservoirpy uniform matrices and ``architectures.basic_res/basic_in``)
    and saves both to the CSV as ``spinup`` and ``spinup_arch`` respectively.
    These values are used as the training discard and inference warmup in all
    downstream tasks, regardless of ``arch_components`` or ``solve``.

    The saved CSV includes: time_step, size, seed, leak, radius, in_scale,
    bias_scale, spinup, spinup_arch, sparse, subsample_t, ridge, noise_reg,
    vt_0_2_std_med, l2_gmean_4/16/64/256.

    Parameters
    ----------
    use_new_standards : bool, default False
        When True, load training data from the ``/new_standards`` CSV (same
        source as the reference expected metrics) rather than the local
        integrator.  Passed on by ``run_pass`` to match the data source in use.
    """
    print(f"\n{'='*60}")
    print(f"  task_b_hp: {system_name}")
    print(f"{'='*60}")

    _step("reading best hyperparameters from full_losses CSV (max valid time 0.2 std med)")
    hp = _load_best_hp(system_name)

    # Derive physical time step (includes stepper_substeps for Mackey-Glass)
    _step("deriving physical time step (system t_step × subsample_t × stepper_substeps)")
    sys_obj  = _system_factory(_SYSTEM_CONFIGS_5[system_name])
    substeps = _SYSTEM_CONFIGS_5[system_name].get("stepper_substeps", 1)
    physical_t_step = sys_obj.t_step * hp["subsample_t"] * substeps

    # Load training data for the analytic spinup computation.
    _step("loading training data for analytic spinup estimation")
    if use_new_standards:
        train_data, _, _ = _load_standard_data(system_name, n_train=_N_TRAIN, n_val=1)
    else:
        train_data, _, _ = generate_train_val(system_name, n_train=_N_TRAIN, n_val=1)

    # Compute analytic spinup for both reservoir builders.
    _step("estimating analytic spinup (uniform / default reservoirpy matrices)")
    est_uniform = _analytic_spinup(hp, train_data, arch_components=False)
    _step("estimating analytic spinup (architectures basic_res/basic_in matrices)")
    est_arch    = _analytic_spinup(hp, train_data, arch_components=True)

    print(f"\n  Analytic spinup estimates for {system_name}:")
    print(f"  {'builder':24s}  {'contraction':>12}  {'sync_time':>10}  {'rec_spin':>9}  {'spinup':>7}")
    for label, est in [("uniform (reservoirpy)", est_uniform), ("arch (basic_res)", est_arch)]:
        print(f"  {label:24s}  {est['contraction']:12.6f}  {est['sync_time']:10.2f}"
              f"  {est['rec_spin']:9d}  {est['spinup']:7d}")

    # Build the row to save (all spec-listed HPs + expected metrics)
    save_row = {
        "time_step":      physical_t_step,
        "size":           hp["size"],
        "seed":           hp["seed"],
        "leak":           hp["leak"],
        "radius":         hp["radius"],
        "in_scale":       hp["in_scale"],
        "bias_scale":     hp["bias_scale"],
        "spinup":         est_uniform["spinup"],
        "spinup_arch":    est_arch["spinup"],
        "sparse":         hp["sparse"],
        "subsample_t":    hp["subsample_t"],
        "ridge":          hp["ridge"],
        "noise_reg":      hp["noise_reg"],
        "vt_0_2_std_med": hp["vt_0_2_std_med"],
        "l2_gmean_4":     hp["l2_gmean_4"],
        "l2_gmean_16":    hp["l2_gmean_16"],
        "l2_gmean_64":    hp["l2_gmean_64"],
        "l2_gmean_256":   hp["l2_gmean_256"],
    }

    _step("writing hyperparameters + expected metrics to CSV")
    _write_hp_csv(system_name, save_row)
    print(f"  Written: {_hp_csv_path()} (row: {system_name})")

    # Reload and display
    _step("reloading the HP CSV and displaying")
    reloaded = _load_hp_csv(system_name)
    print(f"\n  Reloaded HPs for {system_name}:")
    for k, v in reloaded.items():
        print(f"    {k:20s} = {v}")

    print(f"\n  Expected performance:")
    print(f"    vt_0_2_std_med = {reloaded['vt_0_2_std_med']:.4f}")
    for k in _KS:
        print(f"    l2_gmean_{k:3d}   = {reloaded[f'l2_gmean_{k}']:.4f}")


# ---------------------------------------------------------------------------
# task_c — ESN construction, training, caching, and prediction overlay
# ---------------------------------------------------------------------------

def task_c_train(system_name: str, arch_components: bool = False,
                 solve: str = "reservoirpy", square_method: str = "svd",
                 use_new_standards: bool = False) -> None:
    """Train ESN (always overwriting the cache); show predicted vs validation for 5 windows.

    Validation in blue, predicted in orange.  x-axis is aligned so prediction
    occupies indices spinup .. spinup+pred_len.  Plots in normalised space.

    Parameters
    ----------
    arch_components : bool, default False
        Passed to ``_setup_esn``.  When True, uses exact ``architectures.basic_res``
        / ``basic_in`` matrices for the reservoir.
    solve : str, default ``"reservoirpy"``
        Passed to ``_setup_esn``.  ``"reservoirpy"`` uses reservoirpy's Ridge;
        ``"arch"`` re-solves with ``architectures.solve_square``.
    square_method : str, default ``"svd"``
        Passed to ``_setup_esn``.  ``architectures.solve_square`` square method;
        used only when ``solve="arch"``.
    use_new_standards : bool, default False
        When True, load train/val data from the ``/new_standards`` CSV instead
        of the local integrator.
    """
    print(f"\n{'='*60}")
    print(f"  task_c_train: {system_name}")
    print(f"{'='*60}")

    _step("loading hyperparameters from CSV (task_b output)")
    hp_full = _load_hp_csv(system_name)
    if use_new_standards:
        _step("loading train + val data from /new_standards CSV")
        train_data, val_block, t_step = _load_standard_data(
            system_name, n_train=_N_TRAIN, n_val=_N_VAL,
        )
    else:
        _step("loading/generating train + val data (cached)")
        train_data, val_block, t_step = generate_train_val(
            system_name, n_train=_N_TRAIN, n_val=_N_VAL,
        )
    # _setup_esn expects keys: size, seed, leak, radius, in_scale,
    #   bias_scale, sparse, ridge  (subsample_t not used by build_esn itself)
    _step("training ESN (overwriting any cached model)")
    model, norm = _setup_esn(system_name, hp_full, train_data, overwrite=True,
                             arch_components=arch_components,
                             solve=solve, square_method=square_method)

    spinup = _spinup_for(hp_full, arch_components)
    _step(f"sampling 5 random validation windows (spinup={spinup})")
    windows = sample_val_windows(
        val_block, n=5, spinup=spinup, pred_len=_PRED_LEN, seed=0,
    )

    fig, axes = plt.subplots(5, 1, figsize=(13, 16))
    for i, window in enumerate(windows):
        _step(f"window {i+1}/5: warming up {spinup} steps, "
              f"rolling out {_PRED_LEN} steps closed-loop")
        window_norm = (window - norm["mean"]) / norm["std"]
        # Warm up a fresh copy then roll out closed-loop
        m_copy = deepcopy(model)
        m_copy.run(window_norm[:spinup])
        pred = _rc_run_generative(m_copy, _PRED_LEN, prepend_current=True)  # (pred_len, d)

        ax = axes[i]
        steps_full = np.arange(len(window_norm))
        steps_pred = np.arange(spinup, spinup + _PRED_LEN)
        ax.plot(steps_full, window_norm[:, 0], color="blue",   linewidth=0.8,
                label="validation" if i == 0 else None)
        ax.plot(steps_pred,  pred[:, 0],        color="orange", linewidth=0.8,
                label="predicted"  if i == 0 else None)
        ax.axvline(spinup, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel(f"window {i+1}")
        if i == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("steps")
    fig.suptitle(f"{system_name} — 5 prediction overlays  (spinup={spinup}  pred={_PRED_LEN})")
    plt.tight_layout()
    _show_hint()
    plt.show()


# ---------------------------------------------------------------------------
# task_d — Valid-time evaluation and RMSE comparison
# ---------------------------------------------------------------------------

def task_d_valid_time(system_name: str, use_new_standards: bool = False) -> None:
    """Evaluate valid prediction time and RMSE over 50 windows; compare with expected.

    Prints observed vs expected for both valid time and RMSE at each horizon.
    Shows a loglog plot of RMSE vs step count for observed (solid) and expected
    (dashed) results.

    Parameters
    ----------
    use_new_standards : bool, default False
        When True, load train/val data from the ``/new_standards`` CSV instead
        of the local integrator.
    """
    print(f"\n{'='*60}")
    print(f"  task_d_valid_time: {system_name}")
    print(f"{'='*60}")

    _step("loading hyperparameters from CSV (task_b output)")
    hp_full = _load_hp_csv(system_name)
    if use_new_standards:
        _step("loading train + val data from /new_standards CSV")
        train_data, val_block, t_step = _load_standard_data(
            system_name, n_train=_N_TRAIN, n_val=_N_VAL,
        )
    else:
        _step("loading/generating train + val data (cached)")
        train_data, val_block, t_step = generate_train_val(
            system_name, n_train=_N_TRAIN, n_val=_N_VAL,
        )
    _step("loading cached (or training) ESN")
    model, norm = _setup_esn(system_name, hp_full, train_data)

    spinup   = _spinup_for(hp_full, arch_components=False)
    pred_len = _pred_len_for(hp_full, t_step)
    _step(f"sampling {_N_WINDOWS} validation windows (spinup={spinup}, pred_len={pred_len}) and normalising")
    windows = sample_val_windows(
        val_block, n=_N_WINDOWS, spinup=spinup, pred_len=pred_len, seed=1,
    )
    windows_norm = [(w - norm["mean"]) / norm["std"] for w in windows]

    _step(f"running valid_time_multitest (eps=[0.2,0.4], ks={_KS}) on {_N_WINDOWS} windows")
    results = valid_time_multitest(
        model, windows_norm, n_segments=_N_WINDOWS,
        spinup=spinup, eps=[0.2, 0.4], ks=_KS,
        block=pred_len, t_step=t_step,
    )

    # Print comparison table
    _step("comparing observed vs expected valid time and RMSE")
    obs_vt  = results.get("valid_time_0.2_median", float("nan"))
    exp_vt  = hp_full["vt_0_2_std_med"]
    ratio   = obs_vt / exp_vt if exp_vt > 0 else float("nan")
    print(f"\n  Valid time (0.2×std threshold, median):")
    print(f"    observed  = {obs_vt:.4f}")
    print(f"    expected  = {exp_vt:.4f}")
    print(f"    ratio     = {ratio:.3f}")
    print(f"\n  valid_time_0.4_median  = "
          f"{results.get('valid_time_0.4_median', float('nan')):.4f}")
    print(f"  valid_time_fitness     = "
          f"{results.get('valid_time_fitness', float('nan')):.4f}")

    obs_rmse = [results.get(f"rmse_{k}_gmean", float("nan")) for k in _KS]
    exp_rmse = [hp_full[f"l2_gmean_{k}"] for k in _KS]
    print(f"\n  RMSE gmean by horizon:")
    print(f"  {'k':>6}  {'observed':>12}  {'expected':>12}  {'ratio':>8}")
    for k, obs, exp in zip(_KS, obs_rmse, exp_rmse):
        r = obs / exp if exp > 0 and np.isfinite(obs) else float("nan")
        print(f"  {k:6d}  {obs:12.5f}  {exp:12.5f}  {r:8.3f}")

    # Loglog RMSE plot
    _show_hint()
    fig, ax = plt.subplots(figsize=(7, 5))
    ks_arr = np.array(_KS, dtype=float)
    ax.loglog(ks_arr, obs_rmse, "o-",  color="blue",   label="observed")
    ax.loglog(ks_arr, exp_rmse, "s--", color="orange", label="expected")
    ax.set_xlabel("steps k")
    ax.set_ylabel("normalised RMSE (geometric mean)")
    ax.set_title(f"{system_name} — RMSE vs forecast horizon")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# task_e — Regularization sweep
# ---------------------------------------------------------------------------

def task_e_reg_sweep(system_name: str, use_new_standards: bool = False,
                     arch_solve: bool = True) -> None:
    """Train ESNs over the full reg sweep; compare observed and expected valid time.

    Uses the same best HP config but sweeps the regularisation parameter across
    all values in the source full_losses CSV.  For ridge systems this is the
    ridge β; for noise systems this is the noise std.  Shows a semilogx plot of
    reg value vs valid time for both observed and expected.  The reg=0 point is
    plotted at ``min_nonzero × 1e-3``.

    Parameters
    ----------
    use_new_standards : bool, default False
        When True, load train/val data from the ``/new_standards`` CSV instead
        of the local integrator.
    arch_solve : bool, default True
        When True, re-solve the readout via ``architectures.solve_square`` with
        ``square_method="svd"`` (matching the benchmark's square/svd convention).
        When False, use reservoirpy's built-in Ridge solver.
    """
    print(f"\n{'='*60}")
    print(f"  task_e_reg_sweep: {system_name}  arch_solve={arch_solve}")
    print(f"{'='*60}")

    method = _SYSTEM_CONFIGS_5[system_name]["reg_method"]  # "ridge" or "noise"
    reg_label = "noise std" if method == "noise" else "ridge β"
    solve = "arch" if arch_solve else "reservoirpy"

    _step("loading hyperparameters from CSV (task_b output)")
    hp_full = _load_hp_csv(system_name)
    _step("loading the regularization sweep (all reg values) from full_losses CSV")
    sweep = _load_reg_sweep(system_name)   # sorted by reg_value
    if use_new_standards:
        _step("loading train + val data from /new_standards CSV")
        train_data, val_block, t_step = _load_standard_data(
            system_name, n_train=_N_TRAIN, n_val=_N_VAL,
        )
    else:
        _step("loading/generating train + val data (cached)")
        train_data, val_block, t_step = generate_train_val(
            system_name, n_train=_N_TRAIN, n_val=_N_VAL,
        )

    spinup   = _spinup_for(hp_full, arch_components=False)
    pred_len = _pred_len_for(hp_full, t_step)
    _step(f"sampling {_N_WINDOWS} validation windows (spinup={spinup}, pred_len={pred_len})")
    windows = sample_val_windows(
        val_block, n=_N_WINDOWS, spinup=spinup, pred_len=pred_len, seed=2,
    )

    regs_exp, vt_exp = [], []
    regs_obs, vt_obs = [], []

    print(f"  sweeping {len(sweep)} {reg_label} values (train/load ESN + evaluate each):")
    for i, entry in enumerate(sweep):
        reg_val = entry["reg_value"]
        print(f"  [{i+1}/{len(sweep)}] {reg_label}={reg_val:.2e} ...", flush=True)

        # Build hp dict with the swept reg value; zero out the other regulariser.
        # Preserve the analytic spinup columns from hp_full.
        hp_b = dict(hp_full)
        if method == "noise":
            hp_b["noise_reg"] = reg_val
            hp_b["ridge"]     = 0.0
        else:
            hp_b["ridge"]     = reg_val
            hp_b["noise_reg"] = 0.0

        model_b, norm_b = _setup_esn(system_name, hp_b, train_data,
                                     arch_components=True, solve=solve,
                                     square_method="svd", cache=False)
        windows_norm = [(w - norm_b["mean"]) / norm_b["std"] for w in windows]

        res = valid_time_multitest(
            model_b, windows_norm, n_segments=_N_WINDOWS,
            spinup=spinup, eps=[0.2, 0.4],
            block=pred_len, t_step=t_step,
        )
        obs_vt = res.get("valid_time_0.2_median", float("nan"))

        regs_exp.append(reg_val)
        vt_exp.append(entry["vt_0_2_std_med"])
        regs_obs.append(reg_val)
        vt_obs.append(obs_vt)
        print(f"    obs={obs_vt:.3f}  exp={entry['vt_0_2_std_med']:.3f}")

    # Replace reg=0 with min_nonzero × 1e-3 for log-axis placement
    nonzero = [r for r in regs_exp if r > 0]
    reg_zero_plot = (min(nonzero) * 1e-3) if nonzero else 1e-21

    def _plot_regs(rs):
        return [reg_zero_plot if r == 0.0 else r for r in rs]

    regs_exp_plot = _plot_regs(regs_exp)
    regs_obs_plot = _plot_regs(regs_obs)

    _show_hint()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(regs_obs_plot, vt_obs, "o-",  color="blue",   label="observed")
    ax.semilogx(regs_exp_plot, vt_exp, "s--", color="orange", label="expected")
    # Annotate the reg=0 tick
    if 0.0 in regs_exp:
        ax.axvline(reg_zero_plot, color="gray", linestyle=":", alpha=0.6,
                   label=f"reg=0 → {reg_zero_plot:.1e}")
    ax.set_xlabel(f"{reg_label} (log scale)")
    ax.set_ylabel("valid time 0.2×std median")
    ax.set_title(f"{system_name} — valid time vs {reg_label} sweep")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# task_f — closed-loop fidelity diagnostic
# ---------------------------------------------------------------------------

def task_f_rollout(system_name: str, arch_components: bool = False,
                   solve: str = "reservoirpy", square_method: str = "svd",
                   use_new_standards: bool = False,
                   pred_len: int = 512) -> None:
    """Bisect the closed-loop fidelity gap for *system_name*.

    Two complementary tests, each on a single validation window:

    **Step 1 — Open-loop one-step NRMSE (teacher-forced)**
        Run the model with true inputs at every step and measure one-step
        prediction error.  If NRMSE at k = 4,16,64 matches the expected
        ``l2_gmean_k`` values from /results, the W/Win/Wout matrices are
        correct and any gap must be in the closed-loop mechanics.

    **Step 2 — Manual numpy vs reservoirpy closed-loop**
        Extract reservoir and readout weights, warm up via a manual loop, then
        roll out both a direct numpy closed-loop and ``_rc_run_generative``.
        Identical per-step errors confirm reservoirpy's feedback routing is
        correct; a discrepancy localises the bug to the feedback plumbing.

    After both tests a figure shows the per-step error curves (open-loop,
    numpy CL, reservoirpy CL) alongside the expected NRMSE reference lines.

    Parameters
    ----------
    system_name : str
        Key into ``_SYSTEM_CONFIGS_5`` / ``PASS_CONFIGS``.
    arch_components : bool, default False
        Use exact ``architectures.basic_res`` / ``basic_in`` matrices.
    solve : str, default ``"reservoirpy"``
        ``"reservoirpy"`` (Ridge) or ``"arch"`` (architectures.solve_square).
    square_method : str, default ``"svd"``
        Passed to ``_arch_ridge_solve``; used only when ``solve="arch"``.
    use_new_standards : bool, default False
        Load data from ``/new_standards`` CSV instead of local integrator.
    pred_len : int, default 512
        Closed-loop prediction steps.  Open-loop warmup uses the analytic
        spinup from the HP CSV via ``_spinup_for(hp, arch_components)``.
    """
    print(f"\n{'='*60}")
    print(f"  task_f_rollout: {system_name}")
    print(f"  arch_components={arch_components}  solve={solve!r}  "
          f"use_new_standards={use_new_standards}")
    print(f"{'='*60}")

    _step("loading hyperparameters from CSV (task_b output)")
    hp_full = _load_hp_csv(system_name)
    spinup = _spinup_for(hp_full, arch_components)
    print(f"  -> spinup (analytic, arch_components={arch_components}): {spinup}")

    if use_new_standards:
        _step("loading train + val data from /new_standards CSV")
        train_data, val_block, t_step = _load_standard_data(
            system_name, n_train=_N_TRAIN, n_val=_N_VAL,
        )
    else:
        _step("loading/generating train + val data")
        train_data, val_block, t_step = generate_train_val(
            system_name, n_train=_N_TRAIN, n_val=_N_VAL,
        )

    # -----------------------------------------------------------------------
    # Helper: closed-loop rollout and NRMSE for one model
    # -----------------------------------------------------------------------
    def _rollout_one(mdl, wn, sp, pl):
        """Warm up mdl on wn[:sp], roll out pl steps, return (preds, err)."""
        m = deepcopy(mdl)
        m.run(wn[:sp])
        preds = _rc_run_generative(m, pl, prepend_current=True)
        err = np.linalg.norm(preds - wn[sp:sp + pl], axis=-1)
        return preds, err

    # Normalisation std from full val block (matches framework's val-set norm)
    val_norm_full = (val_block - train_data.mean(axis=0)) / np.maximum(
        train_data.std(axis=0), 1e-8
    )

    # -----------------------------------------------------------------------
    # Train a single model using the saved analytic spinup (via _setup_esn).
    # -----------------------------------------------------------------------
    _step("training ESN (analytic spinup, no-cache)")
    model_d0, norm = _setup_esn(system_name, hp_full, train_data,
                                arch_components=arch_components,
                                solve=solve, square_method=square_method,
                                cache=False)
    model_d100 = None   # retained for table column header compatibility below

    # -----------------------------------------------------------------------
    # Sample one deterministic validation window
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(42)
    win_len = spinup + pred_len
    max_start = len(val_block) - win_len
    start = int(rng.integers(0, max_start + 1))
    window = val_block[start: start + win_len]
    window_norm = (window - norm["mean"]) / norm["std"]
    truth = window_norm[spinup:]              # shape (pred_len, d)

    # Use full val block for norm_std (matches framework)
    from reservoirpy.observables import _data_std as _dstd
    norm_std = _dstd(val_norm_full)

    # -----------------------------------------------------------------------
    # Step 1 — Open-loop one-step (teacher-forced) NRMSE
    # -----------------------------------------------------------------------
    _step("step 1: open-loop teacher-forcing")
    m_ol = deepcopy(model_d0)
    ol_preds = np.asarray(m_ol.run(window_norm[:-1]), dtype=float)
    ol_err = np.linalg.norm(ol_preds - window_norm[1:], axis=-1)

    print(f"\n  Open-loop one-step NRMSE (steps aligned to CL start):")
    print(f"  {'k':>6}  {'OL-NRMSE':>12}  {'expected':>12}  {'ratio':>8}")
    for k in _KS:
        ol_k = float(np.sqrt(np.mean(ol_err[spinup - 1: spinup - 1 + k] ** 2))) / norm_std
        exp_k = hp_full[f"l2_gmean_{k}"]
        r = ol_k / exp_k if exp_k > 0 else float("nan")
        print(f"  {k:6d}  {ol_k:12.5f}  {exp_k:12.5f}  {r:8.3f}")

    # -----------------------------------------------------------------------
    # Step 2 — Closed-loop comparison: discard=0 vs discard=100 vs numpy
    # -----------------------------------------------------------------------
    _step("step 2: closed-loop rollout (discard=0)")
    preds_d0, err_d0 = _rollout_one(model_d0, window_norm, spinup, pred_len)

    if model_d100 is not None:
        _step("step 2: closed-loop rollout (discard=100)")
        preds_d100, err_d100 = _rollout_one(model_d100, window_norm, spinup, pred_len)

    _step("step 2b: manual numpy closed-loop (discard=0)")
    res    = model_d0.reservoir
    W      = np.asarray(res.W, dtype=float)
    Win    = np.asarray(res.Win, dtype=float)
    b_res  = np.asarray(res.bias, dtype=float)
    lr     = float(res.lr)
    Wout   = np.asarray(model_d0.readout.Wout, dtype=float)
    b_out  = np.asarray(model_d0.readout.bias, dtype=float)

    r = np.zeros(W.shape[0])
    for t in range(spinup):
        u = window_norm[t]
        r = (1.0 - lr) * r + lr * np.tanh(W @ r + Win @ u + b_res)
    u_in = window_norm[spinup - 1]
    feat_seed = np.concatenate([r, u_in])
    p0 = feat_seed @ Wout + b_out
    preds_np = [p0]; u = p0
    for _ in range(pred_len - 1):
        r = (1.0 - lr) * r + lr * np.tanh(W @ r + Win @ u + b_res)
        u = np.concatenate([r, u]) @ Wout + b_out
        preds_np.append(u.copy())
    preds_np = np.array(preds_np)
    err_np = np.linalg.norm(preds_np - truth, axis=-1)

    # -----------------------------------------------------------------------
    # Step 2c — Framework-style CL: spinup-1 true warmup, seed from predicted
    #           u_{spinup-1} (mirrors Architecture.predict with reset=True).
    #
    # Architecture.predict processes us_init[0:spinup-2] (spinup-1 steps) then
    # seeds CL with the model's prediction from us_init[spinup-2], not the OL
    # output from the true us_init[spinup-1].  The seed is therefore 1 CL step
    # ahead of the reservoirpy prepend_current seed.
    # -----------------------------------------------------------------------
    _step("step 2c: framework-style CL (spinup-1 warmup, predicted-u seed)")
    r_fw = np.zeros(W.shape[0])
    for t in range(spinup - 1):                  # window_norm[0] … window_norm[spinup-2]
        r_fw = (1.0 - lr) * r_fw + lr * np.tanh(W @ r_fw + Win @ window_norm[t] + b_res)
    # r_fw = reservoir state after seeing window_norm[spinup-2]
    # y_fw = model's OL prediction of window_norm[spinup-1] (the "missing" last warmup step)
    y_fw = np.concatenate([r_fw, window_norm[spinup - 2]]) @ Wout + b_out
    preds_fw = []; u_fw = y_fw
    for _ in range(pred_len):
        r_fw = (1.0 - lr) * r_fw + lr * np.tanh(W @ r_fw + Win @ u_fw + b_res)
        u_fw = np.concatenate([r_fw, u_fw]) @ Wout + b_out
        preds_fw.append(u_fw.copy())
    preds_fw = np.array(preds_fw)
    err_fw = np.linalg.norm(preds_fw - truth, axis=-1)

    seed_diff = float(np.linalg.norm(preds_np[0] - preds_fw[0]))
    print(f"  Seed difference |rpy[0] - fw[0]|: {seed_diff:.3e}  "
          f"(0 ⟹ seeding is identical; nonzero ⟹ predicted vs true u_{{spinup-1}})")

    # -----------------------------------------------------------------------
    # NRMSE table
    # -----------------------------------------------------------------------
    print(f"\n  Closed-loop NRMSE summary (spinup={spinup}, norm_std from full val block):")
    print(f"  {'k':>6}  {'rpy':>10}  {'numpy':>10}  {'fw-style':>10}  {'expected':>10}")

    for k in _KS:
        nrmse_rpy = float(np.sqrt(np.mean(err_d0[:k] ** 2))) / norm_std
        nrmse_np  = float(np.sqrt(np.mean(err_np[:k] ** 2))) / norm_std
        nrmse_fw  = float(np.sqrt(np.mean(err_fw[:k] ** 2))) / norm_std
        exp_k     = hp_full[f"l2_gmean_{k}"]
        print(f"  {k:6d}  {nrmse_rpy:10.5f}  {nrmse_np:10.5f}  {nrmse_fw:10.5f}  {exp_k:10.5f}")

    max_diff_rpy_np = float(np.max(np.abs(preds_d0 - preds_np)))
    print(f"\n  Max |rpy - numpy|: {max_diff_rpy_np:.3e}  "
          f"({'match' if max_diff_rpy_np < 1e-10 else 'MISMATCH — feedback bug?'})")

    # -----------------------------------------------------------------------
    # Figure: per-step error curves
    # -----------------------------------------------------------------------
    _show_hint()
    n_ax = 2
    fig, axes = plt.subplots(n_ax, 1, figsize=(12, 5 * n_ax))
    steps = np.arange(1, pred_len + 1)

    ax = axes[0]
    ol_aligned = ol_err[spinup - 1: spinup - 1 + pred_len]
    ax.semilogy(steps, ol_aligned, color="green",  lw=0.7, label="open-loop (teacher-forced)")
    ax.semilogy(steps, err_d0,     color="blue",   lw=0.9, label="CL (rpy)")
    ax.semilogy(steps, err_np,     color="cyan",   lw=0.9, ls="--", alpha=0.8, label="CL (numpy)")
    ax.semilogy(steps, err_fw,     color="red",    lw=0.9, ls=":",  alpha=0.8, label="CL (fw-style)")
    for k in _KS:
        ax.axvline(k, color="gray", ls=":", lw=0.5)
    ax.set_ylabel("per-step Euclidean error")
    ax.set_title(f"{system_name} — per-step error  spinup={spinup}")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ks_arr   = np.array(_KS, dtype=float)
    nrmse_rpy_k = [float(np.sqrt(np.mean(err_d0[:k] ** 2))) / norm_std for k in _KS]
    nrmse_np_k  = [float(np.sqrt(np.mean(err_np[:k] ** 2))) / norm_std for k in _KS]
    nrmse_fw_k  = [float(np.sqrt(np.mean(err_fw[:k] ** 2))) / norm_std for k in _KS]
    exp_rmse    = [hp_full[f"l2_gmean_{k}"] for k in _KS]
    ax2.loglog(ks_arr, nrmse_rpy_k, "o-",  color="blue",   label="rpy CL")
    ax2.loglog(ks_arr, nrmse_np_k,  "x--", color="cyan",   label="numpy CL",    alpha=0.8)
    ax2.loglog(ks_arr, nrmse_fw_k,  "^:",  color="red",    label="fw-style CL", alpha=0.8)
    ax2.loglog(ks_arr, exp_rmse,    "s--", color="orange", label="expected (/results)")
    ax2.set_xlabel("steps k")
    ax2.set_ylabel("NRMSE (cumulative over k steps)")
    ax2.set_title(f"{system_name} — NRMSE vs horizon")
    ax2.legend(fontsize=9)

    plt.suptitle(f"task_f: {system_name}  spinup={spinup}  pred_len={pred_len}")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# run_pass — run all tasks in order for a single system
# ---------------------------------------------------------------------------

def _prompt_task(label: str) -> str:
    """Ask whether to run, skip, or quit before a task.

    Returns ``"run"``, ``"skip"``, or ``"quit"``.
    Default (Enter) is run.  Non-interactive context always runs.
    """
    try:
        ans = input(f"\n  {label}\n  [r]un / [s]kip / [q]uit  (default run): ").strip().lower()
    except EOFError:
        return "run"
    if ans in {"s", "skip"}:
        return "skip"
    if ans in {"q", "quit"}:
        return "quit"
    return "run"


def run_pass(system_name: str, arch_components: bool = False,
             solve: str = "reservoirpy", square_method: str = "svd",
             use_new_standards: bool = False) -> None:
    """Run tasks a → e for *system_name*, relying on cached intermediate results.

    Before each task a prompt asks whether to run it, skip it (relying on its
    cached output), or quit the pass entirely.  Press Enter to run.

    The ESN-variant kwargs are forwarded to ``task_c_train`` only; ``task_d``
    and ``task_e`` load whatever model ``task_c`` cached.  ``use_new_standards``
    is forwarded to all data-loading tasks (a, c, d, e, f).  ``task_f``
    (fidelity diagnostic) receives the same ESN-variant kwargs as ``task_c``.

    Parameters
    ----------
    arch_components : bool, default False
        When True, reservoir ``W``/``Win``/``bias`` are replaced with exact
        matrices from ``architectures.basic_res`` / ``basic_in``.
    solve : str, default ``"reservoirpy"``
        ``"reservoirpy"`` keeps reservoirpy's Ridge readout; ``"arch"``
        re-solves with ``architectures.solve_square`` (classic_ridge).
    square_method : str, default ``"svd"``
        ``architectures.solve_square`` square method; used only when
        ``solve="arch"``.  Valid values: ``"svd"``, ``"pinv"``, ``"solve"``,
        ``"cholesky"``.
    use_new_standards : bool, default False
        When True, all data-loading tasks (a, c, d, e) read from the canonical
        ``/new_standards`` CSV instead of the local integrator.
    """
    if system_name not in PASS_CONFIGS:
        raise ValueError(
            f"Unknown system {system_name!r}. "
            f"Choose from {list(PASS_CONFIGS.keys())}."
        )
    data_tag = " [new_standards]" if use_new_standards else ""
    print(f"\n{'#'*60}")
    print(f"#  RUN PASS: {system_name}{data_tag}")
    print(f"{'#'*60}")

    data_kwargs = {"use_new_standards": use_new_standards}
    esn_kwargs  = {
        "arch_components": arch_components,
        "solve": solve,
        "square_method": square_method,
        "use_new_standards": use_new_standards,
    }
    tasks = [
        ("[1/6] task_a_data — load data and compare to new_standards",        task_a_data,       data_kwargs),
        ("[2/6] task_b_hp — find, save, and reload best hyperparameters",     task_b_hp,         data_kwargs),
        ("[3/6] task_c_train — train ESN and overlay predictions",            task_c_train,      esn_kwargs),
        ("[4/6] task_d_valid_time — measure valid time / RMSE vs expected",   task_d_valid_time, data_kwargs),
        ("[5/6] task_e_reg_sweep — sweep ridge beta and compare valid times",  task_e_reg_sweep,  data_kwargs),
        ("[6/6] task_f_rollout — bisect closed-loop fidelity (OL vs CL)",     task_f_rollout,    esn_kwargs),
    ]

    for label, fn, kwargs in tasks:
        action = _prompt_task(label)
        if action == "quit":
            print("  -> pass aborted.")
            return
        if action == "skip":
            print(f"  -> skipped (using cached results).")
            continue
        fn(system_name, **kwargs)

    print(f"\n{'#'*60}")
    print(f"#  PASS COMPLETE: {system_name}")
    print(f"{'#'*60}")


# ---------------------------------------------------------------------------
# run_benchmark — quick multi-system expected vs observed comparison
# ---------------------------------------------------------------------------

def _bench_one(system_name: str, use_new_standards: bool = False) -> dict:
    """Train a default ESN with best-known HPs and compare observed vs expected.

    Sources hyperparameters and expected metrics from the /results CSV via
    ``_load_best_hp``.  Trains fresh (``cache=False``) so the benchmark is
    independent of the ``task_c`` model cache.  Prints a per-system
    expected-vs-observed block (valid time + RMSE at each horizon).

    Parameters
    ----------
    system_name : str
        Key into ``_SYSTEM_CONFIGS_5`` / ``_BENCH_LABELS``.
    use_new_standards : bool, default False
        When True, load train/val data from the canonical ``/new_standards``
        CSV (the exact data the ``/results`` expected metrics were computed on)
        instead of generating data with the local integrator.  Useful for
        isolating ESN/solve fidelity from data-generator differences.

    Returns
    -------
    dict
        Keys: ``system_name``, ``label``, ``reg_method``, ``exp_vt``,
        ``obs_vt``, ``exp_rmse`` (dict k→float), ``obs_rmse`` (dict k→float).
    """
    label      = _BENCH_LABELS[system_name]
    reg_method = _SYSTEM_CONFIGS_5[system_name]["reg_method"]

    _step("loading best HPs + expected metrics from /results CSV")
    hp = _load_best_hp(system_name)

    if use_new_standards:
        _step("loading train + val data from /new_standards CSV")
        train_data, val_block, t_step = _load_standard_data(
            system_name, n_train=_N_TRAIN, n_val=_N_VAL,
        )
    else:
        _step("loading/generating train + val data (cached)")
        train_data, val_block, t_step = generate_train_val(
            system_name, n_train=_N_TRAIN, n_val=_N_VAL,
        )

    # _bench_one bypasses the HP CSV (uses _load_best_hp), so compute the analytic
    # spinup directly here rather than reading it from the saved columns.
    _step("estimating analytic spinup (uniform reservoirpy matrices)")
    est = _analytic_spinup(hp, train_data, arch_components=False)
    spinup = est["spinup"]
    hp["spinup"] = spinup          # inject so _setup_esn / _spinup_for can read it
    print(f"  [bench] spinup={spinup}  (contraction={est['contraction']:.5f})")

    _step("training fresh default ESN (cache=False)")
    model, norm = _setup_esn(system_name, hp, train_data, cache=False)

    pred_len = _pred_len_for(hp, t_step)
    _step(f"sampling {_N_WINDOWS} validation windows (spinup={spinup}, pred_len={pred_len}) and normalising")
    windows = sample_val_windows(
        val_block, n=_N_WINDOWS, spinup=spinup, pred_len=pred_len, seed=1,
    )
    windows_norm = [(w - norm["mean"]) / norm["std"] for w in windows]

    _step(f"running valid_time_multitest (eps=[0.2,0.4], ks={_KS}) on {_N_WINDOWS} windows")
    results = valid_time_multitest(
        model, windows_norm, n_segments=_N_WINDOWS,
        spinup=spinup, eps=[0.2, 0.4], ks=_KS,
        block=pred_len, t_step=t_step,
    )

    exp_vt   = hp["vt_0_2_std_med"]
    obs_vt   = results.get("valid_time_0.2_median", float("nan"))
    vt_ratio = obs_vt / exp_vt if exp_vt > 0 and np.isfinite(obs_vt) else float("nan")

    exp_rmse = {k: hp[f"l2_gmean_{k}"] for k in _KS}
    obs_rmse = {k: results.get(f"rmse_{k}_gmean", float("nan")) for k in _KS}

    noise_reg = float(hp.get("noise_reg", 0.0))
    noise_info = f"  (noise_reg={noise_reg:.2e})" if noise_reg > 0.0 else ""
    print(f"\n  Valid time (0.2×std threshold, median){noise_info}:")
    print(f"    observed  = {obs_vt:.4f}")
    print(f"    expected  = {exp_vt:.4f}")
    print(f"    ratio     = {vt_ratio:.3f}")
    print(f"\n  RMSE gmean by horizon:")
    print(f"  {'k':>6}  {'observed':>12}  {'expected':>12}  {'ratio':>8}")
    for k in _KS:
        obs = obs_rmse[k]
        exp = exp_rmse[k]
        r = obs / exp if exp > 0 and np.isfinite(obs) else float("nan")
        print(f"  {k:6d}  {obs:12.5f}  {exp:12.5f}  {r:8.3f}")

    return {
        "system_name": system_name,
        "label":       label,
        "reg_method":  reg_method,
        "exp_vt":      exp_vt,
        "obs_vt":      obs_vt,
        "exp_rmse":    exp_rmse,
        "obs_rmse":    obs_rmse,
    }


def run_benchmark(systems=None, use_new_standards: bool = False) -> None:
    """Train a default ESN for each benchmark system and print expected vs observed.

    Uses the best hyperparameters from the /results CSV for each system.
    Noise-regularised systems (Lorenz96, KS) are handled automatically — the
    noise path is activated by the non-zero ``noise_reg`` returned by
    ``_load_best_hp``.  No model-cache files are created or modified
    (``cache=False``).  No figures are produced.

    Parameters
    ----------
    systems : list of str, optional
        Subset of ``_BENCH_SYSTEMS`` to evaluate.  Defaults to all five.
        Example: ``run_benchmark(["lorenz_x", "lorenz96"])``.
    use_new_standards : bool, default False
        When True, load train/val data from the canonical ``/new_standards``
        CSV (the exact data the ``/results`` expected metrics were computed on)
        instead of generating data with the local integrator.  Useful for
        isolating ESN/solve fidelity from data-generator differences.
        Example: ``run_benchmark(["mackey_glass"], use_new_standards=True)``.
    """
    systems = systems or _BENCH_SYSTEMS
    data_tag = "new_standards" if use_new_standards else "generated"

    print(f"\n{'#'*60}")
    print(f"#  BENCHMARK: expected vs observed  ({len(systems)} system(s), data={data_tag})")
    print(f"{'#'*60}")

    rows = []
    for sys_name in systems:
        print(f"\n{'='*60}")
        print(f"  {_BENCH_LABELS.get(sys_name, sys_name)}")
        print(f"{'='*60}")
        try:
            row = _bench_one(sys_name, use_new_standards=use_new_standards)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] {sys_name}: {exc}")

    if not rows:
        print("\n  No results to display.")
        return

    # Consolidated summary table
    print(f"\n\n{'='*60}")
    print(f"  CONSOLIDATED SUMMARY")
    print(f"{'='*60}")

    col_w   = 24
    ks_strs = [f"r@{k}" for k in _KS]
    header  = (
        f"  {'System':<{col_w}}  {'Reg':>5}  {'Exp VT':>8}  "
        f"{'Obs VT':>8}  {'VT ratio':>8}  " +
        "  ".join(f"{s:>6}" for s in ks_strs)
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for row in rows:
        exp_vt = row["exp_vt"]
        obs_vt = row["obs_vt"]
        vt_r   = (obs_vt / exp_vt
                  if exp_vt > 0 and np.isfinite(obs_vt) else float("nan"))

        rmse_ratios = []
        for k in _KS:
            obs = row["obs_rmse"][k]
            exp = row["exp_rmse"][k]
            r = obs / exp if exp > 0 and np.isfinite(obs) else float("nan")
            rmse_ratios.append(r)

        rmse_cols = "  ".join(
            f"{r:6.3f}" if np.isfinite(r) else f"{'nan':>6}"
            for r in rmse_ratios
        )
        print(
            f"  {row['label']:<{col_w}}  {row['reg_method']:>5}  "
            f"{exp_vt:8.3f}  {obs_vt:8.3f}  {vt_r:8.3f}  {rmse_cols}"
        )

    print()


# ---------------------------------------------------------------------------
# Ridge-β sweep: load HPs → analytic spinup → pure-reservoirpy β sweep →
# select best β → overwrite HP CSV → print ratio table
# ---------------------------------------------------------------------------

def _ridge_sweep_one(
    system_name: str,
    use_new_standards: bool = False,
    write_csv: bool = True,
    seed: int = 1,
) -> dict:
    """Load HPs, compute analytic spinup, run a pure-reservoirpy ridge-β sweep,
    select the best β, write the result to the HP CSV, and return a summary dict.

    Sweep components are entirely reservoirpy (``_setup_esn(solve="reservoirpy",
    arch_components=False)``). The only architectures dependency is the analytic-
    spinup estimator (``architectures.analytic_synchrony``), which has no
    reservoirpy equivalent and is a separate, pre-sweep step — mirroring
    ``task_b_hp``.

    β grid by regularisation method:

    * ``"ridge"`` systems (lorenz_x, lorenz_f, mackey_glass):
      ``_RIDGE_BETAS = [0, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-18]``.
    * ``"noise"`` systems (lorenz96, kuramoto_sivashinsky):
      three noise_reg levels around the baseline recommendation:
      ``[rec×0.01, rec, rec×100]``; ridge is pinned to 0 for all candidates.

    The best regularisation value is the one that maximises
    ``valid_time_0.2_median``; ties (NaN) are treated as −∞.  When
    ``write_csv=True`` the row for *system_name* in ``best_hp.csv`` is
    overwritten with the same schema as ``task_b_hp``, storing the winning
    value in ``ridge`` (ridge systems) or ``noise_reg`` (noise systems) and
    zeroing the other field.

    Parameters
    ----------
    system_name : str
        Key into ``_SYSTEM_CONFIGS_5`` / ``_BENCH_LABELS``.
    use_new_standards : bool, default False
        Load data from the canonical ``/new_standards`` CSV when True, else
        from the local integrator (same choice as ``run_benchmark``).
    write_csv : bool, default True
        When True, overwrite the ``{system_name}`` row in ``best_hp.csv`` with the swept HPs.
    seed : int, default 1
        RNG seed for ``sample_val_windows``.

    Returns
    -------
    dict with keys:
        ``system_name``, ``label``, ``reg_method``,
        ``best_beta``, ``best_vt``, ``baseline_vt``, ``ratio``,
        ``sweep`` (list of ``(beta, vt)`` tuples, one per candidate).
    """
    label      = _BENCH_LABELS.get(system_name, system_name)
    reg_method = _SYSTEM_CONFIGS_5[system_name]["reg_method"]

    print(f"\n{'='*60}")
    print(f"  _ridge_sweep_one: {label}")
    print(f"{'='*60}")

    # 1. Best HPs from baseline CSV.
    _step("reading best hyperparameters from full_losses CSV")
    hp = _load_best_hp(system_name)

    # 2. Physical time step (includes stepper_substeps for Mackey-Glass).
    sys_obj  = _system_factory(_SYSTEM_CONFIGS_5[system_name])
    substeps = _SYSTEM_CONFIGS_5[system_name].get("stepper_substeps", 1)
    t_step   = sys_obj.t_step * hp["subsample_t"] * substeps

    # 3. Training + validation data.
    _step("loading train + val data")
    if use_new_standards:
        train_data, val_block, _ = _load_standard_data(
            system_name, n_train=_N_TRAIN, n_val=_N_VAL,
        )
    else:
        train_data, val_block, _ = generate_train_val(
            system_name, n_train=_N_TRAIN, n_val=_N_VAL,
        )

    # 4. Analytic spinup — both builders so the CSV schema stays identical to
    #    task_b_hp (columns spinup and spinup_arch).
    _step("estimating analytic spinup (uniform reservoirpy matrices)")
    est_uniform = _analytic_spinup(hp, train_data, arch_components=False)
    _step("estimating analytic spinup (architectures basic_res/basic_in matrices)")
    est_arch    = _analytic_spinup(hp, train_data, arch_components=True)
    spinup      = est_uniform["spinup"]
    print(f"  spinup (uniform)={spinup}  spinup_arch={est_arch['spinup']}"
          f"  (contraction={est_uniform['contraction']:.5f})")

    # 5. Prediction window and validation windows (sampled once, reused per β).
    pred_len = _pred_len_for(hp, t_step)
    _step(f"sampling {_N_WINDOWS} validation windows "
          f"(spinup={spinup}, pred_len={pred_len})")
    windows = sample_val_windows(
        val_block, n=_N_WINDOWS, spinup=spinup, pred_len=pred_len, seed=seed,
    )

    # 6. Regularisation grid.
    #    Ridge systems: sweep ridge β, keep noise_reg=0.
    #    Noise systems: sweep noise_reg, keep ridge=0 (noise is the native
    #    regulariser; the three points bracket the baseline recommendation).
    if reg_method == "noise":
        rec = float(hp.get("noise_reg", 1e-4))
        reg_vals = sorted({rec * 0.01, rec, rec * 100.0} - {0.0})
        print(f"  noise system — noise_reg grid around baseline={rec:.2e}: {reg_vals}")
    else:
        reg_vals = _RIDGE_BETAS
        print(f"  ridge system — ridge-β grid: {reg_vals}")

    # 7. Sweep loop — pure reservoirpy.
    reg_label = "noise_reg" if reg_method == "noise" else "ridge_β"
    _step(f"sweeping {len(reg_vals)} {reg_label} values "
          f"(solve=reservoirpy, arch_components=False)")
    print(f"  {reg_label:>12}  {'valid_time_0.2_med':>20}")
    print(f"  {'-'*12}  {'-'*20}")

    sweep = []
    best_reg, best_vt = reg_vals[0], float("-inf")

    for reg_val in reg_vals:
        hp_b = dict(hp)
        if reg_method == "noise":
            hp_b["noise_reg"] = reg_val
            hp_b["ridge"]     = 0.0
        else:
            hp_b["ridge"]     = reg_val
            hp_b["noise_reg"] = 0.0
        hp_b["spinup"] = spinup             # so _spinup_for reads the analytic value

        model, norm = _setup_esn(
            system_name, hp_b, train_data,
            arch_components=False, solve="reservoirpy", cache=False,
        )
        windows_norm = [(w - norm["mean"]) / norm["std"] for w in windows]

        res = valid_time_multitest(
            model, windows_norm, n_segments=_N_WINDOWS,
            spinup=spinup, eps=[0.2, 0.4], ks=_KS,
            block=pred_len, t_step=t_step,
        )
        vt = res.get("valid_time_0.2_median", float("nan"))
        sweep.append((reg_val, vt))

        rv_str = f"{reg_val:.2e}" if reg_val > 0 else "0"
        vt_str = f"{vt:.4f}" if np.isfinite(vt) else "nan"
        print(f"  {rv_str:>12}  {vt_str:>20}")

        if np.isfinite(vt) and vt > best_vt:
            best_vt, best_reg = vt, reg_val

    # 8. Best regularisation value.
    print(f"\n  Best {reg_label} = {best_reg:.2e}  →  valid_time_0.2_median = {best_vt:.4f}")

    # 9. Overwrite the HP CSV (same schema as task_b_hp).
    if write_csv:
        save_row = {
            "time_step":      t_step,
            "size":           hp["size"],
            "seed":           hp["seed"],
            "leak":           hp["leak"],
            "radius":         hp["radius"],
            "in_scale":       hp["in_scale"],
            "bias_scale":     hp["bias_scale"],
            "spinup":         est_uniform["spinup"],
            "spinup_arch":    est_arch["spinup"],
            "sparse":         hp["sparse"],
            "subsample_t":    hp["subsample_t"],
            "ridge":          0.0 if reg_method == "noise" else best_reg,
            "noise_reg":      best_reg if reg_method == "noise" else 0.0,
            "vt_0_2_std_med": hp["vt_0_2_std_med"],
            "l2_gmean_4":     hp["l2_gmean_4"],
            "l2_gmean_16":    hp["l2_gmean_16"],
            "l2_gmean_64":    hp["l2_gmean_64"],
            "l2_gmean_256":   hp["l2_gmean_256"],
        }
        _step("overwriting HP CSV with swept HPs")
        _write_hp_csv(system_name, save_row)
        print(f"  Written: {_hp_csv_path()} (row: {system_name})")

    # 10. Ratio vs baseline.
    baseline_vt = hp["vt_0_2_std_med"]
    ratio = best_vt / baseline_vt if baseline_vt > 0 and np.isfinite(best_vt) else float("nan")

    print(f"\n  Sweep best VT = {best_vt:.4f}  |  baseline VT = {baseline_vt:.4f}"
          f"  |  ratio = {ratio:.3f}")

    return {
        "system_name":  system_name,
        "label":        label,
        "reg_method":   reg_method,
        "best_reg":     best_reg,
        "best_vt":      best_vt,
        "baseline_vt":  baseline_vt,
        "ratio":        ratio,
        "sweep":        sweep,
    }


def run_ridge_sweep(
    systems=None,
    use_new_standards: bool = False,
    write_csv: bool = True,
) -> list:
    """Run a pure-reservoirpy ridge-β sweep for each system and print a summary.

    For each system, calls ``_ridge_sweep_one`` which: loads HPs, computes the
    analytic spinup, sweeps a β grid, selects the best β by median valid time
    at the 0.2×std threshold, and (when ``write_csv=True``) overwrites the HP
    CSV so that downstream tasks pick up the swept β.

    The summary table shows sweep-best vs. baseline VPT and their ratio, giving
    an at-a-glance view of how well a pure-reservoirpy ridge regression tracks
    the framework's best performance.

    Produces no figures (headless-safe).

    Parameters
    ----------
    systems : list of str or None
        Subset of ``_BENCH_SYSTEMS`` to evaluate.  Defaults to all five.
    use_new_standards : bool, default False
        Load data from the canonical ``/new_standards`` CSV when True.
    write_csv : bool, default True
        When True, overwrite each system's HP CSV with the swept HPs.

    Returns
    -------
    list of dicts (one per system) as returned by ``_ridge_sweep_one``.
    """
    systems   = systems or _BENCH_SYSTEMS
    data_tag  = "new_standards" if use_new_standards else "generated"
    write_tag = "writing CSVs" if write_csv else "dry-run, no CSV write"

    print(f"\n{'#'*60}")
    print(f"#  RIDGE-β SWEEP: {len(systems)} system(s)  "
          f"data={data_tag}  {write_tag}")
    print(f"{'#'*60}")

    rows = []
    for sys_name in systems:
        try:
            row = _ridge_sweep_one(
                sys_name,
                use_new_standards=use_new_standards,
                write_csv=write_csv,
            )
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            print(f"\n  [ERROR] {sys_name}: {exc}")

    if not rows:
        print("\n  No results to display.")
        return rows

    # Consolidated summary table.
    col_w = 24
    print(f"\n\n{'='*60}")
    print(f"  RIDGE-β SWEEP SUMMARY")
    print(f"{'='*60}")
    header = (
        f"  {'System':<{col_w}}  {'Reg':>5}  {'Best β':>10}  "
        f"{'Sweep VT':>9}  {'Baseline VT':>11}  {'Ratio':>7}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for row in rows:
        bvt  = row["best_vt"]
        blvt = row["baseline_vt"]
        reg  = row["best_reg"]
        r    = row["ratio"]
        reg_str = f"{reg:.2e}" if reg > 0 else "0"
        bvt_str = f"{bvt:.3f}" if np.isfinite(bvt) else "nan"
        r_str   = f"{r:.3f}"   if np.isfinite(r)   else "nan"
        print(
            f"  {row['label']:<{col_w}}  {row['reg_method']:>5}  "
            f"{reg_str:>10}  {bvt_str:>9}  {blvt:>11.3f}  {r_str:>7}"
        )

    print()
    return rows


# ---------------------------------------------------------------------------
# task_g — Benchmark-faithful valid-time test via loss_funcs.PredictionLoss
# ---------------------------------------------------------------------------


class _PredictionLossESN:
    """Adapt a trained reservoirpy ESN to the interface expected by PredictionLoss.

    ``loss_funcs.PredictionLoss.evaluate`` touches three things on the model:
    ``model.recommended_spinup``, ``model.reset_state()``, and
    ``model.predict(us_init_df, steps, reset=..., depro=True)``.  This class
    wraps an already-trained ESN, extracts the numpy weight matrices, and
    provides those three touchpoints using the same closed-loop semantics as
    ``architectures.Operator.predict`` (confirmed by task_f step-2c).

    Closed-loop seeding mirrors ``architectures.Operator.predict`` with
    ``reset=True``: warm up over ``us_init[0 .. len-2]``; the loop's final
    return value (readout from state-after-absorbing ``us_init[len-2]``)
    seeds the closed-loop — ``us_init[-1]`` is **not** fed open-loop.
    With ``reset=False`` the reservoir state is preserved across calls
    (used by ``PredictionLoss`` to extend a rollout in blocks).

    Parameters
    ----------
    model : ESN
        A fitted reservoirpy ESN with ``input_to_readout=True`` and
        ``Wout`` / ``bias`` injected by ``_arch_ridge_solve``.
    recommended_spinup : int
        Warmup length reported to ``PredictionLoss``; also used as the
        training discard via ``_setup_esn``.
    t_step : float
        Physical time per step; used to build the output DataFrame index.
    columns : list of str
        Variable names for both input and output DataFrames (must match the
        ``val_df`` columns passed to ``PredictionLoss``).
    """

    def __init__(self, model, recommended_spinup, t_step, columns):
        res = model.reservoir
        self.W = np.asarray(res.W, dtype=float)
        self.Win = np.asarray(res.Win, dtype=float)
        self.b_res = np.asarray(res.bias, dtype=float).ravel()
        self.lr = float(res.lr)
        # Wout shape: (units + d_in, d_out) — injected by _arch_ridge_solve
        self.Wout = np.asarray(model.readout.Wout, dtype=float)
        self.b_out = np.asarray(model.readout.bias, dtype=float).ravel()
        self.recommended_spinup = int(recommended_spinup)
        self.t_step = float(t_step)
        self.columns = list(columns)
        self.preprocess = None  # PredictionLoss checks this attribute
        self.r = np.zeros(self.W.shape[0])

    def reset_state(self):
        """Zero the reservoir state."""
        self.r = np.zeros(self.W.shape[0])

    def _step(self, u):
        """Advance one timestep; update ``self.r``; return the readout output."""
        self.r = (
            (1.0 - self.lr) * self.r
            + self.lr * np.tanh(self.W @ self.r + self.Win @ u + self.b_res)
        )
        return np.concatenate([self.r, u]) @ self.Wout + self.b_out

    def predict(self, us_init, steps, reset=True, depro=True, **kw):
        """Run a closed-loop forecast, mirroring ``architectures.Operator.predict``.

        Parameters
        ----------
        us_init : pd.DataFrame
            Warmup / seed window.  With ``reset=True``, rows ``[0 .. len-2]``
            are absorbed open-loop and the readout output from the final
            warmup step seeds the closed loop (``us_init[-1]`` is not fed).
            With ``reset=False``, ``us_init.iloc[-1]`` is used as the
            seed input and the reservoir state is not reset.
        steps : int
            Number of closed-loop steps to generate.
        reset : bool
            When True, zero the reservoir state before warming up.
        depro : bool
            Ignored (present for interface compatibility with PredictionLoss).

        Returns
        -------
        pd.DataFrame
            Shape ``(steps, len(self.columns))``, indexed starting at
            ``us_init.index[-1] + t_step``.  Truncated on non-finite blow-up.
        """
        arr = np.asarray(us_init[self.columns], dtype=float)
        start_t = float(us_init.index[-1])

        if reset:
            self.reset_state()
            # absorb us_init[0 .. len-2]; seed = last warmup readout
            u = arr[-1]  # fallback for len == 1
            for i in range(len(arr) - 1):
                u = self._step(arr[i])
        else:
            # continue from persisted self.r; seed = last value of previous call
            u = arr[-1]

        out = []
        for _ in range(int(steps)):
            u = self._step(u)
            if not np.isfinite(u).all():
                break
            out.append(u.copy())

        idx = start_t + np.arange(1, len(out) + 1) * self.t_step
        return pd.DataFrame(out, columns=self.columns, index=idx)


def task_g_benchmark_compare(
    system_name="mackey_glass",
    n_val=80000,
    max_trials=200,
    arch_components=True,
    solve="arch",
    square_method="svd",
    spinup=930,
):
    """Evaluate a reservoirpy ESN using the same procedure as the /reservoir benchmark.

    Addresses the measurement-setup incompatibility identified in item-0: the
    benchmark (``vt_0_2_std_med`` in ``mg_2221_cr_an0.csv``) was produced by
    ``loss_funcs.PredictionLoss`` over ~80k val steps with 200 random-start
    trials and a 3000-step prediction cap, whereas ``task_d`` uses 50 contiguous
    windows over a 10k val prefix with a ~930-step cap.  This function runs the
    exact benchmark evaluation code against the reservoirpy model so that the
    ratio (observed / expected) reflects model quality rather than measurement
    setup.

    The ESN uses architecture.basic_res / basic_in components (``arch_components``
    default True) and architectures.solve_square for the readout (``solve``
    default "arch"), matching the /reservoir training that produced the benchmarks.

    Parameters
    ----------
    system_name : str
        Key into ``_SYSTEM_CONFIGS_5`` / ``PASS_CONFIGS`` (default: "mackey_glass").
    n_val : int
        Validation steps to load from the new_standards CSV; default 80000
        matches the ~80k val set used by ``training.load_data`` in /reservoir.
    max_trials : int
        Random-start trials for ``PredictionLoss``; default 200 matches benchmark.
    arch_components : bool
        Use ``architectures.basic_res`` / ``basic_in`` matrices (default True).
    solve : str
        Readout solve: "arch" (architectures.solve_square) or "reservoirpy"
        (default "arch").
    square_method : str
        Passed to ``_arch_ridge_solve`` when ``solve="arch"`` (default "svd").
    spinup : int
        Warmup steps reported to ``PredictionLoss`` and used as training discard
        (default 930 = ``spinup_arch`` for Mackey-Glass).

    Returns
    -------
    pd.Series
        Full output from ``PredictionLoss.evaluate``, keyed by metric name.
    """
    # -- lazy import: reservoir/ is not on the module path at import time ------
    reservoir_dir = os.path.normpath(
        os.path.join(_THIS_DIR, "../../../../reservoir")
    )
    if reservoir_dir not in sys.path:
        sys.path.insert(0, reservoir_dir)
    import loss_funcs  # noqa: PLC0415

    print(f"\n{'=' * 60}")
    print(f"  task_g_benchmark_compare: {system_name}")
    print(
        f"  arch_components={arch_components}  solve={solve!r}"
        f"  spinup={spinup}  n_val={n_val}  max_trials={max_trials}"
    )
    print(f"{'=' * 60}")

    _step("loading hyperparameters from CSV (task_b output)")
    hp = _load_hp_csv(system_name)

    _step(f"loading train ({_N_TRAIN}) + val ({n_val}) data from /new_standards CSV")
    train_data, val_block, t_step = _load_standard_data(
        system_name, n_train=_N_TRAIN, n_val=n_val
    )

    _step("training ESN (no cache)")
    model, norm = _setup_esn(
        system_name,
        hp,
        train_data,
        arch_components=arch_components,
        solve=solve,
        square_method=square_method,
        cache=False,
    )

    # Column names for DataFrames: ref_col for 1-D, x0/x1/… for multi-dim.
    ref_col = PASS_CONFIGS[system_name]["ref_col"]
    d = train_data.shape[1]
    cols = [ref_col] if d == 1 else [f"x{i}" for i in range(d)]

    _step("building adapter and normalising validation data")
    adapter = _PredictionLossESN(
        model,
        recommended_spinup=spinup,
        t_step=t_step,
        columns=cols,
    )

    val_norm = (val_block - norm["mean"]) / np.maximum(norm["std"], 1e-8)
    val_idx = np.arange(len(val_norm), dtype=float) * t_step
    val_df = pd.DataFrame(val_norm, columns=cols, index=val_idx)

    _step(
        f"running PredictionLoss.evaluate "
        f"({max_trials} trials, spinup={spinup}, "
        f"forward_steps=300, max_multiples=10)"
    )
    pl = loss_funcs.PredictionLoss()
    result = pl.evaluate(
        adapter,
        [val_df],
        t_forward=[4, 16, 64, 256],
        max_trials=max_trials,
        epsilon=[0.2, 0.4],
        spinup=spinup,  # overridden by adapter.recommended_spinup anyway
    )

    # ------------------------------------------------------------------
    # Comparison report (mirrors task_d format)
    # ------------------------------------------------------------------
    ks = [4, 16, 64, 256]
    obs_vt = result.get("valid time 0.2 std med", float("nan"))
    exp_vt = hp["vt_0_2_std_med"]
    vt_ratio = obs_vt / exp_vt if exp_vt > 0 else float("nan")

    print(f"\n  Valid time (0.2×std threshold, median over {max_trials} trials):")
    print(f"    observed  = {obs_vt:.4f}")
    print(f"    expected  = {exp_vt:.4f}  (from /results CSV)")
    print(f"    ratio     = {vt_ratio:.3f}")

    print(f"\n  RMSE gmean by horizon ({max_trials} trials):")
    print(f"  {'k':>6}  {'observed':>12}  {'expected':>12}  {'ratio':>8}")
    for k in ks:
        obs = result.get(f"{k} step L2 error gmean", float("nan"))
        exp = hp[f"l2_gmean_{k}"]
        r = obs / exp if exp > 0 and np.isfinite(obs) else float("nan")
        print(f"  {k:6d}  {obs:12.5f}  {exp:12.5f}  {r:8.3f}")

    print()
    return result


# ---------------------------------------------------------------------------
# task_h — native end-to-end benchmark rerun via reservoir/training.py
# ---------------------------------------------------------------------------

def task_h_benchmark_rerun(system_name: str = "mackey_glass", verbose: bool = True):
    """Validate the /reservoir benchmark by rerunning it natively via training.main().

    Reconstructs the best-trial parameters of the mg_2221_cr_an0 batch and drives
    ``reservoir/training.py`` end-to-end (architectures.py components +
    loss_funcs.PredictionLoss + the default multireg ridge sweep).  Confirms that the
    ``ridge=1e-9`` row of the returned DataFrame reproduces
    ``vt_0_2_std_med = 715`` and the four L2-gmean values stored in
    ``best_hp.csv`` (mackey_glass row).

    Nothing from reservoirpy is used — this is a pure self-consistency check of the
    benchmark.  If it matches, the 715 baseline is validated and the 60 % reservoirpy
    gap is real (or measurement); if it does not, the benchmark CSV itself is suspect.

    Single-trial, no joblib — safe to run directly.
    """
    assert system_name == "mackey_glass", (
        "task_h_benchmark_rerun is wired for mackey_glass only"
    )
    hp = _load_hp_csv(system_name)

    # Lazy-add reservoir/ to sys.path (same pattern as _arch_components / task_g).
    reservoir_dir = os.path.normpath(os.path.join(_THIS_DIR, "../../../../reservoir"))
    if reservoir_dir not in sys.path:
        sys.path.insert(0, reservoir_dir)
    import training as tr  # noqa: PLC0415

    # Reconstruct the best-trial param Series exactly as the batch ran it.
    # Key choices (from queues/4_24_26_a_esn2221_multireg_main.csv + best-row CSV):
    #   • 'reg ridge' (not 'regularization') is the current key for the beta
    #     array after the standardise-keys commit.  Pass a 1-element list string
    #     so exactly ridge=1e-9 is solved; training.py's is_multi check still
    #     looks for model.regularizations (renamed to model.reg_keys), so the
    #     result is a single-row DataFrame with no "regularization" column.
    #   • test data=True  → prepare_data uses split [0.7, 0.2, 0.1].
    #   • path is relative → must run with CWD = reservoir_dir.
    target_beta = hp["ridge"]   # 1e-9 from best_hp CSV
    params = pd.Series(
        {
            "leak":            hp["leak"],
            "radius":          hp["radius"],
            "in scale":        hp["in_scale"],
            "bias scale":      hp["bias_scale"],
            "sparse":          hp["sparse"],
            "seed":            hp["seed"],
            "size":            hp["size"],
            "n training steps": _N_TRAIN,
            "architecture":    "TradRC",
            "loss 0":          "prediction",
            "loss test 0":     "prediction",
            "regression":      "ridge",
            "solve shape":     "square",
            "square method":   "svd",
            "reg ridge":       f"[{target_beta}]",  # str → parsed by default_setter
            "added noise":     0.0,
            "test data":       True,
            "subsample t":     1,
            "path":            "datasets/new_standards/mackeyglass_det.csv",
        },
        name=0,
    )

    cwd = os.getcwd()
    try:
        os.chdir(reservoir_dir)          # relative path resolution for main()
        result = tr.main(params, print_progress=verbose)
    finally:
        os.chdir(cwd)

    # Single-reg result: one row, no "regularization" column.
    row = result.iloc[0] if isinstance(result, pd.DataFrame) else result

    vt_obs = float(row["prediction valid time 0.2 std med"])
    vt_exp = float(hp["vt_0_2_std_med"])          # expected: 715.0
    l2_obs = {k: float(row[f"prediction {k} step L2 error gmean"])
              for k in (4, 16, 64, 256)}
    l2_exp = {k: float(hp[f"l2_gmean_{k}"])
              for k in (4, 16, 64, 256)}

    if verbose:
        print("\n── ridge=1e-9 vs benchmark ─────────────────────────────────────────")
        print(f"  {'metric':32s}  {'observed':>12}  {'expected':>12}  {'ratio':>8}")
        print(f"  {'-'*32}  {'-'*12}  {'-'*12}  {'-'*8}")
        vt_ratio = vt_obs / vt_exp if vt_exp > 0 else float("nan")
        print(f"  {'vt 0.2 std med (steps)':32s}  {vt_obs:12.1f}  {vt_exp:12.1f}"
              f"  {vt_ratio:8.3f}")
        for k in (4, 16, 64, 256):
            obs = l2_obs[k]
            exp = l2_exp[k]
            ratio = obs / exp if exp > 0 and np.isfinite(obs) else float("nan")
            print(f"  {f'L2 gmean k={k}':32s}  {obs:12.6f}  {exp:12.6f}  {ratio:8.3f}")
        print()
        verdict = "✓ MATCH" if abs(vt_ratio - 1.0) < 0.02 else "✗ MISMATCH"
        print(f"  Benchmark validation: {verdict}  (VT ratio = {vt_ratio:.4f})")
        print()

    return result


def task_i_openloop_state_compare(
    system_name: str = "mackey_glass",
    n_steps: int = None,
    save_dir: str = None,
    show: bool = False,
) -> dict:
    """Compare open-loop reservoir states between reservoirpy and architectures.EchoStateNetwork.

    Drives both frameworks with the **same** normalized input sequence and the
    **same** W / Win / bias (from ``_arch_components``), then computes the
    difference between the collected state trajectories.  If the update equations
    are algebraically equivalent (which they should be), the states will match to
    ~machine epsilon.

    This is Round-2 item 2 of the performance-gap investigation documented in
    ``reservoirpy_research/lyapunov/instructions/26_6_5_performance.md``.

    Parameters
    ----------
    system_name : str
        Key into ``_load_hp_csv`` / ``_load_standard_data``.  Only
        ``"mackey_glass"`` is tested; others may work if HP CSV exists.
    n_steps : int, optional
        Truncate the training block to this many steps (speeds up the Python
        ``next_step`` loop on the arch side).  Default: full training block.
    save_dir : str, optional
        Directory for the PNG time-series plot.  Default:
        ``<_THIS_DIR>/figures/``.
    show : bool
        Call ``plt.show()`` after saving the figure (default False).

    Returns
    -------
    dict
        Keys: ``"global_rel_diff"``, ``"max_step_rel"``, ``"mean_step_rel"``,
        ``"states_resy"``, ``"states_arch"``, ``"fig_path"``, ``"passed"``.
    """
    # ---- Setup ----------------------------------------------------------------
    hp = _load_hp_csv(system_name)
    train_data, _, _ = _load_standard_data(system_name)
    d_in = train_data.shape[1]
    size = int(hp["size"])

    if n_steps is not None:
        train_data = train_data[: n_steps + 1]  # +1 so we get n_steps x/y pairs

    # Normalize exactly as _setup_esn does.
    x_mean = train_data[:-1].mean(axis=0)
    x_std = np.maximum(train_data[:-1].std(axis=0), 1e-8)
    x_norm = (train_data[:-1] - x_mean) / x_std
    y_norm = (train_data[1:] - x_mean) / x_std
    T = len(x_norm)

    # Identical matrices given to both frameworks.
    W_a, Win_a, bias_a = _arch_components(hp, d_in)

    # ---- reservoirpy side -----------------------------------------------------
    model_r = build_esn(hp, d_in, T)
    model_r.reservoir.W = W_a
    model_r.reservoir.Win = Win_a
    model_r.reservoir.bias = bias_a
    # fit() initializes node shapes; we discard the readout result.
    if not model_r.initialized:
        model_r.initialize(x_norm, y_norm)
    model_r.reset()
    model_r.fit(x_norm, y_norm)
    model_r.reservoir.reset()
    states_resy = np.asarray(model_r.reservoir.run(x_norm), dtype=float)  # (T, size)

    # ---- architectures side ---------------------------------------------------
    reservoir_dir = os.path.normpath(os.path.join(_THIS_DIR, "../../../../reservoir"))
    if reservoir_dir not in sys.path:
        sys.path.insert(0, reservoir_dir)
    import architectures as _arch  # noqa: PLC0415

    # w_in for EchoStateNetwork is [Win | bias] so that next_step computes
    # w_in @ [u - input_means, 1] == Win @ u + bias  (with input_means=0).
    w_in_combined = np.column_stack([Win_a, bias_a.reshape(-1, 1)])  # (size, d_in+1)
    feat_len = size + d_in + 1  # [r, u, 1] feature vector length
    arch_params = pd.Series({
        "size": size,
        "leak": float(hp["leak"]),
        "name": "arch_cmp",
        "inputs": "all",
        "fit to": "all",
    })
    esn = _arch.EchoStateNetwork(
        arch_params,
        w_in=w_in_combined,
        w_res=W_a,
        v_leak=float(hp["leak"]),
        in_means=np.zeros(d_in),
        out_means=np.zeros(d_in),
        w_out=np.zeros((d_in, feat_len)),  # dummy — return value of next_step ignored
    )
    esn.input_stds = np.ones(d_in)  # not set by __init__; only set during train()
    esn.reset_state()

    states_arch = np.empty((T, size), dtype=float)
    for t, u in enumerate(x_norm):
        esn.next_step(t, u)  # updates esn.r in place; return value ignored
        states_arch[t] = esn.get_state()

    # ---- Compare --------------------------------------------------------------
    diff = np.abs(states_resy - states_arch)
    norm_arch = np.linalg.norm(states_arch, "fro")
    global_rel_diff = np.linalg.norm(diff, "fro") / (norm_arch + 1e-300)

    eps = 1e-300
    step_norms_arch = np.linalg.norm(states_arch, axis=1)  # (T,)
    step_norms_diff = np.linalg.norm(diff, axis=1)          # (T,)
    step_rel = step_norms_diff / (step_norms_arch + eps)     # (T,)

    max_step_rel = float(np.max(step_rel))
    mean_step_rel = float(np.mean(step_rel))
    passed = global_rel_diff < 1e-8

    print(f"\n── Open-loop state comparison: {system_name} ({'PASS' if passed else 'FAIL'}) ──")
    print(f"  Trajectory length       : {T} steps")
    print(f"  Global relative diff    : {global_rel_diff:.3e}  "
          f"(||resy - arch||_F / ||arch||_F)")
    print(f"  Per-step max  rel error : {max_step_rel:.3e}")
    print(f"  Per-step mean rel error : {mean_step_rel:.3e}")
    print(f"  Per-step final rel error: {step_rel[-1]:.3e}")
    if global_rel_diff > 1e-10:
        print(f"  ** NONZERO DISCREPANCY DETECTED — ratio = {global_rel_diff:.3e} **")
    else:
        print(f"  States match to machine-epsilon precision.")
    verdict = "PASS" if passed else "FAIL"
    print(f"  Result: {verdict}  (threshold: global_rel_diff < 1e-8)")

    # ---- Plot -----------------------------------------------------------------
    if save_dir is None:
        save_dir = os.path.join(_THIS_DIR, "figures")
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, f"openloop_state_compare_{system_name}.png")

    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt  # noqa: PLC0415

    fig, ax = _plt.subplots(figsize=(10, 4))
    ax.semilogy(step_rel, lw=0.8, color="steelblue")
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$\Vert r_{\rm resy}[t] - r_{\rm arch}[t] \Vert \;/\; \Vert r_{\rm arch}[t] \Vert$")
    ax.set_title(
        f"Open-loop state error — {system_name}\n"
        f"global rel diff = {global_rel_diff:.2e}  ({verdict})"
    )
    ax.grid(True, which="both", alpha=0.3)
    _plt.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"  Figure saved → {fig_path}")
    if show:
        _plt.show()
    _plt.close(fig)

    return {
        "global_rel_diff": global_rel_diff,
        "max_step_rel": max_step_rel,
        "mean_step_rel": mean_step_rel,
        "states_resy": states_resy,
        "states_arch": states_arch,
        "fig_path": fig_path,
        "passed": passed,
    }


def task_j_closedloop_prediction_compare(
    system_name: str = "mackey_glass",
    spinup: int = 1000,
    n_pred: int = 1000,
    ridge_beta: float = None,
    save_dir: str = None,
    show: bool = False,
) -> dict:
    """Compare closed-loop predictions of a reservoirpy ESN and an architectures ESN.

    Both models receive identical components (W, Win, bias) and an identical
    w_out (computed by ``architectures.solve_ridge_rect`` and imported into the
    reservoirpy readout).  They are warmed up on the same ``spinup``-step
    validation trajectory, then rolled out autoregressively for ``n_pred`` steps.
    Predictions must match to ~machine epsilon if the closed-loop prediction
    path is equivalent.

    This is Round-2 item 3 of the performance-gap investigation documented in
    ``reservoirpy_research/lyapunov/instructions/26_6_5_performance.md``.

    Parameters
    ----------
    system_name : str
        Key into ``_load_hp_csv`` / ``_load_standard_data``.
    spinup : int
        Open-loop warmup steps on validation data.  Default 1000.
    n_pred : int
        Number of closed-loop prediction steps.  Default 1000.
    ridge_beta : float, optional
        Ridge regularisation coefficient for the arch solver.  Defaults to
        ``hp["ridge"]`` from the benchmark HP CSV.
    save_dir : str, optional
        Directory for PNG output.  Default: ``<_THIS_DIR>/figures/``.
    show : bool
        Call ``plt.show()`` after saving (default False).

    Returns
    -------
    dict
        Keys: ``"global_rel_diff"``, ``"max_step_rel"``, ``"mean_step_rel"``,
        ``"preds_resy"``, ``"preds_arch"``, ``"fig_path"``, ``"passed"``.
    """
    # ---- Data + normalization ------------------------------------------------
    hp = _load_hp_csv(system_name)
    train_data, val_block, _ = _load_standard_data(system_name)
    d_in = train_data.shape[1]
    size = int(hp["size"])

    # Train-set normalization stats (same as _setup_esn / task_i).
    x_mean = train_data[:-1].mean(axis=0)
    x_std = np.maximum(train_data[:-1].std(axis=0), 1e-8)
    x_norm = (train_data[:-1] - x_mean) / x_std
    y_norm = (train_data[1:] - x_mean) / x_std
    T = len(x_norm)

    # Validation spinup in the same normalized space.
    val_norm = (val_block - x_mean) / x_std
    spin = val_norm[:spinup]    # (spinup, d_in)

    # Identical matrices for both frameworks.
    W_a, Win_a, bias_a = _arch_components(hp, d_in)

    # ---- Lazy-import architectures -------------------------------------------
    reservoir_dir = os.path.normpath(os.path.join(_THIS_DIR, "../../../../reservoir"))
    if reservoir_dir not in sys.path:
        sys.path.insert(0, reservoir_dir)
    import architectures as _arch  # noqa: PLC0415

    # ---- arch ESN (normalized space, dummy w_out for feature collection) -----
    feat_len = size + d_in + 1  # [r, u, 1]
    arch_params = pd.Series({
        "size": size,
        "leak": float(hp["leak"]),
        "name": "arch_cmp",
        "inputs": "all",
        "fit to": "all",
    })
    esn = _arch.EchoStateNetwork(
        arch_params,
        w_in=np.column_stack([Win_a, bias_a.reshape(-1, 1)]),
        w_res=W_a,
        v_leak=float(hp["leak"]),
        in_means=np.zeros(d_in),
        out_means=np.zeros(d_in),
        w_out=np.zeros((d_in, feat_len)),  # placeholder; updated below
    )
    esn.input_stds = np.ones(d_in)

    # ---- Collect features from an open-loop pass to solve for w_out ----------
    esn.reset_state()
    feat_list = []
    for t, u in enumerate(x_norm):
        esn.next_step(t, u)
        feat_list.append(esn.last_feat.copy())
    X_feat = np.asarray(feat_list)   # (T, size+d_in+1)

    # Discard a short burn-in (same pattern as architectures.py default discard).
    burn = min(1000, T // 10)
    Xb = X_feat[burn:]
    Yb = y_norm[burn:]
    Tn = len(Xb)
    beta = float(ridge_beta if ridge_beta is not None else hp["ridge"])
    ws, _, _ = _arch.solve_ridge_rect(
        Xb / np.sqrt(Tn),
        Yb / np.sqrt(Tn),
        {"reg ridge": [beta], "run error analysis": False},
    )
    w_out = ws[_arch.reg_key(ridge=beta)]   # (d_out, size+d_in+1) = (d_in, feat_len)
    esn.w_out = w_out

    # ---- reservoirpy ESN: same components + same readout --------------------
    # n_train_samples only affects the readout's ridge, which is overwritten
    # below with the arch w_out, so its exact value is immaterial here.
    model = build_esn(hp, d_in, len(x_norm))
    model.reservoir.W = W_a
    model.reservoir.Win = Win_a
    model.reservoir.bias = bias_a
    if not model.initialized:
        model.initialize(x_norm, y_norm)
    model.reset()
    model.fit(x_norm, y_norm)   # init + readout trained; we overwrite readout below

    # Inject the arch w_out into the reservoirpy readout.
    # Ridge._step: out = x @ Wout + bias, so Wout has shape (n_feat, d_out).
    # w_out columns: [:size+d_in] → weight part, [-1] → bias of the bias unit.
    # The readout receives input_to_readout concatenation: [reservoir_state, input].
    # Arch feature: [r, u, 1]. Resy readout feature: [r, u]. Bias is separate.
    model.readout.Wout = w_out[:, : size + d_in].T       # (size+d_in, d_out)
    model.readout.bias = w_out[:, size + d_in].ravel()   # (d_out,) — last column of w_out

    # ---- Readout-wiring sanity check (open-loop over spinup) ----------------
    model.reset()
    R_resy_spin = np.asarray(model.run(spin), dtype=float)   # (spinup, d_out) resy readout

    esn.reset_state()
    arch_spin_preds = []
    for t_s, u_s in enumerate(spin):
        arch_spin_preds.append(esn.next_step(t_s, u_s))
    arch_spin_preds = np.asarray(arch_spin_preds, dtype=float).reshape(spinup, d_in)

    ol_diff = np.linalg.norm(R_resy_spin - arch_spin_preds, "fro")
    ol_denom = np.linalg.norm(arch_spin_preds, "fro") + 1e-300
    ol_rel = ol_diff / ol_denom
    print(f"\n── Readout-wiring sanity check (open-loop over spinup) ─────────────────")
    print(f"  Open-loop readout rel diff : {ol_rel:.3e}  ", end="")
    if ol_rel < 1e-8:
        print("✓ PASS — feature order confirmed")
    else:
        print("✗ FAIL — possible feature-order mismatch; closed-loop result unreliable")

    # ---- Closed-loop rollout ------------------------------------------------
    # reservoirpy: reset, warm up, then generate.
    model.reset()
    model.run(spin)
    preds_resy = _rc_run_generative(model, n_pred, prepend_current=True)

    # arch: reset, warm up spin[:-1], then seed from spin[-1] and auto-regressively predict.
    esn.reset_state()
    for t_w, u_w in enumerate(spin[:-1]):
        esn.next_step(t_w, u_w)   # warm-up; discard predictions
    u = spin[-1].copy()
    preds_arch_list = []
    for _ in range(n_pred):
        u = np.asarray(esn.next_step(0, u), dtype=float).ravel()
        preds_arch_list.append(u.copy())
    preds_arch = np.asarray(preds_arch_list)   # (n_pred, d_out)

    # ---- Compare + report ---------------------------------------------------
    diff = np.abs(preds_resy - preds_arch)
    global_rel_diff = np.linalg.norm(diff, "fro") / (np.linalg.norm(preds_arch, "fro") + 1e-300)
    eps = 1e-300
    step_norms_arch = np.linalg.norm(preds_arch, axis=1)
    step_norms_diff = np.linalg.norm(diff, axis=1)
    step_rel = step_norms_diff / (step_norms_arch + eps)
    max_step_rel = float(np.max(step_rel))
    mean_step_rel = float(np.mean(step_rel))
    passed = global_rel_diff < 1e-8
    verdict = "PASS" if passed else "FAIL"

    print(f"\n── Closed-loop prediction comparison: {system_name} ({verdict}) ──────")
    print(f"  Spinup / pred length        : {spinup} / {n_pred} steps")
    print(f"  Ridge beta used             : {beta:.2e}")
    print(f"  Global relative diff        : {global_rel_diff:.3e}")
    print(f"  Per-step max  rel error     : {max_step_rel:.3e}")
    print(f"  Per-step mean rel error     : {mean_step_rel:.3e}")
    print(f"  Per-step final rel error    : {step_rel[-1]:.3e}")
    if global_rel_diff > 1e-10:
        print(f"  ** NONZERO DISCREPANCY DETECTED — ratio = {global_rel_diff:.3e} **")
    else:
        print(f"  Predictions match to machine-epsilon precision.")
    print(f"  Result: {verdict}  (threshold: global_rel_diff < 1e-8)")

    # ---- Plot ---------------------------------------------------------------
    if save_dir is None:
        save_dir = os.path.join(_THIS_DIR, "figures")
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, f"closedloop_prediction_compare_{system_name}.png")

    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt  # noqa: PLC0415

    n_plot_dims = min(d_in, 3)
    fig, axes = _plt.subplots(n_plot_dims + 1, 1, figsize=(12, 3 * (n_plot_dims + 1)),
                               sharex=True)
    t_axis = np.arange(n_pred)
    for dim_i in range(n_plot_dims):
        ax = axes[dim_i]
        ax.plot(t_axis, preds_arch[:, dim_i], lw=1.2, color="steelblue",
                label="arch (EchoStateNetwork)")
        ax.plot(t_axis, preds_resy[:, dim_i], lw=0.8, color="tomato",
                linestyle="--", label="reservoirpy")
        ax.set_ylabel(f"dim {dim_i} (norm.)")
        ax.legend(fontsize=8, loc="upper right")
    ax_err = axes[-1]
    ax_err.semilogy(t_axis, step_rel, lw=0.8, color="darkgreen")
    ax_err.set_ylabel(r"$\Vert \delta \Vert / \Vert \hat{y}_{\rm arch} \Vert$")
    ax_err.set_xlabel("Prediction step")
    ax_err.set_title(f"Per-step relative error — {verdict}")
    ax_err.grid(True, which="both", alpha=0.3)
    _plt.suptitle(
        f"Closed-loop prediction comparison: {system_name}\n"
        f"global rel diff = {global_rel_diff:.2e}  ({verdict})",
        fontsize=10,
    )
    _plt.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"  Figure saved → {fig_path}")
    if show:
        _plt.show()
    _plt.close(fig)

    return {
        "global_rel_diff": global_rel_diff,
        "max_step_rel": max_step_rel,
        "mean_step_rel": mean_step_rel,
        "preds_resy": preds_resy,
        "preds_arch": preds_arch,
        "fig_path": fig_path,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # run_pass("lorenz_x", arch_components=True, solve="arch")
    # run_benchmark()                                        # generated data
    # run_benchmark(use_new_standards=True)                 # canonical /new_standards data
    # run_ridge_sweep()                                      # all 5 systems
    # run_ridge_sweep(["lorenz_x"])                          # single system smoke-test
    # run_pass("mackey_glass", use_new_standards=True, arch_components=True, solve="arch")
    # run_pass("lorenz96")
    # task_f_rollout("mackey_glass", use_new_standards=True, arch_components=True, solve="arch")
    # task_g_benchmark_compare()
    run_ridge_sweep()
    # task_h_benchmark_rerun()