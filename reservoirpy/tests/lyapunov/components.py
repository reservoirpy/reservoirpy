"""Component-level A/B isolation of `architectures.py` vs `reservoirpy` ESN.

Four tests swap one component at a time and compare median valid prediction time
(0.2×std threshold) on Lorenz (x only) using a fixed validation seed.  Designed
to isolate which differences between the two codebases account for the gap.

Tests
-----
0. Distribution family   — reservoirpy defaults (bernoulli/normal) vs uniform[−1,1]
1. Exact matrices         — reservoirpy-generated (uniform) vs architectures basic_in/basic_res
2. Readout solve          — reservoirpy Ridge vs architectures classic_ridge (solve_square)
3. Training data origin   — tests/lyapunov generated stream vs new_standards reference CSV

Usage (from IDE, relying on task_b_hp having been run first):
    from reservoirpy.tests.lyapunov.components import run_all
    run_all("lorenz_x")
"""

import os
import sys
from copy import deepcopy

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
# Ensure the local reservoirpy fork takes precedence.
sys.path.insert(
    0,
    os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")
    ),
)
# Make `import architectures` (and its neighbours) resolvable.
_RESERVOIR_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../reservoir")
)
if _RESERVOIR_DIR not in sys.path:
    sys.path.insert(0, _RESERVOIR_DIR)

import numpy as np
import pandas as pd

from reservoirpy import ESN
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.mat_gen import uniform, bernoulli, normal
from reservoirpy.observables import valid_time_multitest

from reservoirpy.tests.lyapunov.lyapunov_test import (
    _SYSTEM_CONFIGS_5,
    _load_best_hp,
    _load_hp_csv,
    generate_train_val,
    sample_val_windows,
    build_esn,
    load_or_train_esn,
    _arch_components,
)
from reservoirpy.tests.lyapunov.esn_test import (
    _NEW_STANDARDS_DIR,
    PASS_CONFIGS,
    _SPINUP, _PRED_LEN, _N_WINDOWS, _KS, _N_TRAIN, _N_VAL,
    _step, _show_hint, _prompt_task,
)

# ---------------------------------------------------------------------------
# architectures import (optional: needed only for Tests 1 & 2)
# ---------------------------------------------------------------------------
try:
    import architectures as arch
    _ARCH_AVAILABLE = True
except ImportError as _e:
    arch = None
    _ARCH_AVAILABLE = False
    print(f"[components] WARNING: architectures import failed: {_e}")
    print("[components] Tests 1 and 2 require architectures — they will be skipped.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VAL_SEED = 0   # fixed for every A/B comparison (same validation windows across all tests)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalize_train(train_data):
    """Compute normalization from clean training inputs.

    Returns (traj_norm, norm, x_norm, y_norm) where norm = {"mean", "std"}.
    """
    x_mean = train_data[:-1].mean(axis=0)
    x_std  = np.maximum(train_data[:-1].std(axis=0), 1e-8)
    norm   = {"mean": x_mean, "std": x_std}
    traj   = (train_data - x_mean) / x_std
    return traj, norm, traj[:-1], traj[1:]


def _build_esn_with_inits(hp, input_dim, n_train_samples, Win, W, bias_init):
    """Like build_esn but with explicit Win/W/bias initializers or precomputed arrays.

    All other conventions (Ridge β·T, input_to_readout=True) are unchanged.
    When Win/W/bias are arrays, Reservoir ignores sr/input_scaling/connectivity.
    When they are callables, Reservoir applies those parameters normally.
    """
    rc_conn    = hp["sparse"] / hp["size"]
    ridge_beta = float(hp.get("ridge", 0.0))
    noise_reg  = float(hp.get("noise_reg", 0.0))
    if ridge_beta == 0.0 and noise_reg == 0.0:
        ridge_beta = 1e-10
    reservoir = Reservoir(
        units=hp["size"],
        sr=hp["radius"],
        lr=hp["leak"],
        input_scaling=hp["in_scale"],
        input_connectivity=1.0 / input_dim,
        rc_connectivity=rc_conn,
        W=W,
        Win=Win,
        bias=bias_init,
        seed=hp["seed"],
    )
    readout = Ridge(ridge=ridge_beta * n_train_samples, fit_bias=True)
    return ESN(reservoir=reservoir, readout=readout, input_to_readout=True)


def _uniform_bias(bias_scale):
    """Uniform[−1,1] bias initializer scaled by bias_scale (partial Initializer)."""
    return uniform(input_scaling=bias_scale)


def _median_vt(model, norm, val_block, t_step):
    """Score *model* on a fixed set of validation windows (seed=_VAL_SEED).

    Returns the full valid_time_multitest result dict.
    """
    windows      = sample_val_windows(
        val_block, n=_N_WINDOWS, spinup=_SPINUP, pred_len=_PRED_LEN, seed=_VAL_SEED,
    )
    windows_norm = [(w - norm["mean"]) / norm["std"] for w in windows]
    return valid_time_multitest(
        model, windows_norm, n_segments=_N_WINDOWS,
        spinup=_SPINUP, eps=[0.2, 0.4], ks=_KS,
        block=_PRED_LEN, t_step=t_step,
    )


def _print_ab(label_a, res_a, label_b, res_b, extra_a=None, extra_b=None):
    """Print an A-vs-B valid-time and RMSE block.  extra_* adds a suffix line."""
    def _fmt(res):
        vt  = res.get("valid_time_0.2_median", float("nan"))
        r4  = res.get("rmse_4_gmean",          float("nan"))
        r64 = res.get("rmse_64_gmean",         float("nan"))
        r256= res.get("rmse_256_gmean",        float("nan"))
        return vt, r4, r64, r256

    vt_a, r4_a, r64_a, r256_a = _fmt(res_a)
    vt_b, r4_b, r64_b, r256_b = _fmt(res_b)

    hdr = f"  {'variant':35s}  {'vt_0.2_med':>11}  {'rmse_4':>9}  {'rmse_64':>9}  {'rmse_256':>9}"
    print(hdr)
    print("  " + "-"*(len(hdr)-2))
    for lbl, vt, r4, r64, r256, extra in [
        (label_a, vt_a, r4_a, r64_a, r256_a, extra_a),
        (label_b, vt_b, r4_b, r64_b, r256_b, extra_b),
    ]:
        line = (f"  {lbl:35s}  {vt:11.4f}  {r4:9.5f}  {r64:9.5f}  {r256:9.5f}")
        if extra:
            line += f"  ({extra})"
        print(line)
    ratio = vt_b / vt_a if vt_a > 0 and np.isfinite(vt_a) else float("nan")
    print(f"\n  VT ratio B/A = {ratio:.3f}")
    return {"vt_A": vt_a, "vt_B": vt_b, "ratio": ratio}


# ---------------------------------------------------------------------------
# Test 0 — initializer distribution family
# ---------------------------------------------------------------------------

def component_0_distributions(system_name="lorenz_x"):
    """Compare reservoirpy library defaults (bernoulli/normal) vs uniform[−1,1].

    build_esn (and Tests 1–3) now use uniform as the standard, matching
    architectures.basic_in / basic_res.  This test isolates whether the
    distribution change alone accounts for part of the gap.

    A = old reservoirpy defaults: Win=bernoulli (discrete ±1), W=normal (Gaussian),
        bias=bernoulli — explicitly passed, NOT the current build_esn.
    B = uniform[−1,1] for Win, W, and bias — the current build_esn standard.

    Everything else (sr, in_scale, bias_scale, sparsity, seed) is identical.

    Returns a summary dict.
    """
    print(f"\n{'='*60}")
    print(f"  component_0_distributions: {system_name}")
    print(f"  Probe: does the distribution family (bernoulli/normal vs uniform) matter?")
    print(f"  A = old reservoirpy defaults (Win=bernoulli, W=normal, bias=bernoulli)")
    print(f"  B = uniform[−1,1] for Win, W, bias  [current build_esn standard]")
    print(f"  Everything else (sr, in_scale, bias_scale, sparsity, seed) is identical.")
    print(f"{'='*60}")

    _step("loading hyperparameters")
    hp = _load_hp_csv(system_name)
    print(f"    seed={hp['seed']}  size={hp['size']}  ridge={hp['ridge']:.2e}")

    _step(f"generating/loading {_N_TRAIN}-step train + {_N_VAL}-step val (cached)")
    train_data, val_block, t_step = generate_train_val(
        system_name, n_train=_N_TRAIN, n_val=_N_VAL,
    )
    _, norm, x_norm, y_norm = _normalize_train(train_data)
    input_dim       = x_norm.shape[1]
    n_train_samples = len(x_norm)

    # --- Variant A: old reservoirpy library defaults (bernoulli/normal) ---
    _step("Variant A: building + training ESN with bernoulli/normal defaults")
    model_a = _build_esn_with_inits(
        hp, input_dim, n_train_samples,
        Win=bernoulli,
        W=normal,
        bias_init=bernoulli(input_scaling=hp["bias_scale"]),
    )
    model_a.fit(x_norm, y_norm)
    _step("Variant A: scoring")
    res_a = _median_vt(model_a, norm, val_block, t_step)

    # --- Variant B: uniform (current build_esn standard) ---
    _step("Variant B: building + training uniform ESN (current build_esn standard)")
    model_b = _build_esn_with_inits(
        hp, input_dim, n_train_samples,
        Win=uniform,
        W=uniform,
        bias_init=_uniform_bias(hp["bias_scale"]),
    )
    model_b.fit(x_norm, y_norm)
    _step("Variant B: scoring")
    res_b = _median_vt(model_b, norm, val_block, t_step)

    print(f"\n  Results (validation seed={_VAL_SEED}, {_N_WINDOWS} windows):")
    summary = _print_ab(
        "A: bernoulli Win / normal W / bernoulli bias", res_a,
        "B: uniform[−1,1] (build_esn standard)",       res_b,
    )
    print(f"\n  Interpretation: ratio B/A >> 1 means the distribution change accounts")
    print(f"  for part of the performance gap vs architectures.")
    return {**summary, "label": "Test 0: distribution family"}


# ---------------------------------------------------------------------------
# Test 1 — exact w_in / w_res from architectures
# ---------------------------------------------------------------------------

def component_1_matrices(system_name="lorenz_x"):
    """Compare reservoirpy-generated uniform matrices vs architectures' exact basic_in/basic_res.

    Both sides use uniform distributions; Variant B substitutes the exact matrices produced
    by architectures.basic_in / architectures.basic_res (same seed).  This isolates the
    effect of sparsity structure, the 1/dim-sparse input pattern, and exact numerical values.

    The injected Win/bias come from raw basic_in output (in_scale/bias_scale applied),
    without the 1/input_stds fold-in that architectures applies internally — justified
    because we always feed normalised (std≈1) data.

    Returns a summary dict.
    """
    if not _ARCH_AVAILABLE:
        print("[component_1_matrices] skipped: architectures not importable.")
        return {"label": "Test 1: exact matrices", "vt_A": float("nan"), "vt_B": float("nan")}

    print(f"\n{'='*60}")
    print(f"  component_1_matrices: {system_name}")
    print(f"  Probe: does exact matrix structure (sparsity, values) matter beyond distribution?")
    print(f"  A = reservoirpy-generated uniform matrices (same sr/in_scale/seed/sparsity)")
    print(f"  B = architectures basic_in / basic_res injected into reservoirpy Reservoir")
    print(f"{'='*60}")

    _step("loading hyperparameters")
    hp = _load_hp_csv(system_name)

    _step(f"generating/loading data (cached)")
    train_data, val_block, t_step = generate_train_val(
        system_name, n_train=_N_TRAIN, n_val=_N_VAL,
    )
    _, norm, x_norm, y_norm = _normalize_train(train_data)
    input_dim       = x_norm.shape[1]
    n_train_samples = len(x_norm)

    # --- Variant A: reservoirpy-generated uniform ---
    _step("Variant A: building + training all-uniform ESN (reservoirpy-generated)")
    model_a = _build_esn_with_inits(
        hp, input_dim, n_train_samples,
        Win=uniform, W=uniform,
        bias_init=_uniform_bias(hp["bias_scale"]),
    )
    model_a.fit(x_norm, y_norm)
    _step("Variant A: scoring")
    res_a = _median_vt(model_a, norm, val_block, t_step)

    # --- Variant B: architectures matrices (via shared lyapunov_test._arch_components) ---
    _step("Variant B: generating architectures matrices (basic_res / basic_in)")
    W_arch, Win_b, bias_b = _arch_components(hp, input_dim)

    # Sanity-check spectral radius and sparsity
    eigs = np.linalg.eigvals(W_arch)
    actual_sr      = float(np.max(np.abs(eigs)))
    actual_density = float(np.count_nonzero(W_arch)) / W_arch.size
    target_density = hp["sparse"] / hp["size"]
    print(f"    W_arch: spectral radius = {actual_sr:.4f} (target {hp['radius']:.4f})")
    print(f"    W_arch: density         = {actual_density:.4f} (target ≈{target_density:.4f})")

    _step("Variant B: building + training ESN with injected matrices")
    model_b = _build_esn_with_inits(
        hp, input_dim, n_train_samples,
        Win=Win_b, W=W_arch, bias_init=bias_b,
    )
    model_b.fit(x_norm, y_norm)
    _step("Variant B: scoring")
    res_b = _median_vt(model_b, norm, val_block, t_step)

    print(f"\n  Results (validation seed={_VAL_SEED}, {_N_WINDOWS} windows):")
    summary = _print_ab(
        "A: reservoirpy uniform (generated)", res_a,
        "B: architectures basic_in/basic_res", res_b,
    )
    print(f"\n  Interpretation: if ratio B/A >> 1 (beyond what Test 0 showed), the")
    print(f"  exact sparsity structure / numerical values of the matrices matter.")
    return {**summary, "label": "Test 1: exact matrices"}


# ---------------------------------------------------------------------------
# Test 2 — readout solve (classic_ridge vs reservoirpy Ridge)
# ---------------------------------------------------------------------------

def component_2_readout(system_name="lorenz_x"):
    """Compare reservoirpy Ridge vs architectures classic_ridge (solve_square) on the same reservoir.

    Both use uniform reservoir matrices so the reservoir is held constant.
    Convention: architectures classic_ridge divided X,Y by sqrt(T) before solve_square
    (averaged-Gram), matched by ridge_reservoirpy = beta × T (the build_esn convention).

    Also runs a diagnostic variant with ridge = beta (NOT ×T) to confirm build_esn's
    convention is correct.

    Returns a summary dict.
    """
    if not _ARCH_AVAILABLE:
        print("[component_2_readout] skipped: architectures not importable.")
        return {"label": "Test 2: readout solve", "vt_A": float("nan"), "vt_B": float("nan")}

    print(f"\n{'='*60}")
    print(f"  component_2_readout: {system_name}")
    print(f"  Probe: does the readout solve method matter?")
    print(f"  A = reservoirpy Ridge(ridge=β·T)  [build_esn convention]")
    print(f"  B = architectures solve_square (classic_ridge) with √T-scaled inputs")
    print(f"  Diag = reservoirpy Ridge(ridge=β)  [no ×T — should be WORSE if β·T is right]")
    print(f"{'='*60}")

    _step("loading hyperparameters")
    hp = _load_hp_csv(system_name)
    beta            = float(hp["ridge"])

    _step("generating/loading data (cached)")
    train_data, val_block, t_step = generate_train_val(
        system_name, n_train=_N_TRAIN, n_val=_N_VAL,
    )
    _, norm, x_norm, y_norm = _normalize_train(train_data)
    input_dim       = x_norm.shape[1]
    n_train_samples = len(x_norm)

    # -----------------------------------------------------------------------
    # Step 1: train ESN A (reservoirpy Ridge, ridge = β·T)
    # -----------------------------------------------------------------------
    _step("Variant A: building + training uniform ESN (Ridge ridge=β·T)")
    model_a = _build_esn_with_inits(
        hp, input_dim, n_train_samples,
        Win=uniform, W=uniform,
        bias_init=_uniform_bias(hp["bias_scale"]),
    )
    model_a.fit(x_norm, y_norm)
    Wout_A = model_a.readout.Wout.copy()   # (units+d_in, d_in)
    bias_A = model_a.readout.bias.copy()   # (d_in,)

    # -----------------------------------------------------------------------
    # Step 2: collect reservoir states by re-running a fresh identical reservoir
    # -----------------------------------------------------------------------
    _step("collecting reservoir states (fresh uniform Reservoir, same seed/params)")
    rc_conn = hp["sparse"] / hp["size"]
    res_node = Reservoir(
        units=hp["size"], sr=hp["radius"], lr=hp["leak"],
        input_scaling=hp["in_scale"],
        input_connectivity=1.0 / input_dim,
        rc_connectivity=rc_conn,
        W=uniform, Win=uniform,
        bias=_uniform_bias(hp["bias_scale"]),
        seed=hp["seed"],
    )
    R = res_node.run(x_norm)   # (n_train_samples, units)

    # -----------------------------------------------------------------------
    # Step 3: solve_square (architectures classic_ridge)
    # -----------------------------------------------------------------------
    _step("Variant B: assembling feature matrix [R, x_norm, 1] and running solve_square")
    T = n_train_samples
    X_full  = np.hstack([R, x_norm, np.ones((T, 1))])   # (T, units+d_in+1)
    # Replicate the historical caller's √T scaling that classic_ridge used:
    X_sc    = X_full / np.sqrt(T)
    y_sc    = y_norm / np.sqrt(T)
    ws, _   = arch.solve_square(
        X_sc, y_sc, regression="ridge",
        p={"reg ridge": [beta], "square method": "svd"},
    )
    key_b   = arch.reg_key(ridge=beta)
    W_B     = ws[key_b]                       # (d_out, units+d_in+1) = (d_in, units+d_in+1)
    Wout_B  = W_B[:, :-1].T                   # (units+d_in, d_in) — matches reservoirpy layout
    bias_B  = W_B[:, -1]                      # (d_in,)

    # -----------------------------------------------------------------------
    # Step 4: Frobenius comparison of the two solutions
    # -----------------------------------------------------------------------
    dW   = np.linalg.norm(Wout_B - Wout_A, "fro")
    nW   = np.linalg.norm(Wout_B,          "fro")
    db   = np.linalg.norm(bias_B  - bias_A)
    nb   = np.linalg.norm(bias_B)
    print(f"\n  Frobenius comparison (both should be small if conventions match):")
    print(f"    ‖Wout_B − Wout_A‖ / ‖Wout_B‖ = {dW/nW:.4e}")
    print(f"    ‖bias_B − bias_A‖ / ‖bias_B‖  = {db/nb:.4e}")

    # -----------------------------------------------------------------------
    # Step 5: performance of ESN B (inject solve_square Wout into copy of ESN A)
    # -----------------------------------------------------------------------
    _step("Variant B: injecting solve_square Wout into ESN copy and scoring")
    model_b = deepcopy(model_a)
    model_b.readout.Wout = Wout_B.copy()
    model_b.readout.bias = bias_B.copy()
    res_b = _median_vt(model_b, norm, val_block, t_step)

    # -----------------------------------------------------------------------
    # Step 6: diagnostic — same setup but Ridge with ridge=β (no ×T)
    # -----------------------------------------------------------------------
    _step("Diagnostic: building Ridge(ridge=β, no ×T) to confirm β·T convention")
    model_diag = _build_esn_with_inits(
        hp, input_dim, n_train_samples,
        Win=uniform, W=uniform,
        bias_init=_uniform_bias(hp["bias_scale"]),
    )
    # Override the ridge value: replace readout with plain β
    model_diag.readout = Ridge(ridge=beta, fit_bias=True)
    # Re-wire: easiest — just replace the readout then refit the whole ESN
    model_diag2 = _build_esn_with_inits(
        hp, input_dim, n_train_samples,
        Win=uniform, W=uniform,
        bias_init=_uniform_bias(hp["bias_scale"]),
    )
    model_diag2.readout.ridge = beta
    model_diag2.fit(x_norm, y_norm)
    _step("Diagnostic: scoring Ridge(ridge=β, no ×T)")
    res_diag = _median_vt(model_diag2, norm, val_block, t_step)

    _step("Variant A: scoring")
    res_a_scored = _median_vt(model_a, norm, val_block, t_step)

    print(f"\n  Results (validation seed={_VAL_SEED}, {_N_WINDOWS} windows):")
    summary = _print_ab(
        f"A: Ridge(β·T = {beta*T:.2e})", res_a_scored,
        "B: solve_square (classic_ridge)", res_b,
    )
    vt_diag = res_diag.get("valid_time_0.2_median", float("nan"))
    print(f"  Diag: Ridge(β = {beta:.2e}) [no ×T]  vt = {vt_diag:.4f}  "
          f"{'<< A confirms β·T is right' if vt_diag < summary['vt_A'] else '>> surprising'}")
    print(f"\n  Interpretation: if ‖ΔW‖/‖W‖ is tiny and ratio B/A ≈ 1, the readout solve")
    print(f"  is not the source of the gap.  A residual gap points at bias handling")
    print(f"  (mean-centering vs explicit constant column).")
    return {**summary, "label": "Test 2: readout solve", "vt_diag": vt_diag}


# ---------------------------------------------------------------------------
# Test 3 — training data source
# ---------------------------------------------------------------------------

def component_3_data_source(system_name="lorenz_x"):
    """Compare tests/lyapunov generated stream vs new_standards reference CSV.

    Both variants use the same uniform ESN configuration; only the training data
    (and validation data) differ.  Variant B trains and validates on the
    new_standards/lorenz_det.csv trajectory subsampled by subsample_t=3.

    Returns a summary dict.
    """
    print(f"\n{'='*60}")
    print(f"  component_3_data_source: {system_name}")
    print(f"  Probe: does the training data origin matter?")
    print(f"  A = lyapunov test stream (generate_train_val, custom IC)")
    print(f"  B = new_standards/lorenz_det.csv  (subsampled ×3)")
    print(f"  Both: same uniform ESN config, scored on their respective val blocks.")
    print(f"{'='*60}")

    cfg = PASS_CONFIGS[system_name]   # {"ref_csv": ..., "ref_col": ...}
    sub_t = _SYSTEM_CONFIGS_5[system_name]["subsample_t"]

    _step("loading hyperparameters")
    hp = _load_hp_csv(system_name)

    # --- Variant A: lyapunov stream ---
    _step(f"Variant A: generating/loading {_N_TRAIN}-step train + {_N_VAL}-step val (cached)")
    train_a, val_a, t_step = generate_train_val(
        system_name, n_train=_N_TRAIN, n_val=_N_VAL,
    )
    _, norm_a, x_a, y_a = _normalize_train(train_a)
    input_dim       = x_a.shape[1]
    n_train_samples = len(x_a)
    _step("Variant A: building + training uniform ESN")
    model_a = _build_esn_with_inits(
        hp, input_dim, n_train_samples,
        Win=uniform, W=uniform,
        bias_init=_uniform_bias(hp["bias_scale"]),
    )
    model_a.fit(x_a, y_a)
    _step("Variant A: scoring on its own val block")
    res_a = _median_vt(model_a, norm_a, val_a, t_step)
    gen_stats = (train_a[:, 0].mean(), train_a[:, 0].std())

    # --- Variant B: new_standards CSV ---
    ref_path = os.path.join(_NEW_STANDARDS_DIR, cfg["ref_csv"])
    if not os.path.exists(ref_path):
        print(f"  [WARN] reference CSV not found: {ref_path}")
        print(f"  Variant B skipped.")
        return {
            "label": "Test 3: data source",
            "vt_A": res_a.get("valid_time_0.2_median", float("nan")),
            "vt_B": float("nan"),
        }

    _step(f"Variant B: loading {cfg['ref_csv']} col '{cfg['ref_col']}' (subsampled ×{sub_t})")
    ref_df   = pd.read_csv(ref_path, index_col=0)
    ref_col  = ref_df[cfg["ref_col"]].values[::sub_t]   # subsample
    need     = _N_TRAIN + _N_VAL + 1   # +1 for the target row
    if len(ref_col) < need:
        actual_val = len(ref_col) - _N_TRAIN - 1
        print(f"  [NOTE] only {len(ref_col)} rows available; reducing n_val "
              f"from {_N_VAL} to {actual_val}.")
        n_val_b = max(actual_val, _SPINUP + _PRED_LEN + 1)
    else:
        n_val_b = _N_VAL
    train_b  = ref_col[:_N_TRAIN + 1].reshape(-1, 1)            # shape (N_TRAIN+1, 1)
    val_b    = ref_col[_N_TRAIN + 1: _N_TRAIN + 1 + n_val_b].reshape(-1, 1)
    ref_stats = (train_b[:, 0].mean(), train_b[:, 0].std())

    print(f"  Data stats (first variable):")
    print(f"    A gen   mean={gen_stats[0]:.4f}  std={gen_stats[1]:.4f}")
    print(f"    B ref   mean={ref_stats[0]:.4f}  std={ref_stats[1]:.4f}")

    _, norm_b, x_b, y_b = _normalize_train(train_b)
    n_train_b = len(x_b)
    _step("Variant B: building + training uniform ESN on new_standards data")
    model_b = _build_esn_with_inits(
        hp, input_dim, n_train_b,
        Win=uniform, W=uniform,
        bias_init=_uniform_bias(hp["bias_scale"]),
    )
    model_b.fit(x_b, y_b)
    _step("Variant B: scoring on its own val block")
    res_b = _median_vt(model_b, norm_b, val_b, t_step)

    print(f"\n  Results (validation seed={_VAL_SEED}, {_N_WINDOWS} windows):")
    summary = _print_ab(
        "A: lyapunov stream", res_a,
        "B: new_standards ref", res_b,
    )
    print(f"\n  Interpretation: a ratio B/A far from 1 means the data source (IC, transient,")
    print(f"  or trajectory length) explains part of the gap.")
    return {**summary, "label": "Test 3: data source"}


# ---------------------------------------------------------------------------
# run_all — run all four tests with interactive prompts
# ---------------------------------------------------------------------------

def run_all(system_name="lorenz_x"):
    """Run all four component tests for *system_name*.

    Before each test an interactive prompt asks whether to run, skip, or quit
    (default: run).  After all tests, prints a consolidated summary table.

    Requires task_b_hp(system_name) to have been run first so that
    ``best_hp.csv`` contains a row for *system_name*.
    """
    if system_name not in PASS_CONFIGS:
        raise ValueError(
            f"Unknown system {system_name!r}. Choose from {list(PASS_CONFIGS.keys())}."
        )

    print(f"\n{'#'*60}")
    print(f"#  COMPONENTS RUN: {system_name}")
    print(f"#  Four A/B tests; each swaps one component.  Validation seed={_VAL_SEED}.")
    print(f"{'#'*60}")

    tests = [
        ("[1/4] Test 0 — distribution family (bernoulli/normal vs uniform)",
         lambda: component_0_distributions(system_name)),
        ("[2/4] Test 1 — exact architectures matrices (basic_in / basic_res)",
         lambda: component_1_matrices(system_name)),
        ("[3/4] Test 2 — readout solve (Ridge vs solve_square / classic_ridge)",
         lambda: component_2_readout(system_name)),
        ("[4/4] Test 3 — training data source (generated stream vs new_standards)",
         lambda: component_3_data_source(system_name)),
    ]

    summaries = []
    for label, fn in tests:
        action = _prompt_task(label)
        if action == "quit":
            print("  -> run aborted.")
            break
        if action == "skip":
            print("  -> skipped.")
            continue
        result = fn()
        if result is not None:
            summaries.append(result)

    # Final summary table
    if summaries:
        print(f"\n{'#'*60}")
        print(f"#  SUMMARY: {system_name}")
        print(f"{'#'*60}")
        print(f"  {'test':40s}  {'vt_A':>8}  {'vt_B':>8}  {'B/A':>6}")
        print("  " + "-"*68)
        for s in summaries:
            vt_a  = s.get("vt_A", float("nan"))
            vt_b  = s.get("vt_B", float("nan"))
            ratio = s.get("ratio", float("nan"))
            print(f"  {s['label']:40s}  {vt_a:8.4f}  {vt_b:8.4f}  {ratio:6.3f}")
        print(f"\n  B/A > 1 means the B variant (architectures-like) outperforms A (default).")

    print(f"\n{'#'*60}")
    print(f"#  DONE")
    print(f"{'#'*60}")

run_all()