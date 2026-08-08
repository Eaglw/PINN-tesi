# Postprocessing and Evaluation

## Overview
The standalone post-processing protocol (`postprocess_run.py`) provides an autonomous and resilient mechanism for evaluating completed or interrupted Physics-Informed Neural Network (PINN) experiments. It decoupling post-training diagnostic analysis, parameter reporting, field visualization, and metric aggregation from the main training execution loop.

## Technical Implementation & Architecture

### 1. State Restoration & Checkpoint Resolution
The post-processing framework loads PyTorch checkpoint binaries (`.pth`) saved during or at the conclusion of training stages (such as Phase 2 L-BFGS or Phase 2 Adam).
- **State Dict Ingestion**: Restores model parameters (`model_state_dict`), physical model states (`physics_state_dict`), and historical loss/parameter trajectories (`history_state_dict`).
- **Auto-Detection Strategy**: When launched without arguments, it scans `output_4rollmill/`, selects the directory with the latest modification timestamp (`st_mtime`), and selects the optimal available checkpoint file in order of priority:
  1. `checkpoint_lbfgs_phase2.pth`
  2. `checkpoint.pth`
  3. `checkpoint_lbfgs_phase1.pth`
  4. `checkpoint_phase2_adam.pth`
  5. `checkpoint_phase1_adam.pth`
- **Explicit Target Resolution**: Accepts direct paths to `.pth` files or run output folders.

```python
# Checkpoint resolution priority snippet
checkpoint_candidates = [
    "checkpoint_lbfgs_phase2.pth",
    "checkpoint.pth",
    "checkpoint_lbfgs_phase1.pth",
    "checkpoint_phase2_adam.pth",
    "checkpoint_phase1_adam.pth",
]
```

### 2. Resilient Environment & Constant Safe-Guarding
To prevent runtime crashes during plot generation (such as `NameError` due to missing `builtins` or uninitialized global constants), ground truth reference values ($\eta_s, \eta_p, \lambda, \epsilon, \alpha, \beta$) are safely fetched via dynamic module injection and fallbacks:

$$\beta = \frac{\eta_s}{\eta_s + \eta_p}$$

```python
mod_globals = globals()
mu_s_true = getattr(builtins, "MU_S_TRUE", mod_globals.get("MU_S_TRUE", 0.1))
mu_p_true = getattr(builtins, "MU_P_TRUE", mod_globals.get("MU_P_TRUE", 0.9))
tot_visc = mu_s_true + mu_p_true
default_beta = mu_s_true / tot_visc if tot_visc > 0 else 0.1
beta_true = getattr(builtins, "BETA_TRUE", mod_globals.get("BETA_TRUE", default_beta))
```

### 3. Metric Aggregation & Diagnostic Outputs
The evaluation protocol reuses core functions from `src/physics.py` and `src/utils.py`:
- **Physical Parameters Log**: Displays learned values versus ground truth for inverse problems.
- **Detailed Loss Breakdown**: Evaluates PDE residual losses, boundary condition losses, and data losses via `evaluate_final_losses`.
- **L2 Relative Errors**: Computes relative $L_2$ errors for velocity $(u, v)$, pressure $p$, and extra-stress tensor components $(\tau_{xx}, \tau_{xy}, \tau_{yy})$ via `compute_l2_errors`.
- **Plot Generation**:
  - `loss_history.png`: PDE, BC, and total loss evolution.
  - `params_evolution.png`: Convergence of physical parameters over epochs/iterations.
  - `l2_errors_history.png`: Global and masked stress $L_2$ error histories.
  - `generate_all_diagnostics`: 2D spatial contour plots, error heatmaps, streamline patterns, and high-stress region analysis.

All visual outputs are automatically organized and saved in a dedicated subfolder `postprocess_plots/` within the run directory alongside the target checkpoint.

## CLI Usage

```bash
# Auto-detect latest run in output_4rollmill/ and process:
python final_roll/postprocess_run.py

# Process a specific run directory:
python final_roll/postprocess_run.py final_roll/output_4rollmill/4_roll_mill_lambda1_L8x128_E100000_SiLU_stagedTrue_invFalse_20260808_180000

# Process a specific checkpoint file:
python final_roll/postprocess_run.py final_roll/output_4rollmill/4_roll_mill_lambda1_L8x128_E100000_SiLU_stagedTrue_invFalse_20260808_180000/checkpoint_lbfgs_phase2.pth
```

## References & Back-links
- [[Viscoelastic_Metrics]]
- [[Loss_History_Tracking]]
- [[Viscoelastic_Training]]
- [[Staged_Training_Procedure]]
- [[ViscoelasticNet]]
