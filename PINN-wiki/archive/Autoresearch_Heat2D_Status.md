# Autoresearch Status: Heat2D L2 Error Improvement

**Date:** 2026-04-01
**Current Goal:** Improve L2 Relative Error in `Heat2D` (baseline ~0.06 - 0.14).

## 🛠️ Tools & Infrastructure Created
- **`heat2dmini/Heat2D_weighted_mini.py`**: A parameterized version of the main script optimized for fast iterations (~1-2 mins per run on CPU). Supports command-line arguments for architecture, activation, and run type.
- **`heat2dmini/autoresearch_sweep.py`**: Automation script to sweep through architectures (`[100x4, 80x6, 150x3]`), activations (`Tanh, SiLU, GELU`), and weighting frequencies.
- **`heat2dmini/mini_results.csv`**: Dedicated log for fast iterations to track L2 error vs. Duration.

## 🐛 Critical Fixes Applied
- **`func/history_tracker.py`**: Fixed a `NaN` loss issue when `PINN_PurePhys` was used (caused by `MSELoss` on empty data/boundary tensors).
- **`heat2dmini/src/Heat2D_PINN.py`**: 
    - Refactored `train_modelPINN` to accept `max_total_lbfgs` as an argument.
    - Fixed a bug where L-BFGS would always run for 5000 iterations regardless of configuration.
- **Environment**: Fixed a broken `torch` installation on macOS ARM. Note: `MPS` (Metal) is currently disabled because it doesn't support `float64`.

## 📊 Initial Findings (Baseline)
| Run Type | Arch | Act | Epochs | L-BFGS | L2 Error | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| PurePhys | 100x4 | Tanh | 2000 | 500 | **0.1277** | 130s |
| PurePhys | 100x4 | Tanh | 1000 | 200 | **0.1455** | 45s |

## 🚀 Next Steps (to continue)
1.  **Resume Sweep**: Run `python heat2dmini/autoresearch_sweep.py` to finish the initial parameter exploration.
2.  **Analyze `mini_results.csv`**: Identify the best architecture/activation combination.
3.  **Scale Up**: Take the best "mini" candidate and run it for 40k+ epochs with full L-BFGS (5000+ iters) to verify final performance.
4.  **Weighting Tuning**: Experiment with `update_weights_every` values (currently testing 100 vs 500).
