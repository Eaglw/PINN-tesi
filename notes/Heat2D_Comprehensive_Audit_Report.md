# Heat 2D Comprehensive Audit Report

## 1. Progress Timeline & Historical Context

This section summarizes the evolution of the Heat 2D module from January 19, 2026, to present, highlighting key architectural decisions and experimental milestones.

### Phase 1: Optimization & Refactoring (Jan 19 - Jan 20)
*   **Context:** Initial experiments showed PINN underperformance compared to standard Neural Networks in the forward problem.
*   **Key Actions:**
    *   **Comparison Study:** Rigorous benchmarking of PINN vs. Grid-based NN (`pinn_optim_20260119`).
    *   **Modular Physics:** Introduced `Heat2D/physics.py` to decouple the PDE definition from the training loop.
    *   **Pure Physics Mode:** Implemented a data-free training mode (`GOAL=4`) to test the physics loss in isolation (`heat2d_pure_physics_20260120`).

### Phase 2: Infrastructure Hardening (Jan 21 - Jan 22)
*   **Context:** The codebase was becoming cluttered, and metric tracking was ad-hoc.
*   **Key Actions:**
    *   **Structural Refactor:** Moved core logic to `Heat2D/src/` (Commit `a6c96c9`).
    *   **Centralized Logging:** Created `func/logging_utils.py` and the unified `Heat2D/results.csv` to track experiments across different scripts.
    *   **Visualization Upgrade:** Implemented "Unified Comparison" plots and `analyze_results.py` for post-hoc analysis.
    *   **Point Standardization:** Defined strict point sets (1600 Grid, 1600 Random, 400 Boundary) to ensure fair comparisons between NN and PINN (`heat2d_point_standardization_20260122`).

### Phase 3: Inverse Problems (Jan 25)
*   **Context:** Shifted focus to parameter estimation (finding thermal conductivity $k$).
*   **Key Actions:**
    *   **New Solver:** Created `Heat2D_inverse_main.py` and `src/inverse_physics.py`.
    *   **Validation:** Successfully recovered $k$ from synthetic data with <2% error using joint Adam/L-BFGS optimization.

### Phase 4: Advanced Optimization (Jan 27 - Present)
*   **Context:** Addressing the issue where boundary loss ($\mathcal{L}_{BC}$) disproportionately dominated the optimization landscape.
*   **Key Actions:**
    *   **Loss Weighting:** Implemented static weighting ($\lambda_{BC}=1, \lambda_{Phys}=10, \lambda_{Data}=50$) in `Heat2D_weighted_main.py`.
    *   **Reduced Data Regime:** Created `Heat2D_reduced_main.py` to test PINN performance with limited data points, pushing the "Physics-Informed" advantage.
