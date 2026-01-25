# Specification - Heat2D Inverse Problem Implementation

## Overview
This track implements an inverse problem solver for the 2D Heat Transfer equation. The goal is to recover the thermal conductivity $k$ from sparse temperature observations in a steady-state system with a constant heat source.

## Functional Requirements

### 1. Inverse Solver Script (`Heat2D_inverse_main.py`)
- **Physics:** Implement the Poisson residual $k(T_{xx} + T_{yy}) + Q = 0$ where $Q=1.0$.
- **Learnable Parameter:** Define $k$ as an `nn.Parameter` with an initial guess (e.g., $k=0.5$).
- **Joint Optimization:** Use Adam (and optionally L-BFGS) to minimize the combined loss:
    - $Loss_{Total} = \lambda_{data} Loss_{data} + \lambda_{bc} Loss_{bc} + \lambda_{phys} Loss_{phys}$

### 2. Geometry & BCs (Matching Direct Case)
- Domain: Unit square $[0, 1] \times [0, 1]$.
- BCs: $T=1$ on $x=1$, and $T=0$ on all other three boundaries.

### 3. Data Generation
- Implement a method to generate "observed" data. Since an analytical solution for this specific Poisson case is complex, the script will generate synthetic data by sampling from a verified high-precision solution for a fixed "True $k$".

### 4. Hyperparameter Grid Search
- Support grid search over:
    - Network depth and width.
    - Observation noise levels (e.g., 0%, 2%, 5%).
    - Number of observation points.

### 5. Logging and Visualization
- **CSV Output:** `Heat2D/results_inverse.csv` with columns rearranged for clarity:
    - Experiment metadata (ID, Arch, Params).
    - Training metrics (Final losses, Time).
    - Physics metrics (`True_K`, `Estimated_K`, `Rel_Error_K`).
- **Plots:**
    - Standard comparison grid (Exact vs. Prediction).
    - Convergence plot for the parameter $k$.

## Acceptance Criteria
- [ ] The script can recover $k$ with <2% error for noise-free synthetic data.
- [ ] The folder `Heat2D/experiments_inverse/` is correctly populated with plots and logs.
- [ ] `results_inverse.csv` maintains a logical ordering of columns as specified.
