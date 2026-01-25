# Specification - Heat2D Inverse Problem Implementation

## Overview
This track implements an inverse problem solver for the 2D Heat Transfer equation using Physics-Informed Neural Networks (PINNs). The goal is to estimate an unknown physical parameter—specifically the thermal diffusivity $\alpha$—from sparse and potentially noisy temperature observations.

## Functional Requirements

### 1. Inverse Solver Script (`Heat2D_inverse_main.py`)
- Implement a PINN architecture where thermal diffusivity $\alpha$ is a `nn.Parameter` to be optimized.
- Support joint optimization of network weights and physical parameters.
- Incorporate a two-stage training process (optional pre-training on data followed by joint physics-data training).

### 2. Data Generation & Sampling
- Generate synthetic "observed" data using the analytical solution from `physics.py` with a "ground truth" $\alpha$.
- Support configurable data density (number of observation points).
- Support adding Gaussian noise to synthetic observations to test robustness.

### 3. Hyperparameter Grid Search
- Implement a grid search mechanism for the inverse problem, exploring:
    - Network architecture (layers, neurons).
    - Data density (number of training points).
    - Noise levels.
    - Learning rates (different for weights vs. parameters).

### 4. Logging and Results
- Create `Heat2D/experiments_inverse/` for saving artifacts.
- Log metrics to `Heat2D/results_inverse.csv`.
- Metrics must include: standard errors (MAE, L2) and the Relative Error of the estimated parameter $\alpha$.

### 5. Visualization
- Generate standard comparison grids (Exact vs. Pred, Error, Residual).
- Generate a convergence plot for the estimated parameter $\alpha$ over the training epochs.

## Non-Functional Requirements
- **Precision:** Use `torch.float64` for numerical stability.
- **Consistency:** Maintain code style and modularity consistent with `Heat2D_main.py`.

## Acceptance Criteria
- [ ] The script successfully estimates $\alpha$ within a small tolerance (e.g., <5% error) for noise-free data.
- [ ] Results and plots are correctly saved in the specified directory and CSV.
- [ ] The grid search executes and logs results for all configurations.
