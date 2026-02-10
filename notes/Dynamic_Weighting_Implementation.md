# Dynamic Weighting Implementation: Learning Rate Annealing

This note documents the implementation of the **Learning Rate Annealing** strategy for Physics-Informed Neural Networks (PINNs) in this project, specifically for the 2D Heat Transfer problem.

## Mathematical Foundation

The Learning Rate Annealing strategy, proposed by **Wang et al. (2021)**, aims to mitigate *gradient pathologies* in PINNs. These pathologies occur when the gradients of different loss components (Boundary Conditions, Initial Conditions, and PDE Residuals) have vastly different magnitudes, causing the optimizer to prioritize the term with the steepest gradient (often the boundary condition) while ignoring the underlying physics.

The total loss is defined as:

$$
\mathcal{L}(\theta) = \lambda_{bc} \mathcal{L}_{bc}(\theta) + \lambda_{data} \mathcal{L}_{data}(\theta) + \lambda_{pde} \mathcal{L}_{pde}(\theta)
$$

To ensure balanced training, we enforce that the gradients of each weighted loss component with respect to the network parameters $\theta$ have similar statistical magnitudes. Fixing $\lambda_{bc}$ as the **anchor** (reference term, usually because BCs are the "hard" constraints), we dynamically update the other weights $\hat{\lambda}_k$ at each training step $n$ (or epoch interval) such that:

$$
\hat{\lambda}_k^{(n)} = \frac{\max_{\theta} \left| \nabla_{\theta} (\lambda_{bc} \mathcal{L}_{bc}) \right|}{\overline{\left| \nabla_{\theta} \mathcal{L}_k \right|}}
$$

Where:
- $\nabla_{\theta}$ denotes the gradient with respect to the model parameters (weights and biases).
- $\max_{\theta} |\cdot|$ is the maximum absolute value among the gradient components (or the maximum norm across layers).
- $\overline{|\cdot|}$ represents the mean absolute value of the gradient components for the specific loss term $k$ (e.g., PDE or Data).

In this specific implementation, we use the **maximum of the L2 norms** of the gradients per layer as a robust heuristic for the gradient magnitude.

## Implementation Details

The strategy is implemented in `src/Heat2D_PINN.py` within the `train_modelPINN` function.

### 1. Hyperparameters
- `dynamic_weighting`: Boolean flag to enable the strategy.
- `update_weights_every`: Frequency (in epochs) of the weight updates. Default is **100**. Frequent updates can lead to instability; sporadic updates fail to adapt to the changing loss landscape.
- `alpha_dynamic` ($\alpha$): Smoothing factor for the moving average (set to **0.9**).

### 2. Update Logic & Algorithm
Every `update_weights_every` epochs, the following procedure is triggered:

1.  **Compute Individual Gradients**:
    We compute gradients for each loss term independently.
    *Note: This requires `retain_graph=True` in PyTorch, which increases VRAM usage significantly.*
    $$
    G_{bc} = \nabla_\theta \mathcal{L}_{bc}, \quad G_{pde} = \nabla_\theta \mathcal{L}_{pde}, \quad G_{data} = \nabla_\theta \mathcal{L}_{data}
    $$
2.  **Calculate Norm Metrics**:
    For each loss component, we compute the statistical metric (Max or Mean of L2 norms) across the network's parameter vector.
3.  **Calculate Target Weights**:
    We calculate the instantaneous target weight required to balance the gradients against the anchor ($\lambda_{bc}$):
    $$
    \hat{\lambda}_{pde} = \frac{\max(G_{bc})}{\overline{G_{pde}}} \times \lambda_{bc}
    $$
    $$
    \hat{\lambda}_{data} = \frac{\max(G_{bc})}{\overline{G_{data}}} \times \lambda_{bc}
    $$
4.  **Moving Average Update (Smoothing)**:
    To prevent oscillation and abrupt jumps in the loss landscape, we apply an Exponential Moving Average (EMA):
    $$
    \lambda_{new} = \alpha \cdot \lambda_{old} + (1 - \alpha) \cdot \hat{\lambda}_{target}
    $$

### 3. Integration & Grid Search
The `Heat2D_weighted_main.py` script facilitates comparative analysis:
- **Static Regime**: Fixed weights (e.g., `BC=1.0, PHYS=10.0, DATA=100.0`). Weights are chosen based on order-of-magnitude estimates of the residuals.
- **Dynamic Regime**: Starts with unit weights (`1.0`) and evolves via the annealing logic. This allows the network to "focus" on the physics term once the boundary conditions are sufficiently learned.

## Technical Considerations
* **Computational Overhead**: Calculating gradients for each loss term individually essentially triples the backward pass cost.
* **Stiffness Ratio**: The ratio $\frac{\max(G_{bc})}{\overline{G_{pde}}}$ is essentially a measure of the stiffness of the optimization problem. If this ratio explodes, the PDE is ill-conditioned relative to the boundary.

## References
- Wang, S., Teng, Y., & Perdikaris, P. (2021). *Understanding and mitigating gradient pathologies in physics-informed neural networks*. SIAM Journal on Scientific Computing, 43(5), A3055-A3081.