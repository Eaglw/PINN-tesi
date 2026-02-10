# Dynamic Weighting Implementation: Learning Rate Annealing

This note documents the implementation of the **Learning Rate Annealing** strategy for Physics-Informed Neural Networks (PINNs) in this project, specifically for the 2D Heat Transfer problem.

## Mathematical Foundation

The Learning Rate Annealing strategy, proposed by **Wang et al. (2021)**, aims to mitigate gradient pathologies in PINNs by balancing the contributions of different loss terms (data, boundary, and physics) based on their gradient magnitudes.

The total loss is defined as:
$$\mathcal{L}(	heta) = \lambda_{bc} \mathcal{L}_{bc}(	heta) + \lambda_{data} \mathcal{L}_{data}(	heta) + \lambda_{pde} \mathcal{L}_{pde}(	heta)$$

To ensure balanced training, we want the gradients of each component with respect to the network parameters $	heta$ to be on a similar scale. Fixing $\lambda_{bc}$ as the anchor (reference), we update the other weights $\hat{\lambda}_k$ such that:

$$\hat{\lambda}_k^{(n)} = \frac{\max_{	heta} |
abla_{	heta} \lambda_{bc} \mathcal{L}_{bc}|}{\overline{|
abla_{	heta} \mathcal{L}_k|}}$$

where:
- $
abla_{	heta}$ is the gradient with respect to model parameters.
- $\max$ is the maximum absolute value of the gradient components.
- $\overline{|\cdot|}$ is the mean absolute value of the gradient components.

In this implementation, we use the **maximum of the L2 norms** of the gradients of the parameter tensors as a robust heuristic for the gradient magnitude.

## Implementation Details

The strategy is implemented in `src/Heat2D_PINN.py` within the `train_modelPINN` function.

### 1. Parameters
- `dynamic_weighting`: Boolean flag to enable the strategy.
- `update_weights_every`: Frequency (in epochs) of the weight updates. Default is 100.
- `alpha_dynamic`: Smoothing factor for the moving average of weights (set to 0.9).

### 2. Update Logic
Every `update_weights_every` epochs, after the warmup phase:
1. **Compute Gradients**:
   - The gradients of the unweighted BC loss, Physics loss, and Data loss are computed using `torch.autograd.grad` with `retain_graph=True`.
2. **Calculate Norms**:
   - For each loss component, we find the maximum L2 norm among all model parameter gradients.
3. **Calculate Target Weights**:
   - $	ext{Target} \lambda_{pde} = \frac{\max(	ext{GradNorm}_{bc})}{\max(	ext{GradNorm}_{pde})} 	imes \lambda_{bc}$
   - $	ext{Target} \lambda_{data} = \frac{\max(	ext{GradNorm}_{bc})}{\max(	ext{GradNorm}_{data})} 	imes \lambda_{bc}$
4. **Moving Average Update**:
   - $\lambda_{new} = \alpha \lambda_{old} + (1 - \alpha) 	ext{Target} \lambda$

### 3. Integration in Grid Search
The `Heat2D_weighted_main.py` script executes comparative runs:
- **Static**: Uses fixed weights `BC=1.0, PHYS=10.0, DATA=100.0`.
- **Dynamic**: Starts with unit weights and evolves them using the annealing logic.

## References
- Wang, S., Teng, Y., & Perdikaris, P. (2021). *Understanding and mitigating gradient pathologies in physics-informed neural networks*. SIAM Journal on Scientific Computing, 43(5), A3055-A3081.
