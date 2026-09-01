# Method: Dynamic Weighting

Dynamic weighting refers to algorithms that automatically adjust the scalar weights ($\omega$) in the PINN loss function during training.

## Purpose
The gradients of different loss terms (e.g., boundary vs. physics residue) can have drastically different magnitudes, causing the optimizer to prioritize one at the expense of others. Dynamic weighting balances these contributions.

## Techniques
- **Learning Rate Annealing (Wang et al.)**: Balances loss terms by monitoring gradient statistics during backpropagation ([[Wang_et_al_Gradient_Pathologies]]). For example, the physics loss weight $\lambda_{phys}$ is updated as:
  $$ \lambda_{phys} = (1 - \alpha) \lambda_{phys} + \alpha \frac{\max_{\boldsymbol{\theta}} \{ |\nabla_{\boldsymbol{\theta}} \mathcal{L}_{phys}| \}}{\overline{|\nabla_{\boldsymbol{\theta}} \mathcal{L}_{bc}|}} $$
  where $\alpha = 0.1$ (exponential moving average) ensures smooth transitions.
- **Relative Loss Balancing - ReLoBRaLo (Bischof & Kraus)**: A gradient-free adaptive loss balancing scheme using relative loss progress and random lookbacks ([[Bischof_Kraus_Multi_Objective_Loss_Balancing]]), reducing backward pass overhead by ~40-70%.
- **Adaptive Weighting**: Treat weights as trainable parameters (often requiring specific regularization).
- **Staged Weighting**: Warmup phases where $\lambda_{phys} = 0$ to allow the model to learn boundary data before enforcing physics.

## Related
- **Literature**: [[Wang_et_al_Gradient_Pathologies]], [[Bischof_Kraus_Multi_Objective_Loss_Balancing]], [[Note_01_Framework]]
- **Topics**: [[Loss_Functions]], [[PINN_Fundamentals]], [[Spectral_Bias]]
- **Methods**: [[GPU_Optimization]], [[VRAM_Optimization]], [[Viscoelastic_Residual_Scaling]]
