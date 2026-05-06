# Method: Dynamic Weighting

Dynamic weighting refers to algorithms that automatically adjust the scalar weights (\(\omega\)) in the PINN loss function during training.

## Purpose
The gradients of different loss terms (e.g., boundary vs. physics residue) can have drastically different magnitudes, causing the optimizer to prioritize one at the expense of others. Dynamic weighting balances these contributions.

## Techniques
- **Learning Rate Annealing**: The current implementation balances the loss terms by comparing the gradient norms. For example, the physics loss weight \(\lambda_{phys}\) is updated as:
  \[ \lambda_{phys} = (1 - \alpha) \lambda_{phys} + \alpha \frac{\overline{\|\nabla_{\theta} \mathcal{L}_{bc}\|}}{\overline{\|\nabla_{\theta} \mathcal{L}_{phys}\|}} \lambda_{bc} \]
  where \(\alpha = 0.1\) (exponential moving average) ensures smooth transitions.
- **Adaptive Weighting**: Treat weights as trainable parameters (often requiring specific regularization).
- **Staged Weighting**: Warmup phases where \(\lambda_{phys} = 0\) to allow the model to learn boundary data before enforcing physics.

## Related
- **Literature**: Mentioned in [[Note_01_Framework]].
- **Topics**: [[Loss_Functions]], [[PINN_Fundamentals]]
