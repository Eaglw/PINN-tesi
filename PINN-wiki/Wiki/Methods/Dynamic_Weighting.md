# Method: Dynamic Weighting

Dynamic weighting refers to algorithms that automatically adjust the scalar weights (\(\omega\)) in the PINN loss function during training.

## Purpose
The gradients of different loss terms (e.g., boundary vs. physics residue) can have drastically different magnitudes, causing the optimizer to prioritize one at the expense of others. Dynamic weighting balances these contributions.

## Techniques
- **Learning Rate Annealing**: Adjusting weights based on the relative statistics of gradients.
- **Adaptive Weighting**: Treat weights as trainable parameters (often requiring specific regularization).

## Related
- **Literature**: Mentioned in [[Note_01_Framework]].
- **Topics**: [[Loss_Functions]], [[PINN_Fundamentals]]
