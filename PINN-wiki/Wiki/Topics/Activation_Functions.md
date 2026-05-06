# Topic: Activation Functions

The choice of activation function is critical for PINN stability, especially for higher-order derivatives.

## Common Functions
- **Tanh**: Standard for small networks; provides a "natural brake" due to its limited range `[-1, 1]`.
- **GELU**: Highly stable for deep networks; no upper saturation helps gradient flow.
- **SiLU (Swish)**: Often the **best performer** for second-order PDEs (as validated in [[Sharma_et_al_Hyperparameter_Selection]]) due to the regularity of its second derivative.

## Functions to Avoid in Deep PINNs
- **ReLU**: Suffers from vanishing gradients in PINN contexts because its derivative is zero for negative inputs, preventing proper parameter updates.
- **SELU**: Prone to exploding gradients in very deep PINN architectures.

## Learnable Adaptive Activations (LAA)
Introduced to capture sharp gradients or slow variations:
\[ f(x) = \sigma(a \cdot x) \]
Where \( a \) is a learnable scaling parameter.

## References
- Discussed in [[Note_01_Framework]] and [[Note_03_Heat2D]].
