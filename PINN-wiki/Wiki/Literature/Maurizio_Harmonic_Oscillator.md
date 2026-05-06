# Literature: Damped Harmonic Oscillator PINN (Maurizio)

## Summary
Implementation of a Physics-Informed Neural Network to solve the 1D damped harmonic oscillator problem, exploring both direct (solving the ODE) and inverse (identifying physical parameters) approaches.

## Key Methodology
- **Equation**: $ m \frac{d^2x}{dt^2} + \mu \frac{dx}{dt} + kx = 0 $ with $ x(0)=1, \dot{x}(0)=0 $.
- **Neural Network**: Fully Connected Network (FCN) with SiLU or GELU activations.
- **Inverse Problem**: 
    - Parameters $\mu$ (damping) and $k$ (stiffness) are treated as learnable `nn.Parameter`.
    - Soft constraints applied to prevent unphysical overdamped states using regularization: $ \text{relu}(\mu - 2\sqrt{k})^2 $.
- **Optimization Strategy**:
    - **Adam** for pre-training and exploration.
    - **L-BFGS** for fine-tuning precision.
    - **Lambda Scheduling**: Gradually increasing the weight of the PDE residual loss ($\lambda_{pde}$) during training.
    - **Xavier Initialization** for better gradient flow.

## Key Findings
- PINNs can accurately interpolate and extrapolate physical behavior when a standard NN fails to generalize.
- Inverse identification of $\mu$ and $k$ is highly sensitive to the relative weighting of data and physics losses.
- GELU activation often outperforms Tanh for this second-order ODE due to smoother gradients.

## Related
- **Topics**: [[PINN_Fundamentals]], [[Loss_Functions]], [[Activation_Functions]]
- **Methods**: [[Dynamic_Weighting]], [[Staged_Precision_Strategy]]
- **Systems**: [[Harmonic_Oscillator]]
