# Method: ViscoelasticNet Unified Model

## Overview
The **ViscoelasticNet Unified Model** implements a general constitutive equation capable of representing three distinct viscoelastic fluid models: **Oldroyd-B**, **Giesekus**, and **Linear PTT (Phan-Thien–Tanner)**. By learning the values of the model parameters via Physics-Informed Neural Networks (PINNs), the framework can autonomously identify the rheological constitutive relation that best describes the fluid flow.

The stress tensor $\boldsymbol{\tau}'$ is split into solvent and polymeric parts:
$$
\boldsymbol{\tau}' = \boldsymbol{\tau}^s + \boldsymbol{\tau}
$$

where the solvent stress $\boldsymbol{\tau}^s$ is Newtonian:
$$
\boldsymbol{\tau}^s = \eta_s (\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T)
$$

The polymeric stress $\boldsymbol{\tau}$ is governed by the unified constitutive equation:
$$
\left(1 + \frac{\epsilon \lambda}{\eta_p} tr(\boldsymbol{\tau})\right) \boldsymbol{\tau} + \lambda \overset{\nabla}{\boldsymbol{\tau}} + \alpha \frac{\lambda}{\eta_p} (\boldsymbol{\tau} \cdot \boldsymbol{\tau}) = \eta_p (\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T)
$$

where:
* $\eta_s$ is the solvent viscosity.
* $\eta_p$ is the polymeric viscosity.
* $\lambda$ is the relaxation time.
* $\epsilon$ is the PTT extensibility parameter.
* $\alpha$ is the Giesekus mobility parameter.
* $tr(\boldsymbol{\tau}) = \tau^{xx} + \tau^{yy}$ is the trace of the stress tensor.
* $\overset{\nabla}{\boldsymbol{\tau}}$ is the [[Upper-convected time derivative]].



### Model Identification Matrix
By training the parameters $\epsilon$ and $\alpha$, we can classify the flow behavior as follows:

| Model | $\epsilon$ | $\alpha$ | Description |
|---|---|---|---|
| **Oldroyd-B** | $0$ | $0$ | Classic linear viscoelastic model. |
| **Giesekus** | $0$ | $\neq 0$ | Models shear-thinning and normal stress differences using a quadratic stress term. |
| **Linear PTT** | $\neq 0$ | $0$ | Models polymeric chain extensibility using a trace-dependent coefficient. |
| **Non-matching** | $\neq 0$ | $\neq 0$ | Represents a combined model or indicates that none of the three models alone fits the fluid. |

---

## Technical Implementation & Physical Details

The implementation spans across the main physics script, optimization pipeline, and plotting helpers.

### 1. Physics Model (`ViscoelasticPhysics`)
In `Viscoelastic_physics.py`, the parameters $\epsilon$ and $\alpha$ are registered as trainable parameters if `inverse_mode` is enabled, or as static buffers otherwise.

To guarantee physical validity, the parameters are passed through a `torch.abs()` mapping in the residual calculations to ensure they remain non-negative.

#### Tensor Components for $\boldsymbol{\tau} \cdot \boldsymbol{\tau}$
In two dimensions:
$$
\boldsymbol{\tau} \cdot \boldsymbol{\tau} = \begin{bmatrix} (\tau^{xx})^2 + (\tau^{xy})^2 & \tau^{xy}(\tau^{xx} + \tau^{yy}) \\ \tau^{xy}(\tau^{xx} + \tau^{yy}) & (\tau^{xy})^2 + (\tau^{yy})^2 \end{bmatrix}
$$

#### Optimization of Residual Equations
To reduce computational overhead, common scalar divisions like $\frac{\lambda}{\eta_p}$ are pre-calculated once as `lam_over_etap`. Additionally, during **Phase 1** of the staged training when the momentum loss weight is zero, the calculation of Navier-Stokes second-order derivatives is bypassed to save GPU time and VRAM.

Further, under the stream function formulation ($\psi$), evaluating the second spatial derivative $v_{yy}$ is optimized using Schwarz's theorem (since cross-derivatives commute for smooth and continuous fields):
$$ v_{yy} = \frac{\partial^2 v}{\partial y^2} = -\frac{\partial^3 \psi}{\partial x \partial y^2} = -u_{yx} $$
Thus, the code directly assigns `v_yy = -u_yx`, preventing PyTorch from constructing an entire branch of third-order derivatives in the autograd graph, which dramatically limits VRAM consumption (see [[VRAM_Optimization]] for details).

For the full dimensional analysis and mathematical derivation of the scaled Navier-Stokes and PTT-Giesekus residuals shown below, see [[Nondimensionalization]].

```python
# Extract effective physical parameters
mu_s_eff = torch.abs(self.mu_s)
mu_p_eff = torch.abs(self.mu_p)
lam_eff  = torch.abs(self.lam)
eps_eff  = torch.abs(self.epsilon)
alpha_eff = torch.abs(self.alpha)

# Pre-calculate common scalar factors
lam_over_etap = lam_eff / mu_p_eff
tr_tau = tau_xx + tau_yy

# Compute coefficients
ptt_coeff = 1.0 + (eps_eff * lam_over_etap) * tr_tau
giesekus_coeff = alpha_eff * lam_over_etap

# Unified Rheological Residuals (Eq. 13)
f_tau_xx = ptt_coeff * tau_xx + lam_eff * (
    u * tau_xx_x + v * tau_xx_y - 2 * u_x * tau_xx - 2 * u_y * tau_xy
) + giesekus_coeff * (tau_xx**2 + tau_xy**2) - 2 * mu_p_eff * u_x

f_tau_yy = ptt_coeff * tau_yy + lam_eff * (
    u * tau_yy_x + v * tau_yy_y - 2 * v_x * tau_xy - 2 * v_y * tau_yy
) + giesekus_coeff * (tau_xy**2 + tau_yy**2) - 2 * mu_p_eff * v_y

f_tau_xy = ptt_coeff * tau_xy + lam_eff * (
    u * tau_xy_x + v * tau_xy_y - u_x * tau_xy - u_y * tau_yy - tau_xx * v_x - tau_xy * v_y
) + giesekus_coeff * (tau_xy * tr_tau) - mu_p_eff * (u_y + v_x)
```

---

## Parameter Monitoring & Plotting Pipeline

An exhaustive tracking system is required to evaluate the convergence of all five inverse parameters ($\eta_s$, $\eta_p$, $\lambda$, $\epsilon$, $\alpha$).

### 1. Parameter Clamping & Registration
During both Adam and L-BFGS optimization phases, the parameters must be registered for optimization and clamped at each iteration step to prevent numerical instability or negative physical values.

In `Viscoelastic_PINN.py`:
```python
def setup_inverse_parameters(physics_problem):
    params_to_clamp = []
    if getattr(physics_problem, 'inverse_mode', False):
        for p_name in ['mu_s', 'mu_p', 'lam', 'epsilon', 'alpha']:
            p_val = getattr(physics_problem, p_name)
            if isinstance(p_val, torch.Tensor) and p_val.requires_grad:
                params_to_clamp.append(p_val)
    return params_to_clamp
```

At each optimization epoch, we clamp the parameters in-place:
```python
# Within training loop:
clamp_physical_parameters_(params_to_clamp)
```

### 2. Convergence Logs & History Tracking
The training history dictionary (`history_entry`) captures the scalar values at each log interval:
```python
if getattr(physics_problem, 'inverse_mode', False):
    history_entry.update({
        'param_etas': physics_problem.mu_s.item(),
        'param_etap': physics_problem.mu_p.item(),
        'param_lam': physics_problem.lam.item(),
        'param_epsilon': physics_problem.epsilon.item(),
        'param_alpha': physics_problem.alpha.item()
    })
```

### 3. Matplotlib 5-Subplot Visualization
In `history_tracker.py`, the `plot_physical_parameters` method generates a stacked plot showing the convergence of all parameters:
* Subplot 0: $\eta_s$ (Solvent Viscosity)
* Subplot 1: $\eta_p$ (Polymer Viscosity)
* Subplot 2: $\lambda$ (Relaxation Time)
* Subplot 3: $\epsilon$ (PTT Extensibility)
* Subplot 4: $\alpha$ (Giesekus Mobility)

Each subplot draws a solid line representing the learned parameter values across training steps, alongside a horizontal dashed line depicting the exact/target parameter values.

### 4. CSV Grid Search Logging
To track parameter values across grid search iterations, the learned values are extracted and saved to `results.csv` via `update_results_csv` with new column fields `param_epsilon` and `param_alpha`.

---

## References & Back-links
- [[ViscoelasticNet]]
- [[Oldroyd_B_Model]]
- [[Giesekus_Viscosity_Model]]
- [[Loss_History_Tracking]]
- [[Staged_Training_Procedure]]
- [[Inverse_Problems]]
