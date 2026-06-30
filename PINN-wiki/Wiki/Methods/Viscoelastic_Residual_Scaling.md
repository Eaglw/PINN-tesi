# Viscoelastic Residual Scaling

## Overview
Viscoelastic flow simulations using PINNs face severe optimization stiffness (gradient pathology) because the Momentum and Constitutive equations govern physics at different mathematical derivative orders. Even in non-dimensionalized systems, spatial gradients near boundaries (like the rollers in a four-roll mill) can be orders of magnitude larger than the fields themselves. 

This page documents the mathematical reasoning for their different scaling behaviors and details a velocity-only heuristic to normalize both PDE residuals to $O(1)$ without requiring stress or pressure datasets.

---

## Mathematical Analysis of Scale Incoherence
Governing equations in viscoelastic flows, even when non-dimensionalized, exhibit different derivative behaviors:

### 1. Constitutive Equation Scale
The constitutive equation (e.g., PTT or Oldroyd-B) contains a dominant algebraic term (the extra-stress tensor $\boldsymbol{\tau}$ itself, without derivatives):
$$ f_{PTT}(\boldsymbol{\tau}) \boldsymbol{\tau} + Wi \cdot \overset{\nabla}{\boldsymbol{\tau}} - 2 \beta_p \mathbf{D} = 0 $$
For small or moderate Weissenberg numbers ($Wi$), the algebraic term $\boldsymbol{\tau}$ dictates the scale:
$$ \text{Scale}_{\text{constitutive}} \sim \tau_{scale} $$
Dividing the constitutive residuals ($f_{\tau_{xx}}$, $f_{\tau_{xy}}$, $f_{\tau_{yy}}$) by $\tau_{scale}$ brings the residual to $O(1)$.

### 2. Momentum Equation Scale
The momentum equation (in creeping flows) balances only spatial derivatives:
$$ Re (\mathbf{v} \cdot \nabla \mathbf{v}) + \nabla p - \beta \nabla^2 \mathbf{v} - \nabla \cdot \boldsymbol{\tau} = 0 $$
Here, the dominant force term is $\nabla \cdot \boldsymbol{\tau}$. Every term is a spatial gradient. In a dimensionless system, gradients are scaled by the inverse of the characteristic length $L_{char}$, which is equivalent to the maximum shear rate $\dot{\gamma}^*_{max} = \max |\nabla \mathbf{v}|$:
$$ \text{Scale}_{\text{momentum}} \sim \tau_{scale} \cdot \dot{\gamma}^*_{max} $$
In confined geometries like the 4-roll mill, local spatial gradients are extremely high ($\dot{\gamma}^*_{max} \approx 10 - 50$ near boundaries). Consequently:
- Dividing the momentum residual by `tau_scale` leaves a leftover factor of $\dot{\gamma}^*_{max}$ (resulting in MSE residuals of $O(10^2) - O(10^3)$).
- Not dividing it at all leads to $O(10^4)$ loss.
To bring the momentum residual to $O(1)$, it must be normalized by its own scale: $\tau_{scale} \cdot \dot{\gamma}^*_{max}$.

---

## Velocity-Only Heuristic for Scale Estimation
In inverse problems or configurations where only the velocity field $\mathbf{v}$ is known (no pressure or stress data is available), the scales can be estimated using the following heuristic:

1. **Calculate Maximum Strain Rate**:
   Use numerical differentiation on the velocity field to compute the maximum shear rate:
   $$ \dot{\gamma}^*_{max} = \max \sqrt{2 \text{tr}(\mathbf{D}^2)} $$
2. **Estimate Stress Scale**:
   For low-Wi viscoelastic flows, stress is roughly Newtonian:
   $$ \tau_{scale} \approx 2 \beta_p \dot{\gamma}^*_{max} $$
3. **Define Normalization Factors**:
   - Constitutive Residuals: $\text{norm}_{\text{const}} = \tau_{scale}$
   - Momentum Residuals: $\text{norm}_{\text{mom}} = \tau_{scale} \cdot \dot{\gamma}^*_{max}$

This allows direct $O(1)$ scaling of all PDE residuals using only velocity datasets.

---

## Technical Implementation
In `src/physics.py`, the residuals are updated:
```python
# Momentum (derivative terms)
f_u = f_u / self.momentum_scale   # = tau_scale * gamma_max
f_v = f_v / self.momentum_scale

# Constitutive (algebraic dominant)
f_txx = f_txx / self.tau_scale
f_tyy = f_tyy / self.tau_scale
f_txy = f_txy / self.tau_scale
```
Where `self.momentum_scale` is computed in `load_data()` from the maximum strain rate of the input velocity field and injected into the `Physics` class.

---

## References & Back-links
- [[Nondimensionalization]]
- [[Viscoelastic_Training]]
- [[Loss_Functions]]
- [[Viscoelastic_Fluids]]
