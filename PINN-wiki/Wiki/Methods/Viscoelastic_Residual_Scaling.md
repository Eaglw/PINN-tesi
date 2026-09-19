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

### 2. Modern Rigorous Nondimensionalization vs Legacy Momentum Scaling
Historically, dividing the momentum residual by a heuristic factor ($\text{Scale}_{\text{momentum}} \sim \tau_{scale} \cdot \dot{\gamma}^*_{max}$) was tested to balance initial residual magnitudes.
However, as established in [[Pressure_Scaling_Issues]] and [[Nondimensionalization]]:
1. **Intrinsically Dimensionless Momentum Balance**:
   Under the global viscous scaling with $\tau_0 = \frac{\eta_0 U_{\text{ref}}}{H_{\text{ref}}}$ and $H_{\text{ref}} = R$:
   $$ Re_{\text{scale}} (\mathbf{u} \cdot \nabla \mathbf{u}) + \nabla p - \tilde{\eta}_s \nabla^2 \mathbf{u} - \nabla \cdot \boldsymbol{\tau} = \mathbf{0} $$
   All physical terms in the momentum equation are already dimensionless and balanced at $\mathcal{O}(1)$.
2. **Current Production State (`scale_mom = 1.0`)**:
   In `final_roll/src/physics.py`:
   ```python
   @property
   def scale_mom(self):
       """[Proposta AA - Rettificata] Il residuo di Navier-Stokes è già intrinsecamente adimensionale (scala = 1.0)."""
       return torch.tensor(1.0, device=self.eta_0.device, dtype=self.eta_0.dtype)
   ```
   No artificial divisor is applied to $f_u$ or $f_v$, preserving the true force balance.

### 3. Per-Component Vector Stress Normalization
For the constitutive equations, the extra-stress tensor components can differ significantly in magnitude (e.g., normal extensional stresses $\tau_{xx}, \tau_{yy}$ vs shear stress $\tau_{xy}$).
The production framework implements per-component normalization:
```python
# Bilanciamento Loss PDE per-componente [Proposta C]
s_xx, s_xy, s_yy = self.tau_scale[0, 0], self.tau_scale[0, 1], self.tau_scale[0, 2]
f_txx = f_txx / s_xx
f_tyy = f_tyy / s_yy
f_txy = f_txy / s_xy
```
This ensures that shear and normal stress equations contribute equally to the backpropagation loss without one component overwhelming the others.

---

## References & Back-links
- [[Nondimensionalization]]
- [[Viscoelastic_Training]]
- [[Loss_Functions]]
- [[Viscoelastic_Fluids]]
