# Report - Curl del Momentum (Analisi di Compatibilità e Rotore)

## Summary
- **Source**: Internal technical research report (`Reference/Report - Curl del Momentum.md`).
- **Core Focus**: Analyzes the mathematical and physical origin of the **Loss Floor** in Phase 2 of staged viscoelastic PINN training on the four-roll mill geometry.
- **Key Insight**: Derives the irrotational compatibility condition $\nabla \times \mathbf{F}(\mathbf{u}, \boldsymbol{\tau}) = \mathbf{0}$ for the momentum force field $\mathbf{F} = \nabla p$. Shows that any numerical error in frozen velocities or stresses generates a solenoidal component $\mathbf{F}_{\text{rot}}$ that makes $\nabla p$ unintegrable, establishing a strictly positive lower bound $\mathcal{L}_{\text{momentum}} \ge \frac{1}{2}\langle \|\mathbf{F}_{\text{rot}}\|^2 \rangle > 0$.

---

## Key Methodology & Mathematical Derivations

### 1. The Momentum Force Field $\mathbf{F}$
In steady 2D non-dimensional viscoelastic flow:
$$ \nabla p = \mathbf{F}(\mathbf{u}, \boldsymbol{\tau}) = -Re (\mathbf{u} \cdot \nabla \mathbf{u}) + \beta \nabla^2 \mathbf{u} + \nabla \cdot \boldsymbol{\tau} $$
where $\mathbf{u} = (u, v)^T$ and $\boldsymbol{\tau}$ is the polymeric extra-stress tensor.

### 2. Rotational Compatibility Condition & Helmholtz Decomposition
By vector calculus, the curl of any $C^2$ scalar gradient is identically zero:
$$ \nabla \times (\nabla p) \equiv \mathbf{0} \implies \text{curl}(\mathbf{F}) = \frac{\partial F_y}{\partial x} - \frac{\partial F_x}{\partial y} = 0 $$
By Helmholtz-Hodge decomposition, any vector field $\mathbf{F}$ decomposes into a gradient and a solenoidal component:
$$ \mathbf{F} = \nabla p_{\text{true}} + \mathbf{F}_{\text{rot}}, \quad \nabla \cdot \mathbf{F}_{\text{rot}} = 0, \quad \nabla \times \mathbf{F}_{\text{rot}} \neq \mathbf{0} $$
Therefore, the momentum residual loss cannot fall below the rotational energy:
$$ \mathcal{L}_{\text{momentum}} = \frac{1}{2} \left\langle \|\nabla p - \mathbf{F}\|^2 \right\rangle = \frac{1}{2} \left\langle \|\nabla p - \nabla p_{\text{true}} - \mathbf{F}_{\text{rot}}\|^2 \right\rangle \ge \frac{1}{2} \left\langle \|\mathbf{F}_{\text{rot}}\|^2 \right\rangle > 0 $$

### 3. Verification Protocol: PINN Checkpoint vs. High-Fidelity COMSOL Fit
To distinguish between Autograd numerical differentiation limits and PINN physical inconsistencies:
- **Case A (PINN Phase 1 Model)**: Evaluates $\text{curl}(\mathbf{F})_{\text{PINN}}$ on `checkpoint_psi+tau`.
- **Case B (Data-Driven COMSOL Interpolator)**: Trains a neural interpolator strictly on high-fidelity COMSOL velocity and stress fields ($L_2 < 10^{-5}$) and evaluates $\text{curl}(\mathbf{F})_{\text{COMSOL\_fit}}$.
- **Diagnostic Metrics**:
  - Mean absolute curl: $\langle |\text{curl}(\mathbf{F})| \rangle = \frac{1}{N} \sum |\text{curl}(\mathbf{F})_i|$
  - Maximum absolute curl: $|\text{curl}(\mathbf{F})|_{\max} = \max |\text{curl}(\mathbf{F})_i|$
  - Curl/Force inconsistency ratio: $\text{Ratio} = \frac{\langle |\text{curl}(\mathbf{F})| \rangle}{\langle |\mathbf{F}| \rangle} \times 100\%$

---

## Key Findings & Project Impact

- **Diagnosis of Pressure Plateaus**: Proved that high pressure relative errors ($>100\%$) during early Phase 2 implementations were not optimizer failures, but mathematical consequences of $\text{curl}(\mathbf{F}) \neq 0$ in frozen Phase 1 fields.
- **Architectural Solutions**: Directly motivated the implementation of:
  1. [[Soft_Anti_Drift]] (relaxing the hard freeze of $\psi$ in Phase 2).
  2. [[Vorticity_Regularization]] (adding vorticity transport as a Phase 1 loss).
  3. [[Vorticity_Inversion_Solvent]] (identifying $\eta_s$ via curl-of-momentum directly).
  4. [[Zero_Stress_BC_Compatibility]] (eliminating roll stress BCs via curl constraints).

---

## Related Concepts
- **Topics**: [[Pressure_Stress_Decoupling]], [[Fluid_Dynamics]], [[PINN_Fundamentals]], [[Viscoelastic_Parameter_Identifiability]]
- **Methods**: [[Vorticity_Inversion_Solvent]], [[Vorticity_Regularization]], [[Soft_Anti_Drift]], [[Zero_Stress_BC_Compatibility]]
- **Systems**: [[Viscoelastic_Fluids]], [[Viscoelastic_Training]]
