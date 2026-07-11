# Method: Vorticity Regularization for Staged Training

## Overview
In a [[Staged_Training_Procedure|staged training]] setup where the stream function $\psi$ (and hence velocity) is trained in Phase 1 **without** the momentum equation, the resulting velocity field satisfies boundary conditions and constitutive laws but has **unregularized higher-order spatial derivatives**. When Phase 2 attempts to recover the pressure field from the frozen velocity via the momentum equation, a fundamental mathematical obstruction arises: the [[Pressure_Stress_Decoupling#The Helmholtz-Hodge Pressure Inference Limit|Helmholtz-Hodge limit]].

This method proposes using the **vorticity transport equation** as a physics-based regularizer in Phase 1 to eliminate this obstruction.

## The Problem: Rotational Noise in $\mathbf{F}$

The momentum equation requires finding $p$ such that:
$$ \nabla p = \mathbf{F} = - Re\,(\mathbf{u} \cdot \nabla \mathbf{u}) + \beta\,\nabla^2 \mathbf{u} + \nabla \cdot \boldsymbol{\tau} $$

A scalar pressure $p$ exists if and only if $\mathbf{F}$ is conservative:
$$ \nabla \times \mathbf{F} = 0 $$

When $\psi$ is trained without momentum, the term $\nabla^2 \mathbf{u}$ (which involves **third-order derivatives** of $\psi$) is dominated by high-frequency numerical noise. Experimental measurement (Pearson correlation between Autograd $\nabla^2 u$ and COMSOL MLS $\nabla^2 u$: **-0.375**) confirms that these derivatives are not just noisy but **anticorrelated** with the true values.

This generates a large rotational component $\|\mathbf{g}\|^2$ in the Helmholtz-Hodge decomposition of $\mathbf{F}$, which acts as an **irreducible loss floor** (~0.93 in 4-roll mill experiments). No pressure network, however expressive, can fit this rotational residual.

## The Solution: Vorticity Equation as Phase 1 Regularizer

Taking the curl ($\nabla \times$) of the Navier-Stokes momentum equation eliminates the pressure gradient identically (since $\nabla \times \nabla p \equiv 0$), yielding the **vorticity transport equation**:

$$ Re\,(\mathbf{u} \cdot \nabla \omega) = \beta\,\nabla^2 \omega + \nabla \times (\nabla \cdot \boldsymbol{\tau}) $$

where $\omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$ is the scalar vorticity (in 2D).

### Why This Works
Adding the vorticity equation as a loss term in Phase 1 directly constrains the **second-order derivatives of velocity** (and hence **third-order derivatives of $\psi$**) to be physically consistent with the constitutive stress field. This ensures that:

1. $\nabla^2 \mathbf{u}$ computed via Autograd on $\psi$ is smooth and physically meaningful.
2. The curl $\nabla \times \mathbf{F} \approx 0$, making $\mathbf{F}$ nearly conservative.
3. Phase 2 can cleanly recover $p$ from $\nabla p = \mathbf{F}$ without hitting the Helmholtz-Hodge limit.

### Implementation Considerations
- **Computational Cost**: The vorticity equation involves $\nabla^2 \omega$, which requires **fourth-order derivatives** of $\psi$ via Autograd. This is computationally expensive and may amplify numerical noise in early training. A possible mitigation is to activate the vorticity loss only after a warmup period (e.g., after 20% of Phase 1 epochs).
- **Loss Weighting**: The vorticity loss should be weighted carefully relative to the constitutive loss and BC loss. A conservative starting point is $w_{\text{vort}} = 0.1 \times w_{\text{constitutive}}$.
- **Pressure Independence**: The key advantage over simply activating the full momentum equation in Phase 1 is that **no pressure network is needed**. The vorticity equation is a self-contained constraint on velocity and stress alone.

## Experimental Evidence (4-Roll Mill, July 2026)

| Metric | Without Vorticity Reg. | Expected With Vorticity Reg. |
|---|---|---|
| Pearson corr. $\nabla^2 u$ (Autograd vs MLS) | -0.375 | ~0.9+ |
| Momentum loss floor (Phase 2) | 0.933 | ~0.01 |
| L2 Error $p$ (Phase 2 plateau) | 154% | <10% |
| Training time Phase 1 overhead | — | +30-50% (est.) |

## Related
- [[Pressure_Stress_Decoupling]]: The Helmholtz-Hodge limit this method aims to overcome
- [[Staged_Training_Procedure]]: The multi-phase training workflow
- [[Sobolev_Regularization]]: Related concept of derivative-level supervision
- [[Fluid_Dynamics]]: Vorticity formulation of Navier-Stokes
- [[Viscoelastic_Training]]: Experiment guide
