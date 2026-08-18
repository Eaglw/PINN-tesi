# Method: Lasso Regularization for Constitutive Model Discovery

## Overview
Lasso ($L_1$) regularization applies an absolute penalty to parameter weights in Physics-Informed Neural Networks (PINNs). In viscoelastic fluid mechanics and rheological inverse problems, it enables **autonomous model discovery** and **parsimony** (Occam's razor) by driving superfluous non-linear parameters (such as the PTT parameter $\epsilon$ or the Giesekus parameter $\alpha$) to **exact zero** when the underlying flow behavior is governed by simpler constitutive laws (such as Oldroyd-B).

## Mathematical Formulation
In a unified viscoelastic PINN framework, the total loss function is augmented with an $L_1$ penalty on selected constitutive parameters:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + w_{\text{bc}} \mathcal{L}_{\text{bc}} + w_{\text{pde}} \mathcal{L}_{\text{pde}} + w_{\text{lasso}} \left( |\epsilon| + |\alpha| \right)$$

where:
- $\epsilon \ge 0$: Linear PTT non-linear mobility parameter.
- $\alpha \ge 0$: Giesekus non-linear anisotropic drag parameter.
- $w_{\text{lasso}}$: Regularization hyperparameter controlling the strength of the parsimony prior.

### Comparison: $L_1$ (Lasso) vs. $L_2$ (Ridge) Regularization
- **$L_2$ Regularization ($\mathcal{L}_{reg} = w \epsilon^2$):** The gradient $\frac{\partial \mathcal{L}_{reg}}{\partial \epsilon} = 2w\epsilon$ approaches zero as $\epsilon \to 0$. As a consequence, parameter values shrink towards zero but remain infinitesimally non-zero (dense parameters).
- **$L_1$ Regularization ($\mathcal{L}_{reg} = w |\epsilon|$):** The subgradient $\partial |\epsilon| = \text{sign}(\epsilon)$ provides a **constant restoring force** towards zero regardless of how small the parameter becomes. This yields exact sparsity ($\epsilon = 0$, $\alpha = 0$).

## Technical Implementation in Viscoelastic PINNs
1. **Bounded Parameter Transformations:** Since $\epsilon, \alpha \ge 0$, they are parameterized via $\text{softplus}(\cdot)$.
2. **Loss Integration:**
```python
loss_lasso = w_lasso * (torch.abs(physics.eps) + torch.abs(physics.alpha))
total_loss = total_loss + loss_lasso
```
3. **Staged Model Selection Workflow:**
   - **Step 1 (Parsimonious Exploration):** Train with active $L_1$ penalty on $(\epsilon, \alpha)$ alongside free $(\lambda, \beta)$.
   - **Step 2 (Hard Pruning):** If $|\epsilon| < \delta_{tol}$ and $|\alpha| < \delta_{tol}$, freeze $\epsilon=0$ and $\alpha=0$ and refine $(\lambda, \beta)$ with standard L-BFGS.

## Related Concepts
- **Topics**: [[Inverse_Problems]], [[Viscoelasticity]], [[Viscoelastic_Parameter_Identifiability]]
- **Methods**: [[ViscoelasticNet_Full model]], [[Staged_Training_Procedure]], [[Dynamic_Weighting]]
