# Pressure Scaling Issues in Momentum Equation

## The Pitfall of Scaling Momentum by $p_{scale}$
A common mistake in Physics-Informed Neural Networks (PINNs) applied to fluid dynamics is attempting to normalize the Momentum equation residuals by dividing them by the pressure scale factor ($p_{scale}$). 

In our framework, the pressure is parameterized as:
$$ p_{vero} = p_{rete} \cdot p_{scale} $$
where $p_{rete}$ is the normalized output of the neural network (ideally $\mathcal{O}(1)$), and $p_{scale}$ is the maximum absolute pressure in the domain.

The dimensionless Momentum equation is a balance of forces:
$$ f_u = Re (\mathbf{v} \cdot \nabla \mathbf{v}) + \nabla p_{vero} - \beta \nabla^2 \mathbf{v} - \nabla \cdot \boldsymbol{\tau}_{vero} $$
Substituting the parameterized pressure, we get:
$$ f_u = Re (\mathbf{v} \cdot \nabla \mathbf{v}) + p_{scale} \nabla p_{rete} - \beta \nabla^2 \mathbf{v} - \nabla \cdot \boldsymbol{\tau}_{vero} $$

If we divide the entire equation by $p_{scale}$ to "normalize" it, we obtain:
$$ \frac{f_u}{p_{scale}} = \frac{Re (\mathbf{v} \cdot \nabla \mathbf{v}) - \beta \nabla^2 \mathbf{v} - \nabla \cdot \boldsymbol{\tau}_{vero}}{p_{scale}} + \nabla p_{rete} $$

### The "Lazy Network" Effect (Gradient Starvation)
In complex geometries (e.g., the 4-roll mill), pressure singularities near walls or stagnation points cause $p_{scale}$ to be extremely large. 
By dividing by a massive $p_{scale}$, all the kinematic and rheological terms (velocity gradients, viscous dissipation, polymeric stress divergence) are artificially shrunk to near-zero. 

The equation effectively becomes:
$$ \frac{f_u}{p_{scale}} \approx 0 + \nabla p_{rete} $$

To minimize this loss, the neural network will take the easiest mathematical path: it will learn a completely flat pressure field ($p_{rete} = \text{const} \implies \nabla p_{rete} = 0$). Since the other physics terms have been mathematically suppressed by the division, the overall loss evaluates to a tiny number (e.g., $10^{-4}$). The optimizer is "satisfied" and stops learning, completely ignoring the true fluid dynamics. The physical coupling is destroyed.

---

## The Correct Solution: Intrinsic Dimensionless Momentum Balance ($\text{scale}_{mom} = 1.0$)

Rather than relying on heuristic ad-hoc divisors such as dividing by $p_{scale}$ (which causes gradient starvation) or dividing by `tau_scale * shear_max`, the production framework adopts a **rigorous global viscous nondimensionalization** (see [[Nondimensionalization]]):
$$ \tau_0 = \frac{\eta_0 U_{\text{ref}}}{H_{\text{ref}}}, \qquad p = \frac{p^*}{\tau_0}, \qquad \boldsymbol{\tau} = \frac{\boldsymbol{\tau}^*}{\tau_0}, \qquad \mathbf{u} = \frac{\mathbf{u}^*}{U_{\text{ref}}}, \qquad \mathbf{x} = \frac{\mathbf{x}^*}{H_{\text{ref}}} $$

Under these definitions, the Cauchy momentum equation transforms into:
$$ Re_{\text{scale}} (\mathbf{u} \cdot \nabla \mathbf{u}) + \nabla p - \tilde{\eta}_s \nabla^2 \mathbf{u} - \nabla \cdot \boldsymbol{\tau} = \mathbf{0} $$

### Key Theoretical Advantages:
1. **Natural $\mathcal{O}(1)$ Balance**: Every term in this dimensionless momentum equation is naturally of order $\mathcal{O}(1)$ without needing any artificial scalar denominator.
2. **Preservation of Force Coupling**: Setting $\text{scale}_{mom} = 1.0$ (`final_roll/src/physics.py`) guarantees that $\nabla p$ is directly coupled to $\nabla \cdot \boldsymbol{\tau}$ and viscous diffusion, completely avoiding the flat-pressure degenerate attractor.
3. **Algebraic Hard Anchoring**: Gauge indeterminacy is resolved via exact hard anchoring in the forward pass rather than through penalty loss terms (see [[Pressure_Point_Anchoring]]).

## References
- [[Nondimensionalization]]
- [[Pressure_Point_Anchoring]]
- [[Viscoelastic_Residual_Scaling]]
- [[Numerical_Hygiene_and_Phase2_Reforms]]
