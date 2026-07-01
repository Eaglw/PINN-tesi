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

## Why is `tau_scale * shear_max` different?
Dividing the Momentum equation by the momentum scale heuristic (`momentum_scale` = $\tau_{scale} \cdot \dot{\gamma}_{max}$) works because it targets the **true dominant force term** in the dimensionless system.

In creeping viscoelastic flows, the momentum balance is dominated by the divergence of the stress tensor ($\nabla \cdot \boldsymbol{\tau}$). The magnitude of this term is precisely given by the magnitude of the stress ($\tau_{scale}$) multiplied by the intensity of spatial variations (the maximum dimensionless shear rate $\dot{\gamma}_{max}$).

By dividing the equation by this specific scale:
$$ \frac{f_u}{\text{momentum\_scale}} $$
We are dividing the equation by the *exact expected magnitude* of its largest term. 
This ensures that the maximum residual naturally tops out at $\mathcal{O}(1)$. It prevents the Momentum Loss from exploding to $\mathcal{O}(10^3)$ (which would overshadow the Data Loss) without suppressing the internal balance of the equation, because we are using a scale derived from the kinematics/stresses themselves, not from an isolated scalar singularity like the pressure.
