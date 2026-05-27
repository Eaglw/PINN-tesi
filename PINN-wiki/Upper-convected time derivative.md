### Correct Mathematical Form

For a second-order tensor $\boldsymbol{\tau}$, the upper-convected derivative is:

$$
\overset{\nabla}{\boldsymbol{\tau}}
=
\frac{\partial \boldsymbol{\tau}}{\partial t}
+ \boldsymbol{u}\cdot\nabla \boldsymbol{\tau}
- (\nabla \boldsymbol{u})^{T}\cdot \boldsymbol{\tau}
- \boldsymbol{\tau}\cdot (\nabla \boldsymbol{u})
$$

In your 2D case, with velocity $\boldsymbol{u}=(u,v)$ and polymeric stress:

$$
\boldsymbol{\tau}
=
\begin{bmatrix}
\tau^{xx} & \tau^{xy} \\
\tau^{xy} & \tau^{yy}
\end{bmatrix}
$$

the componentwise expressions are:

$$
\overset{\nabla}{\tau}^{xx}
=
\frac{\partial \tau^{xx}}{\partial t}
+ u\frac{\partial \tau^{xx}}{\partial x}
+ v\frac{\partial \tau^{xx}}{\partial y}
- 2u_x\tau^{xx}
- 2u_y\tau^{xy}
$$

$$
\overset{\nabla}{\tau}^{yy}
=
\frac{\partial \tau^{yy}}{\partial t}
+ u\frac{\partial \tau^{yy}}{\partial x}
+ v\frac{\partial \tau^{yy}}{\partial y}
- 2v_x\tau^{xy}
- 2v_y\tau^{yy}
$$

$$
\overset{\nabla}{\tau}^{xy}
=
\frac{\partial \tau^{xy}}{\partial t}
+ u\frac{\partial \tau^{xy}}{\partial x}
+ v\frac{\partial \tau^{xy}}{\partial y}
- u_x\tau^{xy}
- u_y\tau^{yy}
- v_x\tau^{xx}
- v_y\tau^{xy}
$$

---

### Oldroyd-B Reduction

For the **ViscoelasticNet** unified equation, the [[Oldroyd_B_Model]] is recovered by setting $\epsilon=0$ and $\alpha=0$, collapsing the constitutive law to:

$$
\boldsymbol{\tau}
+ \lambda \overset{\nabla}{\boldsymbol{\tau}}
= \eta_p \left( \nabla \boldsymbol{u} + \nabla \boldsymbol{u}^{T} \right)
$$

which is the standard Oldroyd-B polymeric stress equation in differential form.

For **steady flow**, the time derivatives vanish and the residuals in `Viscoelastic_physics.py` implement:

$$
\overset{\nabla}{\tau}^{xx}
= u\,\tau^{xx}_x + v\,\tau^{xx}_y - 2u_x\tau^{xx} - 2u_y\tau^{xy}
$$

$$
\overset{\nabla}{\tau}^{yy}
= u\,\tau^{yy}_x + v\,\tau^{yy}_y - 2v_x\tau^{xy} - 2v_y\tau^{yy}
$$

$$
\overset{\nabla}{\tau}^{xy}
= u\,\tau^{xy}_x + v\,\tau^{xy}_y - u_x\tau^{xy} - u_y\tau^{yy} - v_x\tau^{xx} - v_y\tau^{xy}
$$

> [!note] Codice
> Questi termini corrispondono esattamente alle espressioni in `f_tau_xx`, `f_tau_yy` e `f_tau_xy` nel blocco dei residui costitutivi.

---
### Why Use the Upper-Convected Derivative?

The upper-convected time derivative $\overset{\nabla}{\boldsymbol{\tau}}$ is used because it gives an **objective** constitutive description, meaning the stress evolution does not depend on the observer's rigid-body motion or coordinate frame. This is essential in viscoelastic flow, where the constitutive law must represent material behaviour rather than artifacts of translation or rotation.

A second advantage is that it follows the deformation of a material element while correcting for its stretching and rotation under the velocity gradient. In polymeric fluids, this makes it well suited to describe how elastic stress is transported and reoriented by the flow.

A third advantage is that it is the natural derivative behind the Oldroyd-B and Upper-Convected Maxwell families of models, so it captures key viscoelastic effects such as stress advection, normal-stress generation, and nonlinear coupling between flow kinematics and polymer stress.

| Property | Benefit |
|---|---|
| **Frame objectivity** | Constitutive law is independent of rigid-body motion of the observer |
| **Material frame tracking** | Captures stretching and rotation of the fluid element |
| **Polymer physics** | Naturally models stress advection and normal-stress differences |

---

### Back-links
- [[ViscoelasticNet_Unified_Model]]
- [[Oldroyd_B_Model]]
- [[Giesekus_Viscosity_Model]]
- [[Inverse_Problems]]
