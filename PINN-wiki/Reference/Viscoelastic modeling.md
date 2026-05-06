## Pagina 1

Lecture 6
Modeling and simulation of non-Newtonian fluids

---

## Pagina 2

Newtonian fluids

• Let us consider a fluid between two parallel plates with the upper one moving at a constant velocity $U$ (shear flow) $^1$

• If the fluid is Newtonian, the following relationship holds:

$$F = \eta A \frac{U}{H} \rightarrow T = \frac{F}{A} = \eta \frac{U}{H}$$

(1)

• The quantity $\eta$ is the viscosity that is independent of the velocity gradient and only depends on temperature

• More in general:

$$T_{yx} = \eta \frac{du_x}{dy} = \eta \dot{\gamma}$$

$$T_{yx} = \text{shear stress}$$
$$\dot{\gamma} = \text{shear rate}$$

$^1$ In this kind of flow, the only non-zero component of the velocity profile is along $x$ that varies linearly along $y$

---

## Pagina 3

Newtonian fluids

• In a general flow field, the tensional state of the fluid in a point is described by the stress tensor $T$:

$$T = \begin{pmatrix}
T_{xx} & T_{xy} & T_{xz} \\
T_{yx} & T_{yy} & T_{yz} \\
T_{zx} & T_{zy} & T_{zz}
\end{pmatrix} = -pI + \sigma$$

(2)

• For Newtonian fluids, the constitutive equation, i.e., the law that links the deviatoric part of the stress tensor to the velocity gradient, is:

$$\sigma = 2\eta D$$

(3)

where $D$ is the rate-of-deformation tensor:

$$D = \frac{1}{2} \left( \nabla u + (\nabla u)^T \right)$$

where $\nabla u = \begin{pmatrix}
\frac{\partial u_x}{\partial x} & \frac{\partial u_y}{\partial x} & \frac{\partial u_z}{\partial x} \\
\frac{\partial u_x}{\partial y} & \frac{\partial u_y}{\partial y} & \frac{\partial u_z}{\partial y} \\
\frac{\partial u_x}{\partial z} & \frac{\partial u_y}{\partial z} & \frac{\partial u_z}{\partial z}
\end{pmatrix}$$

---

## Pagina 4

Newtonian fluids

• By substituting the constitutive equation (3) in equation (2), we get the generalization of the scalar equation (1)

• For a shear flow, we have:

$$\nabla u = \begin{pmatrix} 0 & 0 & 0 \\ \frac{\partial u_x}{\partial y} & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 0 & 0 \\ \dot{\gamma} & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}$$

• Hence:

$$T = -pI + \sigma = -pI + 2\eta D = -pI + \eta \left[ \nabla u + (\nabla u)^T \right] =$$

$$= \begin{pmatrix} -p & 0 & 0 \\ 0 & -p & 0 \\ 0 & 0 & -p \end{pmatrix} + \begin{pmatrix} 0 & 0 & 0 \\ \eta \dot{\gamma} & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix} + \begin{pmatrix} 0 & \eta \dot{\gamma} & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix} = \begin{pmatrix} -p & \eta \dot{\gamma} & 0 \\ \eta \dot{\gamma} & -p & 0 \\ 0 & 0 & -p \end{pmatrix}$$

• In shear flow, the normal stresses ($\sigma_{xx}, \sigma_{yy}, \sigma_{zz}$) are zero

---

## Pagina 5

Non-Newtonian fluids

• Several fluids do not follow the constitutive equation “stress proportional to the velocity gradient”

• These fluids are defined non-Newtonian

• Examples:
  - polymeric fluids
  - paints
  - detergents
  - foods
  - blood
  - shampoo
  - rubber

• These fluids give rise to peculiar phenomena that are not observed in Newtonian fluids

---

## Pagina 6

Non-Newtonian fluids

• Typical non-Newtonian phenomena

- Die swell or Barus effect
  https://www.youtube.com/watch?v=dYRZTJScill

- Weissenberg effect
  https://www.youtube.com/watch?v=npZzlgKjs0I
  https://www.youtube.com/watch?v=Lzl40NYnBKo

- Fano flow or tubeless syphon
  https://www.youtube.com/watch?v=aY7xiGQ-7iw
  https://www.youtube.com/watch?v=t3neqUhoDRA

- Liquid-like/solid-like material
  https://www.youtube.com/watch?v=G1Op_1yG6lQ
  https://www.youtube.com/watch?v=2mYHGn_Pd5M

---

## Pagina 7

Shear viscosity

• How can we explain these phenomena?

• Let us again consider a shear flow field with a non-Newtonian fluid between the plates

• Measuring the shear stress by varying the shear rate, we found that the direct proportionality between these two quantities does not hold anymore

$$T_{xy} \neq \eta \dot{\gamma}$$

• Specifically, at high shear rates, the shear stress may be higher or lower than the linear case observed for a Newtonian fluid

---

## Pagina 8

Shear viscosity

• Defining the viscosity as the ratio between the shear stress and the shear rate:
$$\eta = \frac{T_{xy}}{\dot{\gamma}}$$
we get the bottom diagram

• Hence, for a non-Newtonian fluid, the viscosity is not constant but it is a function of the shear rate

• In case the function is increasing, the fluid is called shear-thickening

• If it is decreasing (that is the most common behavior), the fluid is shear-thinning

---

## Pagina 9

Elongational viscosity

• Let us now consider a uniaxial elongational flow field

• In this kind of flow, a fluid element is “expanded” along one direction (for instance $x$) and equally “compressed” along the other two (for instance $y$ and $z$)

• The velocity gradient for this kind of flow is given by:

$$\nabla u = \left( \begin{array}{cccc} \frac{\partial u_x}{\partial x} & 0 & 0 \\ 0 & \frac{\partial u_y}{\partial y} & 0 \\ 0 & 0 & \frac{\partial u_z}{\partial z} \end{array} \right)$$

---

## Pagina 10

Elongational viscosity

• For the continuity equation: $\nabla \cdot \boldsymbol{u} = \frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y} + \frac{\partial u_z}{\partial z} = 0$

• Since in uniaxial elongational flow the $y$ and $z$ directions are undistinguishable:

$$\frac{\partial u_y}{\partial y} = \frac{\partial u_z}{\partial z}$$

• Let us define the elongational rate:

$$\frac{\partial u_x}{\partial x} = \dot{\epsilon} \Rightarrow \frac{\partial u_y}{\partial y} = \frac{\partial u_z}{\partial z} = -\frac{\dot{\epsilon}}{2}$$

and then:

$$\nabla \boldsymbol{u} = \begin{pmatrix}
\dot{\epsilon} & 0 & 0 \\
0 & -\dot{\epsilon}/2 & 0 \\
0 & 0 & -\dot{\epsilon}/2
\end{pmatrix}$$

---

## Pagina 11

Elongational viscosity

• The velocity field on the coordinate planes is:

• We can define the elongational viscosity $\eta_{\text{el}}$ as:

$$\eta_{\text{el}} = \frac{T_{xx} - T_{yy}}{\dot{\epsilon}}$$

• What is the relationship between elongational and shear viscosity for a Newtonian fluid?

---

## Pagina 12

Elongational viscosity

• The stress components $T_{xx}$ and $T_{yy}$ are given by:

$$T_{xx} = -p + 2\eta \frac{\partial u_x}{\partial x} = -p + 2\eta \dot{\epsilon}$$

$$T_{yy} = -p + 2\eta \frac{\partial u_y}{\partial y} = -p - \eta \dot{\epsilon}$$

from which we can calculate the elongational viscosity:

$$\eta_{\text{el}} = \frac{T_{xx} - T_{yy}}{\dot{\epsilon}} = \frac{-p + 2\eta \dot{\epsilon} + p + \eta \dot{\epsilon}}{\dot{\epsilon}} = 3\eta$$

• We find out that the uniaxial elongational viscosity for a Newtonian fluid is three times higher than the shear viscosity

• The ratio $\eta_{\text{el}} / \eta = 3$ is called Trouton ratio

• Notice that $\eta_{\text{el}}$ is independent of the elongational rate

---

## Pagina 13

Elongational viscosity

• What about for a non-Newtonian fluid?

• The experimental data in figure show that:

1. at low elongational rates, $\eta_{\text{el}}$ is constant and the Trouton ratio relationship holds

2. by increasing the elongational rate, the elongational viscosity is not constant anymore and increases

---

## Pagina 14

Normal stresses

• Let us come back to the shear flow experiment

• For sufficiently large values of the shear rate, forces that tend to move the plates far away appear

• This means that the normal components of the stress tensor switch on

• Hence, the deviatoric part of the stress tensor will be:

$$\sigma = \begin{pmatrix}
\sigma_{xx} & \sigma_{xy} & 0 \\
\sigma_{xy} & \sigma_{yy} & 0 \\
0 & 0 & \sigma_{zz}
\end{pmatrix}$$

resulting in the following total stress tensor

$$T = \begin{pmatrix}
-p + \sigma_{xx} & \sigma_{xy} & 0 \\
\sigma_{xy} & -p + \sigma_{yy} & 0 \\
0 & 0 & -p + \sigma_{zz}
\end{pmatrix}$$

---

## Pagina 15

Normal stresses

• Due to the pressure contribution, the normal stresses $\sigma_{xx}, \sigma_{yy}, \sigma_{zz}$ cannot be directly measured

• Only their differences can be measured as the pressure cancels out

• We can define the first normal stress difference:

$$N_1 = T_{xx} - T_{yy} = (-p + \sigma_{xx}) - (-p + \sigma_{yy}) = \sigma_{xx} - \sigma_{yy}$$

and the second normal stress difference:

$$N_2 = T_{yy} - T_{zz} = (-p + \sigma_{yy}) - (-p + \sigma_{zz}) = \sigma_{yy} - \sigma_{zz}$$

• Of course, the third normal stress difference is useless as it can be directly obtained from $N_1$ and $N_2$

---

## Pagina 16

Normal stresses

• Experimental data show that:
  - $N_1$ is positive
  - $N_2$ is generally negative
  - $N_2$ is about one order of magnitude lower than $N_1$
  - at low shear rates, $N_1$ and $|N_2|$ are proportional to $\dot{\gamma}^2$

• It is convenient to normalize $N_1$ and $N_2$ with $\dot{\gamma}^2$ and define the first and second normal stress coefficients:

$$\Psi_1 = \frac{N_1}{\dot{\gamma}^2} \quad \Psi_2 = \frac{N_2}{\dot{\gamma}^2}$$

• Generally, both $\Psi_1$ and $|\Psi_2|$ are shear-thinning

---

## Pagina 17

Time-dependent properties

• Let us consider a start-up shear flow experiment:
  - $t < 0$: the fluid is at rest
  - $t \ge 0$: the upper plate starts to move a constant velocity

• A Newtonian fluid instantaneously reaches a steady-state stress value

• On the contrary, a transient is observed for a non-Newtonian fluid

---

## Pagina 18

Time-dependent properties

• A similar behavior is observed in a relaxation experiment:
  - $t < T$: the upper plate moves at a constant velocity
  - $t \ge T$: the upper plate is stopped

• The characteristic time needed to reach the steady-state is called relaxation time

• A Newtonian fluid has an extremely low ($\sim 0$) relaxation time

---

## Pagina 19

Non-Newtonian fluids

• In summary:

| Property | Newtonian fluids | Non-Newtonian fluids |
| :--- | :--- | :--- |
| Shear viscosity | Constant with shear rate $T_{xy} = \eta \dot{\gamma}$ | Function of shear rate $T_{xy} = \eta (\dot{\gamma}) \dot{\gamma}$ |
| Elongational viscosity | Constant with elongational rate $T_{xx} - T_{yy} = \eta_{\text{el}} \dot{\epsilon}$ | Function of elongational rate $T_{xx} - T_{yy} = \eta_{\text{el}} (\dot{\epsilon}) \dot{\epsilon}$ |
| Normal stresses | $N_1 = N_2 = 0$ | $N_1 \neq 0$ $N_2 \neq 0$ |
| Time-dependent | $T_{xy} = \text{constant in time}$ | $T_{xy} = T_{xy}(t)$ $N_1 = N_1(t)$ $N_2 = N_2(t)$ |

Shear-thickening $\eta(\dot{\gamma}) \uparrow$ if $\dot{\gamma} \uparrow$
Shear-thinning $\eta(\dot{\gamma}) \downarrow$ if $\dot{\gamma} \uparrow$
Elongational thickening $\eta_{\text{el}}(\dot{\epsilon}) \uparrow$ if $\dot{\epsilon} \uparrow$
Low shear $N_1 \propto \dot{\gamma}^2$ $-N_2 \propto \dot{\gamma}^2$
High shear $N_1, N_2$ thinning
Relaxation time $\lambda$

---

## Pagina 20

Generalized Newtonian fluids

Property Newtonian fluids Non-Newtonian fluids

Shear viscosity Constant with shear rate $T_{xy} = \eta \dot{\gamma}$ Function of shear rate $T_{xy} = \eta(\dot{\gamma})\dot{\gamma}$

Elongational viscosity Constant with elongational rate $T_{xx} - T_{yy} = \eta_{\text{el}} \dot{\epsilon}$ Function of elongational rate $T_{xx} - T_{yy} = \eta_{\text{el}}(\dot{\epsilon})\dot{\epsilon}$

Normal stresses $N_1 = N_2 = 0$ $N_1 \neq 0$
$N_2 \neq 0$

Time-dependent $T_{xy} = \text{constant in time}$ $T_{xy} = T_{xy}(t)$ $N_1 = N_1(t)$ $N_2 = N_2(t)$

GENERALIZED NEWTONIAN FLUIDS or PURELY VISCOUS FLUIDS

These fluids are characterized by a non-constant viscosity but a negligible elasticity

Property:
1) viscosity function of the shear rate

---

## Pagina 21

Viscoplastic fluids

Property Newtonian fluids Non-Newtonian fluids

Shear viscosity Constant with shear rate $T_{xy} = \eta \dot{\gamma}$
Yield stress $T_{xy} = \tau_{yield} + \eta \dot{\gamma}$
if $|\tau| > \tau_{yield}$

Elongational viscosity Constant with elongational rate $T_{xx} - T_{yy} = \eta_{\text{el}} \dot{\epsilon}$
Function of elongational rate $T_{xx} - T_{yy} = \eta_{\text{el}} (\dot{\epsilon}) \dot{\epsilon}$

Normal stresses $N_1 = N_2 = 0$
$N_1 \neq 0$
$N_2 \neq 0$

Time-dependent $T_{xy} = \text{constant in time}$
$T_{xy} = T_{xy}(t)$
$N_1 = N_1(t)$
$N_2 = N_2(t)$

VISCOPLASTIC FLUIDS
These fluids are characterized by a yield stress below which they behave like solids otherwise like liquids

Property:
1) yield stress

---

## Pagina 22

Viscoelastic fluids

Property | Newtonian fluids | Non-Newtonian fluids
--- | --- | ---
Shear viscosity | Constant with shear rate $T_{xy} = \eta \dot{\gamma}$ | Function of shear rate $T_{xy} = \eta(\dot{\gamma})\dot{\gamma}$
Elongational viscosity | Constant with elongational rate $T_{xx} - T_{yy} = \eta_{\text{el}} \dot{\epsilon}$ | Function of elongational rate $T_{xx} - T_{yy} = \eta_{\text{el}}(\dot{\epsilon})\dot{\epsilon}$
Normal stresses | $N_1 = N_2 = 0$ | $N_1 \neq 0$
$N_2 \neq 0$
Time-dependent | $T_{xy} = \text{constant in time}$ | $T_{xy} = T_{xy}(t)$
$N_1 = N_1(t)$
$N_2 = N_2(t)$

VISCOELASTIC FLUIDS
These fluids show a behavior between viscous liquids and elastic solids
Properties:
1) shear-thinning
2) normal stresses
3) relaxation time

---

## Pagina 23

Generalized Newtonian fluids

---

## Pagina 24

Generalized Newtonian fluids

• Several fluids show a significant variation of the viscosity on the shear rate but very small normal stresses

• These fluids belong to the category of generalized Newtonian fluids

• Due to the industrial relevance, several constitutive equations accounting for the variability of the viscosity with the shear rate have been proposed

• These equations can be used when the fluid elasticity is negligible

---

## Pagina 25

Generalized Newtonian fluids

• Let us assume that we want to predict the flow field of a generalized Newtonian fluid in a domain through numerical simulations

• The fluid is incompressible and the governing equations are the usual continuity and momentum balance:

$$\nabla \cdot u = 0$$

$$\rho \left( \frac{\partial u}{\partial t} + u \cdot \nabla u \right) = -\nabla p + \nabla \cdot \sigma$$

• For a Newtonian fluid:

$$\sigma = 2\eta D \quad \Rightarrow \quad \nabla \cdot \sigma = \nabla \cdot \left[ \eta \left( \nabla u + (\nabla u)^T \right) \right] = \eta \nabla^2 u$$

• For a generalized Newtonian fluid:

$$\sigma = 2\eta (\dot{\gamma}) D \quad \Rightarrow \quad \nabla \cdot \sigma = \nabla \cdot \left[ \eta (\dot{\gamma}) \left( \nabla u + (\nabla u)^T \right) \right] \neq \eta \nabla^2 u$$

---

## Pagina 26

Generalized Newtonian fluids

• We need to specify what we mean by shear rate $\dot{\gamma}$

• For a shear flow, the shear rate is given by: $\dot{\gamma} = \frac{du_x}{dy}$

• We can define a shear rate for a general flow field as$^1$:

$$\dot{\gamma} = \sqrt{2D : D} = \sqrt{2\operatorname{Tr}(D \cdot D)} = \left[ 2 \left( \frac{\partial u_x}{dx} \right)^2 + 2 \left( \frac{\partial u_y}{dy} \right)^2 + 2 \left( \frac{\partial u_z}{dz} \right)^2 + \left( \frac{\partial u_x}{dy} + \frac{\partial u_y}{dx} \right)^2 + \left( \frac{\partial u_x}{dz} + \frac{\partial u_z}{dx} \right)^2 \right]^{\frac{1}{2}}$$

• For a shear flow, this expression reduces to the first one

• A constitutive equation for a generalized Newtonian fluid is $\eta = \eta(\dot{\gamma})$ where $\dot{\gamma}$ is the quantity defined above

$^1$ This quantity is the square root of the negative of the second invariant of the tensor $E = \nabla u + (\nabla u)^T = 2D$, i.e.:

$$\dot{\gamma} = \sqrt{-II_E} = \sqrt{-\frac{1}{2} \left[ (\operatorname{Tr}E)^2 - \operatorname{Tr}(E \cdot E) \right]} = \sqrt{\frac{1}{2} \operatorname{Tr}(E \cdot E)} = \sqrt{\frac{1}{2} E : E} = \sqrt{2D : D}$$ with $\operatorname{Tr}E = 0$ for incompressible fluids

---

## Pagina 27

Generalized Newtonian fluids

• To solve the governing equations we need an expression for the viscosity as a function of the shear rate

• In other words we need a constitutive equation for the generalized Newtonian fluid

• Several expressions have been proposed in the literature

POWER-LAW MODEL: $\eta(\dot{\gamma}) = m\dot{\gamma}^{n-1}$ 2 parameters

CARREAU MODEL: $\frac{\eta(\dot{\gamma}) - \eta_{\infty}}{\eta_{0} - \eta_{\infty}} = \left[1 + (\lambda \dot{\gamma})^{2}\right]^{\frac{n-1}{2}}$ 4 parameters

CARREAU-YASUDA MODEL: $\frac{\eta(\dot{\gamma}) - \eta_{\infty}}{\eta_{0} - \eta_{\infty}} = \left[1 + (\lambda \dot{\gamma})^{a}\right]^{\frac{n-1}{a}}$ 5 parameters

---

## Pagina 28

Power-law model

POWER-LAW MODEL: $\eta(\dot{\gamma}) = m \dot{\gamma}^{n-1}$ 2 parameters

• For $n = 1$ the Newtonian case is recovered where $m$ is the (constant) viscosity

• For $n < 1$ the model predicts shear-thinning

• For $n > 1$ the model predicts shear-thickening

• It is an extremely simple model, inadequate to describe the behavior of a fluid at low shear rate values (the viscosity diverges)

• Useful to describe shear-thinning regions

---

## Pagina 29

Carreau and Carreau-Yasuda models

CARREAU MODEL: $\frac{\eta(\dot{\gamma}) - \eta_{\infty}}{\eta_0 - \eta_{\infty}} = \left[1 + (\lambda \dot{\gamma})^2\right]^{\frac{n-1}{2}}$ 4 parameters

CARREAU-YASUDA MODEL: $\frac{\eta(\dot{\gamma}) - \eta_{\infty}}{\eta_0 - \eta_{\infty}} = \left[1 + (\lambda \dot{\gamma})^a\right]^{\frac{n-1}{a}}$ 5 parameters

• These models are more realistic as they predict shear-thinning and the Newtonian plateau at low and high shear rates

CARREAU: $\eta_0 = 1, \eta_{\infty} = 0.1, \lambda = 1$

CARREAU: $\eta_0 = 1, \eta_{\infty} = 0.1, n = 0.5$

---

## Pagina 30

Power-law fluid in a tube

• Let us see what are the effects of a variable viscosity on the fluid dynamics of a power-law fluid

• We consider the motion of a fluid in a tube

• Because of the geometry, it is convenient to introduce a cylindrical reference frame

• We want to calculate the radial velocity profile

---

## Pagina 31

Power-law fluid in a tube

• For Newtonian fluids, the definition of the Reynolds number is:

$$Re = \frac{\rho U D}{\eta}$$

• The viscosity of a power-law, however, varies with the shear rate

• How can we define the Reynolds number?

• Several expressions of the Reynolds number for non-Newtonian fluids are available in the literature

• One possible expression for power-law fluids is$^1$:

$$Re = \frac{\rho U^2 - n D^n}{m \left( \frac{3n+1}{4n} \right)^n 8^{n-1}}$$

$^1$ Madlener, Frey & Ciezki, Generalized Reynolds number for non-Newtonian fluids, *Prog. Prop. Phys.* (2009)

---

## Pagina 32

Power-law fluid in a tube

• Following the same procedure for Newtonian fluids, the momentum balance reduces to:

$$-\frac{\partial p}{\partial z} + \frac{1}{r} \frac{\partial}{\partial r} (r \sigma_{\text{rz}}) = 0$$

• Since the pressure only changes along $z$ and the tangential stress along $r$, we can replace the partial derivatives with the total derivatives:

$$-\frac{dp}{dz} + \frac{1}{r} \frac{d}{dr} (r \sigma_{\text{rz}}) = 0$$

• Integrating with respect to $z$ we have:

$$\frac{1}{r} \frac{d}{dr} (r \sigma_{\text{rz}}) = -\frac{p_0 - p_L}{L} = -\frac{\Delta p}{L}$$

---

## Pagina 33

Power-law fluid in a tube

• Integrating the previous equation with respect to $r$, after some algebraic manipulation we get:

$$\sigma_{\text{rz}} = -\frac{\Delta p}{2L} r + \frac{C_1}{r}$$

• Since at $r = 0$ (the tube axis) the stress must be finite, $C_1$ must be zero:

$$\sigma_{\text{rz}} = -\frac{\Delta p}{2L} r$$

• Now we can consider the constitutive equation that, for a power-law fluid, is given in tensorial form by:

$$\sigma = 2\eta(\dot{\gamma})D = 2m \dot{\gamma}^{n-1} D$$

---

## Pagina 34

Power-law fluid in a tube

• The tangential stress $\sigma_{rz}$ is given by$^1$:
$$\sigma_{rz} = m \left| \frac{du_z}{dr} \right|^{n-1} \frac{du_z}{dr}$$

from which:
$$\sigma_{rz} = m \left( -\frac{du_z}{dr} \right)^{n-1} \left( \frac{du_z}{dr} \right) = -m \left( -\frac{du_z}{dr} \right)^n$$

• The equation to be solved will be:
$$-m \left( -\frac{du_z}{dr} \right)^n = -\frac{\Delta p}{2L} r \Rightarrow m \left( -\frac{du_z}{dr} \right)^n = \frac{\Delta p}{2L} r \Rightarrow \left( -\frac{du_z}{dr} \right)^n = \frac{\Delta p}{2mL} r$$
$$- \frac{du_z}{dr} = \left( \frac{\Delta p}{2mL} \right)^{\frac{1}{n}} r^{\frac{1}{n}} \Rightarrow -u_z = \left( \frac{\Delta p}{2mL} \right)^{\frac{1}{n}} \frac{1}{\frac{1}{n}+1} r^{\frac{1}{n}+1} + C_2$$

$^1$ Recall that the shear rate is a positive quantity. Hence, the absolute value of $du_z/dr$ (that can be negative) must be considered.

---

## Pagina 35

Power-law fluid in a tube

• The constant $C_2$ can be computed considering that at $r = R, u_z = 0$:

$$C_2 = -\left(\frac{\Delta p}{2mL}\right)^{\frac{1}{n}} \frac{1}{\frac{1}{n} + 1} R^{\frac{1}{n} + 1}$$

• In conclusion, we get:

$$-u_z = \left(\frac{\Delta p}{2mL}\right)^{\frac{1}{n}} \frac{1}{\frac{1}{n} + 1} r^{\frac{1}{n} + 1} - \left(\frac{\Delta p}{2mL}\right)^{\frac{1}{n}} \frac{1}{\frac{1}{n} + 1} R^{\frac{1}{n} + 1}$$

$$u_z = \left(\frac{\Delta p}{2mL}\right)^{\frac{1}{n}} \frac{R^{\frac{1}{n} + 1}}{\frac{1}{n} + 1} \left[1 - \left(\frac{r}{R}\right)^{\frac{1}{n} + 1}\right]$$

---

## Pagina 36

Power-law fluid in a tube

• For $r = 0$ we have the maximum velocity $u_{z,\max}$ and then:

$$u_z = \left( \frac{\Delta p}{2mL} \right)^{\frac{1}{n}} \frac{R^{\frac{1}{n}+1}}{\frac{1}{n}+1} \left[ 1 - \left( \frac{r}{R} \right)^{\frac{1}{n}+1} \right] = u_{z,\max} \left[ 1 - \left( \frac{r}{R} \right)^{\frac{1}{n}+1} \right]

• For $n = 1$ the expression gives the well-known parabolic profile

• For decreasing values of $n$, the profile becomes more and more flat

• In the limit of $n \to 0$, a plug flow profile is obtained

---

## Pagina 37

Power-law fluid in a tube

• Finally, we can evaluate the flow rate through the tube cross-section as:

$$Q = \int_{0}^{2\pi} \int_{0}^{R} u_z(r) r dr d\theta = 2\pi \int_{0}^{R} u_z(r) r dr = \frac{\pi R^3}{\frac{1}{n} + 3} \left( \frac{\Delta p R}{2mL} \right)^{\frac{1}{n}}$$

• The above equation can be used to measure the viscosity of the fluid (i.e., the parameter $m$ and $n$) by applying a flow rate and measuring the pressure drop or the other way around

• Another useful relation is the link between the axial velocity and the flow rate:

$$u_z = \frac{Q}{\pi R^2} \frac{3n + 1}{n + 1} \left[ 1 - \left( \frac{r}{R} \right)^{\frac{1}{n} + 1} \right]$$

---

## Pagina 38

Viscoplastic fluids

---

## Pagina 39

Viscoplastic fluids

• Fluids that exhibit a critical stress below which they do not flow are called viscoplastic fluids

• The critical stress is named “yield stress”

• Hence, if the stress in the material is lower than the yield stress, the fluid behaves like a solid otherwise like a liquid

• Examples:
  - Toothpaste
  - Hair gel
  - Cosmetic creams
  - Mayonnaise
  - Mud

---

## Pagina 40

Viscoplastic fluids

• Several constitutive equations have been proposed to model the viscoplastic behavior

• The simplest one is the Bingham model$^1$ that, for a shear flow, is expressed as:

$$\sigma_{xy} = \sigma_{yield} + K \dot{\gamma} \quad \text{if} \ |\sigma_{xy}| > \sigma_{yield}$$
$$\dot{\gamma} = 0 \quad \text{if} \ |\sigma_{xy}| \leq \sigma_{yield}$$

where $\sigma_{yield}$ and $K$ are two constitutive parameters

• As the material stress is larger than the yield stress $\sigma_{yield}$, it linearly increases with the shear rate

$^1$ Bingham was one of the first scientists to study this kind of fluids. For this reason the viscoplastic materials are also defined Bingham plastic

---

## Pagina 41

Viscoplastic fluids

• The Bingham constitutive equation is characterized by a discontinuity when the shear stress is equal to $\sigma_{\text{yield}}$

• To avoid numerical problems, modifications of the original constitutive equation have been proposed

• A popular one is proposed by Papanastasiou$^1$ where the Bingham model is replaced by:

$$\sigma_{\text{xy}} = \sigma_{\text{yield}} \left[ 1 - \exp(-m \dot{\gamma}) \right] + K \dot{\gamma}$$

• The parameter $m$ is defined “regularization” parameter

$^1$ Papanastasiou, Flow of materials with yield, J. Rheol. 31 (1987) 385-404

---

## Pagina 42

Viscoplastic fluids

• The original Bingham model is obtained for $m$ equal to infinity

• Too small values of $m$ lead to significant deviations from the original model

• Too large values may lead to numerical problems

• If an estimate of the shear rate range is possible, the value of $m$ could be properly chosen

• Otherwise a trial-and-error procedure is needed

---

## Pagina 43

Viscoplastic fluids

• To solve the flow field of a Bingham fluid in an arbitrary geometry, we need the tensorial form of the previous constitutive equation:

$$\sigma = 2 \left( \frac{\sigma_{\text{yield}}}{\dot{\gamma}} + K \right) D \quad \text{if } |\sigma| > \sigma_{\text{yield}}$$
$$D = 0 \quad \text{if } |\sigma| \leq \sigma_{\text{yield}}$$

where:

$$\dot{\gamma} = \sqrt{2D : D}$$

$$|\sigma| = \sqrt{\frac{1}{2}\sigma : \sigma}$$

• The last quantity is the square root of the negative of the second invariant of the deviatoric part of the stress tensor (denoted by $II_{\sigma}$)$^1$

$$| \sigma | = \sqrt{-II_{\sigma}} = \sqrt{-\frac{1}{2} \left[ (\text{Tr} \sigma)^2 - \text{Tr} (\sigma \cdot \sigma) \right]} = \sqrt{\frac{1}{2} \text{Tr} (\sigma \cdot \sigma)} = \sqrt{\frac{1}{2} \sigma : \sigma}$$

$$= 0 \text{ because } \sigma \text{ is traceless by definition}$$

---

## Pagina 44

Viscoplastic fluids

• Let us suppose we are interested in solving the fluid dynamics problem of a viscoplastic fluid in an arbitrary geometry

• Since these fluids generally have a very high viscosity (even several orders of magnitude higher than water), we can assume a Stokes regime:

$$\nabla \cdot \boldsymbol{u} = 0$$

$$\nabla \cdot \boldsymbol{\sigma} = -\nabla p + 2\nabla \cdot [\eta(\dot{\gamma})\boldsymbol{D}] = 0$$

$$\eta(\dot{\gamma}) = \frac{\sigma_{\text{yield}}}{\dot{\gamma}} + K$$

• We can make the equations dimensionless selecting $K$ to nondimensionalize the stress:

$$\nabla \cdot \boldsymbol{u} = 0$$

$$-\nabla p + 2\nabla \cdot \left[1 + \frac{Bn}{\dot{\gamma}}\boldsymbol{D}\right] = 0$$

---

## Pagina 45

Viscoplastic fluids

• The dimensionless Bingham number $Bn$ appears:

$$Bn = \frac{\sigma_{\text{yield}} D}{KU}$$

where $D$ and $U$ are a characteristic length and velocity

• A Newtonian fluid is obtained for $Bn = 0$

• A classical example is the sedimentation of a sphere in an unbounded viscoplastic fluid (i.e., without walls)

$$Bn = \frac{\sigma_{\text{yield}} D}{KU_s}$$

$$C_s = \frac{F}{6\pi KU_s R}$$

---

## Pagina 46

Viscoplastic fluids

• The simulations also predict the extension of the “yielded” and “unyielded” regions, i.e., where the stress is lower or higher that the yield stress

$Bn = 1.28$

• In the yielded region the fluid behaves like a solid (no velocity gradient) whereas the in unyielded region it flows like a liquid

---

## Pagina 47

Viscoelastic fluids

---

## Pagina 48

Viscoelastic fluids

• The generalized Newtonian model is an extension of the linear relation between stress and velocity gradient of the Newtonian law

• The structure of the stress tensor for a Generalized Newtonian fluid is, indeed, the same as the Newtonian fluid

• Furthermore, the velocity field adjusts instantaneously to a change in the stresses

• One of the key features of many non-Newtonian fluids is the presence of memory, i.e., the stresses depend on the flow history

• Another important feature is stress anisotropy, i.e., a non-Newtonian fluid generates stress components that are not present in a Newtonian fluid under the same flow field

---

## Pagina 49

Viscoelastic fluids

• These fluids are called viscoelastic since they exhibit both viscous and elastic behavior under an external solicitation

• Let us consider a material showing both viscous and elastic behavior under a simple shear deformation

• The strength of the deformation is characterized by the deformation gradient $\gamma$

• The shear stress in an elastic solid is given by the Hooke’s law:

$$\sigma = G \gamma$$

where $G$ is the elastic constant of the material or the shear modulus

---

## Pagina 50

Viscoelastic fluids

• The shear stress in an viscous fluid is given by the Newtonian law:

$$\sigma = \eta \dot{\gamma}$$

where $\eta$ is the fluid viscosity and $\dot{\gamma}$ is the velocity gradient

• The simplest viscoelastic model is a linear combination of the two previous materials$^1$:

• This model is called Maxwell fluid

$^1$ Another viscoelastic model can be obtained by combining the spring and the dashpot in parallel. In this case the model is called Kelvin-Voigt solid.

---

## Pagina 51

Viscoelastic fluids

• Let us now apply a total deformation gradient $\gamma$ to the Maxwell material, leading to a total stress $\sigma$

• Let us also denote the deformation gradient and the stress for the solid elastic and the viscous fluid with subscripts “s” and “v”:

$\gamma_s = G \gamma_s$
$\sigma_s = G \gamma_s$

$\gamma_v = \eta \dot{\gamma}_v$

• We have: $\gamma = \gamma_s + \gamma_v$
$\sigma = \sigma_s = \sigma_v$

---

## Pagina 52

Viscoelastic fluids

• From the previous relations we have:
$$\dot{\gamma} = \dot{\gamma}_s + \dot{\gamma}_v \rightarrow \dot{\gamma} = \frac{\dot{\sigma}_s}{G} + \frac{\sigma_v}{\eta} \rightarrow \dot{\gamma} = \frac{\dot{\sigma}}{G} + \frac{\sigma}{\eta} \rightarrow \sigma + \frac{\eta}{G} \dot{\sigma} = \eta \dot{\gamma}$$

• Introducing the relaxation time $\lambda = \eta/G$, we get the linear Maxwell constitutive equation:
$$\sigma + \lambda \dot{\sigma} = \eta \dot{\gamma}$$

• This model can be formally solved giving:
$$\sigma(t) = \frac{1}{\lambda} \int_{-\infty}^{t} \exp \left( -\frac{t-t'}{\lambda} \right) \eta \dot{\gamma}(t') dt'$$

i.e., the stress is time-dependent and relaxes exponentially on the time scale $\lambda$ (viscous behavior) while, at short times $(t-t' \sim 0)$, it is $\sigma(t) \sim \eta \gamma(t)/\lambda$ (solid behavior)

---

## Pagina 53

Viscoelastic fluids

• We want now to extend the Maxwell model to an arbitrary flow

• We could try to replace the stress and the velocity gradient by their tensorial form$^1$:

$$\sigma + \lambda \frac{\partial \sigma}{\partial t} = 2\eta D$$

• However, this equation suffers from a serious physical problem as it is not frame-invariant

• To show this, let us make two experiments:

- one in a stationary laboratory frame
- one on a frame moving with a constant velocity $u_0$ with respect to the laboratory frame

$^1$ As for the Newtonian fluid, the symmetric part of the velocity gradient (i.e., the rate of deformation tensor) is the one that produces a stress in the fluid

---

## Pagina 54

Viscoelastic fluids

• The coordinates of the moving frame $\hat{x}$ can be related to the coordinates of the laboratory frame $x$ as:

$$\hat{x} = x + u_0 t$$

• Hence, the components of the stress tensor in the moving frame can be written in terms of the laboratory frame coordinates as:

$$\hat{\sigma}_{ij}(\hat{x}, t) = \sigma_{ij}(x + u_0 t, t)$$

leading to$^1$:

$$\frac{\partial \hat{\sigma}_{ij}}{\partial t} = \frac{\partial}{\partial t} \sigma_{ij}(x + u_0 t, t) = \frac{\partial \sigma_{ij}}{\partial t} + u_0 \cdot \nabla \sigma_{ij}$$

• Of course, the two experiments should be described by the same equations as a constant velocity does not alter the velocity gradient

• However, the derivatives of $\sigma_{ij}$ and $\hat{\sigma}_{ij}$ differ by a term proportional to $u_0$

$^1$ Recall the multivariable chain rule for derivatives

---

## Pagina 55

Viscoelastic fluids

• This is a problem similar to a vector field transported by a fluid where the frame-invariant time derivative is the material derivative

• We need an expression for the derivative of a second-rank tensor that is frame-invariant

• Omitting the derivation$^1$, it can be shown that one possible frame-invariant time derivative is$^2$:

$$\nabla \sigma \equiv \frac{\partial \sigma}{\partial t} + u \cdot \nabla \sigma - (\nabla u)^T \cdot \sigma - \sigma \cdot \nabla u$$

• This is called the upper-convected time derivative, denoted by the symbol “$\nabla$”

$^1$ The derivation can be found here: [http://www1.maths.leeds.ac.uk/~smt/TEACHING/MATH3454/chapter2_new.pdf](http://www1.maths.leeds.ac.uk/~smt/TEACHING/MATH3454/chapter2_new.pdf)

$^2$ Other expressions exist, e.g., the lower-convected derivative and the co-rotational derivative.

---

## Pagina 56

Viscoelastic fluids

• Replacing the time-derivative in the linear Maxwell model by the upper-convective derivative we get:

$$\lambda \nabla \sigma + \sigma = 2\eta D$$

that is the upper-convected Maxwell (UCM) model

• As previously said, $\lambda$ is the fluid relaxation time that represents the time scale over which stress variations due to an applied deformation take place

• The tensor $\sigma$ is called viscoelastic extra-stress and contributes to the total stress tensor similarly to the Newtonian case:

$$T = -pI + \sigma$$

---

## Pagina 57

Viscoelastic fluids

• In some polymeric solutions, a Newtonian solvent is added, modifying the total stress tensor as:

$$T = -pI + 2\eta_s D + \sigma$$

where $\eta_s$ is the viscosity of the solvent

• In this case, the previous constitutive equation:

$$\lambda \nabla \sigma + \sigma = 2\eta_p D$$

is called Oldroyd-B model

• Notice the subscript “p” to the viscosity in order to distinguish the solvent contribution to the viscosity $\eta_s$ to the polymer contribution to the viscosity $\eta_p$

---

## Pagina 58

Viscoelastic fluids

• The Maxwell and Oldroyd-B models are the simplest constitutive equations for viscoelastic fluids

• We will see that they have some limitations and can produce unphysical results in some flows

• Several other more accurate constitutive equations have been proposed, such as the Giesekus (GSK) model:

$$\lambda \nabla \sigma + \sigma + \frac{\alpha \lambda}{\eta_p} \sigma \cdot \sigma = 2 \eta_p D$$

that includes a non-linear extra term

• The parameter $\alpha$ is a constitutive parameter called “mobility”

• For $\alpha = 0$ the UCM/Oldroyd-B model is recovered

---

## Pagina 59

Viscoelastic fluids

• Another commonly used viscoelastic constitutive equation is the Phan-Thien Tanner (PTT) model:

$$\lambda \nabla \sigma + f(\sigma) \sigma = 2 \eta_p D$$

• The function $f(\sigma)$ can be taken in an exponential form:

$$f(\sigma) = \exp \left[ \frac{\lambda \epsilon}{\eta_p} \text{Tr}(\sigma) \right]$$

or in a linear form:

$$f(\sigma) = 1 + \frac{\lambda \epsilon}{\eta_p} \text{Tr}(\sigma)$$

• The parameter $\varepsilon$ is the constitutive parameter of the PTT model

• For $\varepsilon = 0$ the UCM/Oldroyd-B model is recovered

---

## Pagina 60

Viscoelastic fluids

• Let us now see what are the rheological properties predicted by UCM and GSK models in shear and elongational flow

• UCM model ($\eta = 1, \lambda = 1$)
  - constant $\eta$
  - constant $\Psi_1 (= N_1 \propto \dot{\gamma}^2)$
  - zero $\Psi_2$
  - divergent $\eta_{el}$

---

## Pagina 61

Viscoelastic fluids

• Let us now see what are the rheological properties predicted by UCM and GSK models in shear and elongational flow

• GSK model ($\eta = 1, \lambda = 1$)
  - shear-thinning $\eta$
  - shear-thinning $\Psi_1$
  - shear-thinning $-\Psi_2$
  - elongational thickening $\eta_{el}$

---

## Pagina 62

Viscoelastic fluids

• The fluid dynamics governing equations are still the continuity and momentum balances together with a constitutive equation

• Assuming incompressible fluid and isothermal conditions:

$$\nabla \cdot \boldsymbol{u} = 0$$

$$\rho \left( \frac{\partial \boldsymbol{u}}{\partial t} + \boldsymbol{u} \cdot \nabla \boldsymbol{u} \right) = -\nabla p + \eta_{\text{s}} \nabla^{2} \boldsymbol{u} + \nabla \cdot \boldsymbol{\sigma} + \boldsymbol{F}$$

$$\lambda \nabla \boldsymbol{\sigma} + \boldsymbol{\sigma} = 2\eta_{\text{p}} \boldsymbol{D}$$  (...or other constitutive equations...)

• As compared to the Newtonian case, an extra stress term appears on the right-hand side

• We have 10 scalar equations and 10 unknowns ($\boldsymbol{u}, p, \boldsymbol{\sigma}$)

---

## Pagina 63

Viscoelastic fluids

• It is convenient to make the constitutive equations dimensionless

• Let us introduce the following dimensionless variables:

$$x^* = \frac{x}{D}$$

$$u^* = \frac{u}{U}$$

$$t^* = t \frac{U}{D}$$

$$\nabla^* = D\nabla$$

$$\nabla^*^2 = D^2\nabla^2$$

$$D^* = \frac{D}{U}D$$

$$p^* = \frac{D}{U\eta_0}p$$

$$\sigma^* = \frac{D}{U\eta_0}\sigma$$

$$T^* = \frac{D}{U\eta_0}T$$

where $D, U, U\eta_0/D$ are characteristic length, velocity and stress$^1$

• In the definition of the characteristic stress, $\eta_0$ is the zero-shear viscosity that is the viscosity at very low shear rates

• For a model without solvent: $\eta_0 = \eta$

• For a model with solvent: $\eta_0 = \eta_s + \eta_p$

$^1$ Notice that the stress is made dimensionless by using the viscosity. Indeed, the viscosity on Newtonian fluids is generally very high and viscous effects are predominant over inertial ones.

---

## Pagina 64

Viscoelastic fluids

• Substituting these variables in the governing equations we get$^1$:

$$\nabla \cdot \boldsymbol{u} = 0$$

$$Re \left( \frac{\partial \boldsymbol{u}}{\partial t} + \boldsymbol{u} \cdot \nabla \boldsymbol{u} \right) = -\nabla p + \eta_r \nabla^2 \boldsymbol{u} + \nabla \cdot \boldsymbol{\sigma}$$

$$Wi \nabla \boldsymbol{\sigma} + \boldsymbol{\sigma} = 2(1 - \eta_r) D$$

(Oldroyd-B model)

$$Wi \nabla \boldsymbol{\sigma} + \frac{\alpha Wi}{1 - \eta_r} \boldsymbol{\sigma} \cdot \boldsymbol{\sigma} + \boldsymbol{\sigma} = 2(1 - \eta_r) D$$

(GSK model)

$$Wi \nabla \boldsymbol{\sigma} + \exp \left[ \frac{\epsilon Wi}{1 - \eta_r} Tr(\boldsymbol{\sigma}) \right] \boldsymbol{\sigma} = 2(1 - \eta_r) D$$

(PTT exponential model)

$$Wi \nabla \boldsymbol{\sigma} + \left[ 1 + \frac{\epsilon Wi}{1 - \eta_r} Tr(\boldsymbol{\sigma}) \right] \boldsymbol{\sigma} = 2(1 - \eta_r) D$$

(PTT linear model)

where: $$Re = \frac{\rho U D}{\eta}$$

$$\eta_r = \frac{\eta_s}{\eta_0}$$

$$Wi = \frac{\lambda U}{D}$$

$^1$ The expression of the total stress tensor has been already substituted in the right-hand side of the momentum balance. Furthermore, the Newtonian solvent has been included in the stress tensor.

---

## Pagina 65

Viscoelastic fluids

• Along with the well–known Reynolds number, two more dimensionless parameters appear

• The viscosity ratio $\eta_r$ accounts for the weight of the solvent viscosity over the total viscosity; for a model without Newtonian solvent this parameter is zero

• The Weissenberg number $Wi$ compares the fluid characteristic time ($\lambda$) with the flow characteristic time ($D/U$)

• The Weissenberg number is a measure of the “fluid viscoelasticity”

• A Newtonian fluid is obtained for $Wi = 0$ (purely viscous behavior)

• At high values of the Weissenberg number the solid-like behavior becomes predominant

---

## Pagina 66

Viscoelastic fluids

• In summary, to predict the fluid dynamics of a viscoelastic fluid, the continuity and momentum balance equations must be solved together with a constitutive equation

• The equations are coupled and, in principle, they must be solved simultaneously at each time step

• Hence, the computational effort required for simulating the flow of a viscoelastic fluid is much larger than the Newtonian case

• To speed up the computations and reduce the memory requirements, numerical procedures aimed at decoupling the momentum balance and the constitutive equation have been proposed$^1$

$^1$ See, e.g., D’Avino and Hulsen, Decoupled second-order transient schemes for the flow of viscoelastic fluids without a viscous solvent contribution, *J. non-Newt. Fluid Mech.* 165 (2010) 1602-1612

---

## Pagina 67

Viscoelastic fluids

• Since the upper-convected derivative contains the time-derivative of the stress tensor, an initial condition for $\sigma$ must be specified for a time-dependent problem

• Due to the kind of the constitutive equations, the extra stress tensor needs to be specified only at an inflow boundary

• Indeed, since a viscoelastic fluid has a memory, the flow field on an inflow boundary depends on the fluid motion before entering the domain

• Hence, no boundary condition on $\sigma$ must be provided at a wall

• Regarding the other boundary conditions, many of them (symmetry, outflow, pressure inlet, axial symmetry, periodicity) are similar to those we have already discussed where the total stress tensor also includes $\sigma$

---

## Pagina 68

Viscoelastic fluids

• Due to the mathematical nature of the viscoelastic constitutive equations, severe numerical problems arise at moderate/high values of the Weissenberg number$^1$ ($Wi > 1$)

• These difficulties are related to numerical instabilities that occur regardless of the numerical method used to solve the equations

• Several remedies have been implemented to increase the Weissenberg number assuring numerical stability

• Nowadays, depending on the specific problem, relatively high $Wi$ values (up to 100) can be attained thanks to these stabilization techniques and mesh refinement

• Anyway, the high Weissenberg number problem is still an issue and represents the main limitation of the simulations in reproducing processes characterized by very large $Wi$ values

$^1$ Keunings, On the high Weissenberg number problem, *J. non-Newt. Fluid Mech.* 20 (1986) 209-226