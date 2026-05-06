---
title: "Upper-convected Maxwell model"
source: "https://en.wikipedia.org/wiki/Upper-convected_Maxwell_model"
author:
  - "[[Wikipedia]]"
published: 2005-12-30
created: 2026-05-06
description:
tags:
  - "clippings"
---
The **upper-convected Maxwell** (**UCM**) **model** is a generalisation of the [Maxwell material](https://en.wikipedia.org/wiki/Maxwell_material "Maxwell material") for the case of large deformations using the [upper-convected time derivative](https://en.wikipedia.org/wiki/Upper-convected_time_derivative "Upper-convected time derivative"). The model was proposed by [James G. Oldroyd](https://en.wikipedia.org/wiki/James_G._Oldroyd "James G. Oldroyd"). The concept is named after [James Clerk Maxwell](https://en.wikipedia.org/wiki/James_Clerk_Maxwell "James Clerk Maxwell"). It is the simplest observer independent constitutive equation for [viscoelasticity](https://en.wikipedia.org/wiki/Viscoelasticity "Viscoelasticity") and further is able to reproduce first normal stresses. Thus, it constitutes one of the most fundamental models for [rheology](https://en.wikipedia.org/wiki/Rheology "Rheology").

The model can be written as:

${\displaystyle \mathbf {T} +\lambda {\stackrel {\nabla }{\mathbf {T} }}=2\eta _{0}\mathbf {D} }$

where:

- ${\displaystyle \mathbf {T} }$ is the [stress](https://en.wikipedia.org/wiki/Stress_\(physics\) "Stress (physics)") [tensor](https://en.wikipedia.org/wiki/Tensor "Tensor");
- ${\displaystyle \lambda }$ is the relaxation time;
- ${\displaystyle {\stackrel {\nabla }{\mathbf {T} }}}$ is the [upper-convected time derivative](https://en.wikipedia.org/wiki/Upper-convected_time_derivative "Upper-convected time derivative") of stress tensor:

${\displaystyle {\stackrel {\nabla }{\mathbf {T} }}={\frac {\partial }{\partial t}}\mathbf {T} +\mathbf {v} \cdot \nabla \mathbf {T} -(\nabla \mathbf {v} )^{T}\cdot \mathbf {T} -\mathbf {T} \cdot (\nabla \mathbf {v} )}$

- ${\displaystyle \mathbf {v} }$ is the fluid velocity and the gradient of a vector follows the convention ${\displaystyle (\nabla {\mathbf {v} })_{ij}=\partial _{i}v_{j}}$.
- ${\displaystyle \eta _{0}}$ is material [viscosity](https://en.wikipedia.org/wiki/Viscosity "Viscosity") at steady [simple shear](https://en.wikipedia.org/wiki/Simple_shear "Simple shear");
- ${\displaystyle \mathbf {D} }$ is the [deformation rate tensor](https://en.wikipedia.org/wiki/Strain_rate_tensor "Strain rate tensor").

The model can be derived either by applying the concept of observer invariance to the [Maxwell material](https://en.wikipedia.org/wiki/Maxwell_material "Maxwell material") or by two different mesoscopic models, namely Hookean Dumbells [^1] or Temporary Networks.[^2] Even though both microscopic model lead to the upper evolution equation for the stress, recent work pointed up the differences when accounting also for the stress fluctuations. [^3]

## Case of the steady shear

For this case only two components of the shear stress became non-zero:

${\displaystyle T_{12}=\eta _{0}{\dot {\gamma }}\,}$

and

${\displaystyle T_{11}=2\eta _{0}\lambda {\dot {\gamma }}^{2}\,}$

where ${\displaystyle {\dot {\gamma }}}$ is the shear rate.

Thus, the upper-convected Maxwell model predicts for the simple shear that [shear stress](https://en.wikipedia.org/wiki/Shear_stress "Shear stress") to be proportional to the shear rate and the [first difference of normal stresses](https://en.wikipedia.org/w/index.php?title=First_difference_of_normal_stresses&action=edit&redlink=1 "First difference of normal stresses (page does not exist)") (${\displaystyle T_{11}-T_{22}}$) is proportional to the square of the shear rate, the [second difference of normal stresses](https://en.wikipedia.org/w/index.php?title=Second_difference_of_normal_stresses&action=edit&redlink=1 "Second difference of normal stresses (page does not exist)") (${\displaystyle T_{22}-T_{33}}$) is always zero. In other words, UCM predicts appearance of the first difference of normal stresses but does not predict [non-Newtonian behavior](https://en.wikipedia.org/wiki/Non-Newtonian_fluid "Non-Newtonian fluid") of the shear viscosity nor the second difference of the normal stresses.

Usually quadratic behavior of the first difference of normal stresses and no second difference of the normal stresses is a realistic behavior of polymer melts at moderated shear rates, but constant viscosity is unrealistic and limits usability of the model.

## Case of start-up of steady shear

For this case only two components of the shear stress became non-zero:

${\displaystyle T_{12}=\eta _{0}{\dot {\gamma }}\left(1-\exp \left(-{\frac {t}{\lambda }}\right)\right)}$

and

${\displaystyle T_{11}=2\eta _{0}\lambda {\dot {\gamma }}^{2}\left(1-\exp \left(-{\frac {t}{\lambda }}\right)\left(1+{\frac {t}{\lambda }}\right)\right)}$

The equations above describe stresses gradually risen from zero the steady-state values. The equation is only applicable, when the velocity profile in the shear flow is fully developed. Then the shear rate is constant over the channel height. If the start-up form a zero velocity distribution has to be calculated, the full set of PDEs has to be solved.

## Case of the steady state uniaxial extension or uniaxial compression

For this case UCM predicts the normal stresses ${\displaystyle \sigma =T_{11}-T_{22}=T_{11}-T_{33}}$ calculated by the following equation:

${\displaystyle \sigma ={\frac {2\eta _{0}{\dot {\epsilon }}}{1-2\lambda {\dot {\epsilon }}}}+{\frac {\eta _{0}{\dot {\epsilon }}}{1+\lambda {\dot {\epsilon }}}}}$

where ${\displaystyle {\dot {\epsilon }}}$ is the elongation rate.

The equation predicts the elongation viscosity approaching ${\displaystyle 3\eta _{0}}$ (the same as for the [Newtonian fluids](https://en.wikipedia.org/wiki/Newtonian_fluid "Newtonian fluid")) for the case of low elongation rate ( ${\displaystyle {\dot {\epsilon }}\ll {\frac {1}{\lambda }}}$) with fast deformation thickening with the steady state viscosity approaching infinity at some elongational rate (${\displaystyle {\dot {\epsilon }}_{\infty }={\frac {1}{2\lambda }}}$) and at some compression rate (${\displaystyle {\dot {\epsilon }}_{-\infty }=-{\frac {1}{\lambda }}}$). This behavior seems to be realistic.

## Case of small deformation

For the case of small deformation the nonlinearities introduced by the upper-convected derivative disappear and the model becomes an ordinary model of [Maxwell material](https://en.wikipedia.org/wiki/Maxwell_material "Maxwell material").

## References

- Macosko, Christopher (1993). *Rheology. Principles, Measurements and Applications*. VCH Publisher. [ISBN](https://en.wikipedia.org/wiki/ISBN_\(identifier\) "ISBN (identifier)") [1-56081-579-5](https://en.wikipedia.org/wiki/Special:BookSources/1-56081-579-5 "Special:BookSources/1-56081-579-5").

[^1]: Öttinger, H.C. (1996). *Stochastic processes in polymeric fluids: tools and examples for developing simulation algorithms* (1st ed.). [Springer-Verlag](https://en.wikipedia.org/wiki/Springer-Verlag "Springer-Verlag"). [doi](https://en.wikipedia.org/wiki/Doi_\(identifier\) "Doi (identifier)"):[10.1007/978-3-642-58290-5](https://doi.org/10.1007%2F978-3-642-58290-5). [ISBN](https://en.wikipedia.org/wiki/ISBN_\(identifier\) "ISBN (identifier)") [978-3-540-58353-0](https://en.wikipedia.org/wiki/Special:BookSources/978-3-540-58353-0 "Special:BookSources/978-3-540-58353-0").

[^2]: Larson, Ronald G. (28 January 1999). *The Structure and Rheology of Complex Fluids (Topics in Chemical Engineering): Larson, Ronald G.: 9780195121971: Amazon.com: Books*. Oup USA. [ISBN](https://en.wikipedia.org/wiki/ISBN_\(identifier\) "ISBN (identifier)") [019512197X](https://en.wikipedia.org/wiki/Special:BookSources/019512197X "Special:BookSources/019512197X").

[^3]: Winters, A.; Öttinger, H. C.; Vermant, J. (2024). ["Comparative analysis of fluctuations in viscoelastic stress: A comparison of the temporary network and dumbbell models"](https://pubs.aip.org/aip/jcp/article/161/1/014901/3300367/Comparative-analysis-of-fluctuations-in). *Journal of Chemical Physics*. **161**: 014901. [arXiv](https://en.wikipedia.org/wiki/ArXiv_\(identifier\) "ArXiv (identifier)"):[2404.19743](https://arxiv.org/abs/2404.19743). [doi](https://en.wikipedia.org/wiki/Doi_\(identifier\) "Doi (identifier)"):[10.1063/5.0213660](https://doi.org/10.1063%2F5.0213660).