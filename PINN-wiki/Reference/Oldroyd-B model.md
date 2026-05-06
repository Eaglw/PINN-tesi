---
title: "Oldroyd-B model"
source: "https://en.wikipedia.org/wiki/Oldroyd-B_model"
author:
  - "[[Wikipedia]]"
published:
created: 2026-05-06
description:
tags:
  - "clippings"
---
The **Oldroyd-B model** is a constitutive model used to describe the flow of [viscoelastic](https://en.wikipedia.org/wiki/Viscoelastic "Viscoelastic") fluids. This model can be regarded as an extension of the [upper-convected Maxwell model](https://en.wikipedia.org/wiki/Upper-convected_Maxwell_model "Upper-convected Maxwell model") and is equivalent to a fluid filled with elastic bead and spring dumbbells. The model is named after its creator [James G. Oldroyd](https://en.wikipedia.org/wiki/James_G._Oldroyd "James G. Oldroyd").[^1]

The model can be written as: 
$$
{\displaystyle \mathbf {T} +\lambda _{1}{\stackrel {\nabla }{\mathbf {T} }}=2\eta _{0}(\mathbf {D} +\lambda _{2}{\stackrel {\nabla }{\mathbf {D} }})}
$$
 where:

- ${\displaystyle \mathbf {T} }$ is the deviatoric part of the [stress](https://en.wikipedia.org/wiki/Stress_\(physics\) "Stress (physics)") [tensor](https://en.wikipedia.org/wiki/Tensor "Tensor");
- ${\displaystyle \lambda _{1}}$ is the relaxation time;
- ${\displaystyle \lambda _{2}}$ is the [retardation time](https://en.wikipedia.org/wiki/Retardation_time "Retardation time") = ${\displaystyle {\frac {\eta _{s}}{\eta _{0}}}\lambda _{1}}$;
- ${\displaystyle {\stackrel {\nabla }{\mathbf {T} }}}$ is the [upper-convected time derivative](https://en.wikipedia.org/wiki/Upper-convected_time_derivative "Upper-convected time derivative") of stress tensor: 
	$$
	{\displaystyle {\stackrel {\nabla }{\mathbf {T} }}={\frac {\partial }{\partial t}}\mathbf {T} +\mathbf {v} \cdot \nabla \mathbf {T} -((\nabla \mathbf {v} )^{T}\cdot \mathbf {T} +\mathbf {T} \cdot (\nabla \mathbf {v} ));}
	$$
- ${\displaystyle \mathbf {v} }$ is the fluid velocity;
- ${\displaystyle \eta _{0}}$ is the total [viscosity](https://en.wikipedia.org/wiki/Viscosity "Viscosity") composed of solvent and polymer components, ${\displaystyle \eta _{0}=\eta _{s}+\eta _{p}}$;
- ${\displaystyle \mathbf {D} }$ is the deformation rate tensor or rate of [strain tensor](https://en.wikipedia.org/wiki/Infinitesimal_strain_theory "Infinitesimal strain theory"), ${\displaystyle \mathbf {D} ={\frac {1}{2}}\left[{\boldsymbol {\nabla }}\mathbf {v} +({\boldsymbol {\nabla }}\mathbf {v} )^{T}\right]}$.

The model can also be written split into polymeric (viscoelastic) part separately from the solvent part:[^2] 
$$
{\displaystyle \mathbf {T} =2\eta _{s}\mathbf {D} +\mathbf {\tau } ,}
$$
 where 
$$
{\displaystyle \mathbf {\tau } +\lambda _{1}{\stackrel {\nabla }{\mathbf {\tau } }}=2\eta _{p}\mathbf {D} }
$$

Whilst the model gives good approximations of viscoelastic fluids in [shear flow](https://en.wikipedia.org/wiki/Shear_flow "Shear flow"), it has an unphysical singularity in extensional flow, where the dumbbells are infinitely stretched. This is, however, specific to idealised flow; in the case of a cross-slot geometry the extensional flow is not ideal, so the stress, although singular, remains integrable, i.e. the stress is infinite in a correspondingly infinitely small region.[^3]

If the solvent viscosity is zero, the Oldroyd-B becomes the [upper-convected Maxwell model](https://en.wikipedia.org/wiki/Upper-convected_Maxwell_model "Upper-convected Maxwell model").

## References

[^1]: Oldroyd, James (Feb 1950). "On the Formulation of Rheological Equations of State". *Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences*. **200** (1063): 523–541. [Bibcode](https://en.wikipedia.org/wiki/Bibcode_\(identifier\) "Bibcode (identifier)"):[1950RSPSA.200..523O](https://ui.adsabs.harvard.edu/abs/1950RSPSA.200..523O). [doi](https://en.wikipedia.org/wiki/Doi_\(identifier\) "Doi (identifier)"):[10.1098/rspa.1950.0035](https://doi.org/10.1098%2Frspa.1950.0035).

[^2]: Owens, R. G.; Phillips, Timothy N. (2002). *Computational Rheology*. Imperial College Press. [ISBN](https://en.wikipedia.org/wiki/ISBN_\(identifier\) "ISBN (identifier)") [978-1-86094-186-3](https://en.wikipedia.org/wiki/Special:BookSources/978-1-86094-186-3 "Special:BookSources/978-1-86094-186-3").

[^3]: Poole, Rob (Oct 2007). "Purely elastic flow asymmetries". *Physical Review Letters*. **99** (16) 164503. [Bibcode](https://en.wikipedia.org/wiki/Bibcode_\(identifier\) "Bibcode (identifier)"):[2007PhRvL..99p4503P](https://ui.adsabs.harvard.edu/abs/2007PhRvL..99p4503P). [doi](https://en.wikipedia.org/wiki/Doi_\(identifier\) "Doi (identifier)"):[10.1103/PhysRevLett.99.164503](https://doi.org/10.1103%2FPhysRevLett.99.164503). [hdl](https://en.wikipedia.org/wiki/Hdl_\(identifier\) "Hdl (identifier)"):[10400.6/634](https://hdl.handle.net/10400.6%2F634).