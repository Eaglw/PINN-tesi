# Topic: Fluid Dynamics

Fluid dynamics studies the flow of liquids and gases, often governed by the Navier-Stokes equations.

## PINN Applications
PINNs are particularly effective in fluid dynamics for:
- **Flow Reconstruction**: Recovering velocity and pressure fields from flow visualization data.
- **Turbulence Modeling**: Estimating closure terms in RANS or LES models.
- **Complex Geometries**: Mesh-free nature allows for analysis of flows in complex industrial setups (e.g., jet impingement in [[Hazra_et_al_Convective_Heat_Transfer]]).
- **Non-Newtonian Rheology**: PINNs can learn viscosity models (e.g., Power-law, Carreau, Giesekus) directly from data or residuals.

## Key Systems
- **Navier-Stokes**: The fundamental governing equations for Newtonian fluids.
- **Viscoelastic Flows**: Governed by the coupling of Navier-Stokes and constitutive equations (e.g., [[ViscoelasticNet]]).
- **Magnetohydrodynamics (MHD)**: Flow of electrically conducting fluids.

## References
- Future application goals discussed in [[Klaudio_Peqini_PINNs]].
- Practical application in jet impingement cooling: [[Hazra_et_al_Convective_Heat_Transfer]].
- Theoretical foundations: [[Viscoelastic_Modeling_Lecture]].
