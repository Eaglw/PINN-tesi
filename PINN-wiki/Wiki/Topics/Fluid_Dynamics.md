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
- [[Bird_Stewart_Lightfoot_Transport_Phenomena]]: Foundational equations of change and momentum transport
- [[Deen_Analysis_of_Transport_Phenomena]]: Advanced asymptotic scaling, stream function, and vorticity analysis
- [[Bird_Armstrong_Hassager_Dynamics_of_Polymer_Liquids]]: Viscoelastic and non-Newtonian flow mechanics
- [[Owens_Phillips_Computational_Rheology]]: Computational methods for complex flows
- [[Oldroyd_1950_Rheological_Equations_of_State]]: Material frame indifference and Oldroyd-B constitutive model
- [[Thesis_Chapter_02_Fluid_Dynamics_Guide]]: Comprehensive study and writing guide for Thesis Chapter 2
- Practical application in jet impingement cooling: [[Hazra_et_al_Convective_Heat_Transfer]]
- Theoretical foundations: [[Viscoelastic_Modeling_Lecture]]
- Future application goals discussed in [[Klaudio_Peqini_PINNs]]
