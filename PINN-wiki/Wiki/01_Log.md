# [2026-05-05] bootstrap | Note mio studio
- Processed 5 research notes from `Reference/Note_mio_studio/`.
- Created Topics: [[PINN_Fundamentals]], [[Activation_Functions]], [[Sampling_Strategies]].
- Created Methods: [[Tapered_Architectures]], [[Dynamic_Weighting]], [[Staged_Precision_Strategy]].
- Created Systems: [[CSTR_Modeling]], [[Heat2D_Analysis]].
- Created Literature: [[Note_01_Framework]], [[Note_02_CSTR]], [[Note_03_Heat2D]], [[Note_05_Academic_Context]].

## [2026-05-05] ingest | Klaudio_Peqini_PINNs.pdf
- Summarized PINN fundamentals presentation.
- Extracted ODE examples: Exponential Decay, Harmonic Oscillator, KdV.
- Integrated insights into [[PINN_Fundamentals]] and [[Loss_Functions]].

## [2026-05-05] ingest | Hazra_et_al_Convective_Heat_Transfer.pdf
- Summarized jet impingement cooling inverse problem research.
- Extracted methodology: DeepXDE usage, nondimensionalization, and noise robustness analysis.
- Updated [[Heat2D_Analysis]] and [[PINN_Fundamentals]].

## [2026-05-05] ingest | Sharma_et_al_Hyperparameter_Selection.pdf
- Summarized comprehensive hyperparameter optimization study for heat conduction.
- Extracted insights on activation functions (SiLU/GELU superiority), SDF for discontinuities, and quasi-random sampling.
- Updated [[Activation_Functions]], [[Sampling_Strategies]], and [[PINN_Fundamentals]].

## [2026-05-05] ingest | Thakur_et_al_ViscoelasticNet.pdf
- Summarized ViscoelasticNet framework for stress discovery.
- Extracted methodology: Multi-network architecture (Velocity/Stress/Pressure), constitutive model selection (Oldroyd-B, Giesekus, PTT), and backward Euler discretization.
- Created [[Viscoelastic_Fluids]] system page and updated [[Fluid_Dynamics]].
## [2026-05-06] ingest | Oldroyd-B model.md & Viscoelasticity.md
- Processed theoretical foundations for viscoelastic fluids.
- Extracted constitutive equations (Oldroyd-B, Maxwell, Kelvin-Voigt) and physical properties (creep, relaxation).
- Integrated into [[Viscoelasticity]] topic and [[Viscoelastic_Fluids]] system.

## [2026-05-06] ingest | PINNs_maurizio.py & Harmonic oscillator PINN.ipynb
- Summarized implementation of Damped Harmonic Oscillator PINN.
- Extracted methodology for direct and inverse problems, including parameter estimation (mu, k).
- Documented optimization techniques: L-BFGS, GELU, and lambda scheduling for PDE loss.
- Created [[Harmonic_Oscillator]] system page.
## [2026-05-06] ingest | Viscoelastic/ (Codebase)
- Documented implementation of ViscoelasticNet for Oldroyd-B channel flow.
- Extracted physical residuals (Navier-Stokes + Oldroyd-B) and stream function formulation ($\psi$).
- Logged training workflow: Adam (FP32) -> L-BFGS (FP64) with Dynamic Weighting (Learning Rate Annealing).
- Updated [[ViscoelasticNet]], [[Dynamic_Weighting]], and [[Staged_Precision_Strategy]].
