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

## [2026-05-08] ingest | New Viscoelastic and Theory materials
- Processed: Frequency principlespectral bias.md, Upper-convected Maxwell model.md, Viscoelastic modeling.md, Viscosity model based on Giesekus equation.md,  4_Supervisor_Meetings_Log.md.
- Summarized: Frequency principle/spectral bias in DNNs, Upper-convected Maxwell (UCM) constitutive model, comprehensive viscoelastic modeling principles, and Giesekus-based viscosity models.
- Created Literature: [[Frequency_Spectral_Bias]], [[Upper_Convected_Maxwell]], [[Viscoelastic_Modeling_Lecture]], [[Giesekus_Viscosity_Model]], [[Note_04_Supervisor_Log]].
- Created Topic: [[Spectral_Bias]].
- Integrated into Topics: [[Activation_Functions]], [[Viscoelasticity]], [[Fluid_Dynamics]].

## [2026-05-10] update | Modifiche per plotting viscoelastich.md
- Overhauled the visualization pipeline for Viscoelastic Oldroyd-B simulations.
- Implemented multi-field results plotting (5 fields: $u, p, \tau_{xx}, \tau_{xy}, \tau_{yy}$) with adaptive $V_{max}$ (95th percentile).
- Enhanced `results.csv` logging with aggregated metrics: `L2_avg` (arithmetic mean of relative errors) and `Max_global` (worst-case peak error across all fields).
- Integrated [[EMA_Smoothing]] and phase markers into the [[Loss_History_Tracking]] system.
- Created Method: [[Viscoelastic_Metrics]] and Literature: [[Viscoelastic_Plotting_Updates]].
- Updated [[ViscoelasticNet]].

## [2026-05-11] update | ViscoelasticNet Goal 1 (Semi-Inverse)
- Implemented `semi_inverse` training mode specifically for Goal 1 (Phys+Data), mirroring the ViscoelasticNet methodology.
- Modified data supervision to use only $u$ and $v$ components, scaling all loss terms (PDE, BC, Data) by the variance of the reference velocity field $\sigma^2_u$.
- Introduced mini-batching ($N_{int}=256, N_{bc}=64$) and Cosine Annealing learning rate scheduling for enhanced exploration.
- Updated Staged Training logic: Phase 1 freezes $\tau$, Phase 2 trains $\tau$ and $\psi$ simultaneously.
- Documented future extension path for `inverse_dense` (full-field parameter identification) in [[ViscoelasticNet]].

## [2026-05-14] analysis | GPU Bottlenecks
- Analyzed performance bottlenecks for training small Viscoelastic PINNs on high-end GPUs (RTX 3080).
- Identified implicit CPU/GPU synchronizations (`torch.isnan(loss)` and `.item()` inside the epoch loop) and suboptimal mini-batching as primary issues.
- Created Method: [[GPU_Optimization]].

## [2026-05-16] update | Viscoelastic Boundary Conditions Deduplication
- Analyzed boundary condition generation mechanism in `Viscoelastic_physics.py`.
- Identified corner duplication at `(0,0)`, `(0,Ly)`, `(Lx,0)`, `(Lx,Ly)` causing mini-batch sampling imbalance and gradient fighting between Dirichlet and Neumann targets.
- Implemented Proposta 1 (Rigorous Geometric Slicing): Inlet owns inlet corners `[0, Ly]`, Walls own outlet corners `(0, Lx]`, Outlet strictly internal `(0, Ly)`. Total boundary points exactly match grid perimeter $2Nx + 2Ny - 4$.
- Updated [[Viscoelastic_Fluids]] with comprehensive debugging details and exact slicing logic.

## [2026-05-16] lint | Esecuzione comando LIST e fix link rotti
- Eseguito il controllo di integrità (LIST) sull'Obsidian Vault tramite lo script `scratch/check_links.py`.
- Individuato il link rotto `[[FCN]]` menzionato in `Viscoelastic_Training.md`.
- Creata la pagina mancante `Wiki/Methods/FCN.md` seguendo il Method Template con definizione architetturale, implementazione nel progetto (reti disaccoppiate in ViscoelasticNet, Harmonic Oscillator, Heat2D) e riferimenti bibliografici.
- Aggiornato l'indice `Wiki/00_Index.md` inserendo il nuovo metodo tecnico.

## [2026-05-16] update | Spiegazione Varianza vs Loss Weighting in Viscoelastic
- Chiarito il ruolo teorico e pratico della normalizzazione con la varianza ($\sigma^2$) rispetto alla ponderazione delle loss (Dynamic Weighting).
- Spiegata l'equalizzazione dimensionale e intra-loss (frazione di varianza non spiegata $1-R^2$) per i termini di confronto diretto (`data_loss`, `bc_loss`).
- Inserita la tabella dettagliata di scomposizione per ogni fase di addestramento (Fase 1, Fase 2, Fase 3 / L-BFGS) per il Goal 1 (Phys+Data) in `Wiki/Systems/Viscoelastic_Training.md`.
