# Wiki Index

Welcome to the PINN-tesi LLM Wiki. This is a persistent, compounding knowledge base for research on Physics-Informed Neural Networks.

## Core Navigation
- [[01_Log|Activity Log]]
- [Literature Catalog](#literature-catalog)
- [Thematic Topics](#thematic-topics)
- [Technical Methods](#technical-methods)
- [Physical Systems](#physical-systems)

---

## Literature Catalog
- [[Note_01_Framework]]: Implementation Framework Note
- [[Note_02_CSTR]]: CSTR Research Journal
- [[Note_03_Heat2D]]: Heat2D Path to Precision
- [[Note_05_Academic_Context]]: Safa Jamali & Rheology context
- [[Klaudio_Peqini_PINNs]]: PINN fundamentals presentation
- [[Hazra_et_al_Convective_Heat_Transfer]]: Inverse heat transfer in jet cooling
- [[Sharma_et_al_Hyperparameter_Selection]]: PINN hyperparameter optimization study
- [[Thakur_et_al_ViscoelasticNet]]: Stress discovery in viscoelastic flows
- [[Oldroyd_B_Model]]: Constitutive modeling of viscoelasticity
- [[Viscoelasticity_Theory]]: Fundamental theoretical principles
- [[Maurizio_Harmonic_Oscillator]]: 1D Damped Oscillator implementation
- [[Frequency_Spectral_Bias]]: DNN learning preferences in frequency domain
- [[Upper_Convected_Maxwell]]: Simplest observer-invariant viscoelastic model
- [[Viscoelastic_Modeling_Lecture]]: Comprehensive lecture on non-Newtonian fluids
- [[Giesekus_Viscosity_Model]]: Advanced viscosity model for shear-thinning
- [[Note_04_Supervisor_Log]]: Methodological decisions and supervisor feedback
- [[Viscoelastic_Plotting_Updates]]: Overhaul of visualization and multi-field metrics
- [[Generazione_Dataset_Poiseuille]]: Poiseuille flow dataset generation for Oldroyd-B


## Thematic Topics
- [[Activation_Functions]]: Tanh, GELU, SiLU, and LAA
- [[PINN_Fundamentals]]: Core theory and loss structure
- [[Sampling_Strategies]]: Sobol, SAR, and Overlap management
- [[Loss_Functions]]: Residue components and balancing
- [[Inverse_Problems]]: Parameter identification and law discovery
- [[Nondimensionalization]]: Scaling for numerical stability
- [[Fluid_Dynamics]]: Navier-Stokes and complex flow modeling
- [[Viscoelasticity]]: Theory and modeling of time-dependent materials
- [[Pressure_Stress_Decoupling]]: Physical and mathematical decoupling of pressure and extra-stress
- [[Spectral_Bias]]: Frequency-dependent learning behavior of DNNs
- [[Creep]]: Time-dependent deformation under constant stress
- [[Stress_Relaxation]]: Stress reduction under constant strain
- [[EMA_Smoothing]]: Noise reduction in loss tracking
- [[Pressure_Scaling_Issues]]: Analysis of pressure gradient singularities and why global pressure scaling destroys momentum training
- [[Viscoelastic_Parameter_Identifiability]]: Parameter sensitivity and ill-conditioned inversion in non-dimensional Oldroyd-B models
- [[Upper-convected time derivative]]: Analytical tensor expansion of the frame-invariant convective rate of stress



## Technical Methods
- [[FCN]]: Fully Connected Network
- [[Tapered_Architectures]]: Funnel-style networks
- [[Dynamic_Weighting]]: Learning rate annealing for loss balance
- [[Cosine_Annealing_LR]]: Cosine annealing learning rate scheduling
- [[Staged_Precision_Strategy]]: Hybrid FP32/FP64 training
- [[Staged_Training_Procedure]]: Decoupled multi-phase training strategy
- [[Soft_Anti_Drift]]: Kinematic drift prevention resolving the Helmholtz-Hodge pressure inference limit
- [[Adaptive_Nondimensionalization]]: Block-wise adaptive scaling protocol decoupling numerical Reynolds from physical viscosities
- [[DeepXDE]]: Multi-backend PIML library
- [[SDF_for_Discontinuities]]: Handling sharp transitions in BCs
- [[Sobolev_Regularization]]: Derivative supervision for stream function kinematics
- [[Integral_Loss_Scaling]]: Volume-proportional loss balancing
- [[ViscoelasticNet]]: PINN framework for stress discovery
- [[ViscoelasticNet_Full model]]: Unified constitutive relation model (Oldroyd-B / Giesekus / Linear PTT)
- [[Log_Conformation_Tensor]]: Variable transformation for high Weissenberg numbers
- [[Viscoelastic_Metrics]]: Multi-field error aggregation
- [[Loss_History_Tracking]]: Convergence and gradient visualization
- [[GPU_Optimization]]: Eliminating CPU/GPU synchronization overhead
- [[VRAM_Optimization]]: Memory management and OOM prevention
- [[COMSOL_Boundary_Extraction]]: Boundary definition and naming via Explicit Selections in COMSOL
- [[Pressure_Point_Anchoring]]: Pressure anchoring mechanism and fallback strategy for incompressible flows
- [[Viscoelastic_Residual_Scaling]]: PDE residual normalization using velocity-only heuristics
- [[Vorticity_Regularization]]: Vorticity transport equation as Phase 1 regularizer to ensure conservative momentum fields
- [[Postprocessing_and_Evaluation]]: Standalone checkpoint restoration, metric evaluation, and diagnostic plot generation protocol
- [[Lasso_Regularization]]: L1 regularization for parsimonious constitutive model discovery (PTT/Giesekus pruning)
- [[Vorticity_Inversion_Solvent]]: Decoupled identification of solvent viscosity via vorticity transport to break gauge feedback loop
- [[Zero_Stress_BC_Compatibility]]: Zero-stress BCs and momentum curl compatibility in Phase 1 for Full-PIV rheometry



## Physical Systems
- [[Analisi geometria in tubo semplice]]: Identifiability limits and inverse problem breakdown in 1D channel flow
- [[CSTR_Modeling]]: Non-isothermal reactor analysis
- [[Heat2D_Analysis]]: 2D Heat Transfer (Laplace) optimization
- [[Harmonic_Oscillator]]: Benchmark system for oscillatory dynamics
- [[Viscoelastic_Fluids]]: Non-Newtonian stress discovery (Physics & Benchmark)
- [[Viscoelastic_Training]]: Viscoelastic PINN Training & Architecture (Experiment Guide)
