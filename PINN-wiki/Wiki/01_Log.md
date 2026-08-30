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

## [2026-05-16] update | Spiegazione Architetturale Loss sui Contorni (Orchestrator vs Delegation)
- Chiarito il pattern architetturale di Strategy/Delegation che lega `compute_pinn_loss` in `func/history_tracker.py` (l'orchestratore generico) e `ViscoelasticPhysics.boundary_loss` in `Viscoelastic_physics.py` (l'implementazione fisica specializzata).
- Documentati i motivi fisici e matematici per cui il dominio viscoelastico richiede una logica dedicata al contorno (calcolo di $u,v$ da $\psi$, condizioni miste Dirichlet/Neumann, mascheramento `NaN`, filtraggio per `active_bcs` e normalizzazione della varianza).
- Aggiunta la sezione di specifica architetturale con diagramma Mermaid e snippet di interazione del codice in `Wiki/Systems/Viscoelastic_Training.md`.

## [2026-05-16] update | Analisi Performance SoloData (Neumann BCs Overhead)
- Analizzata la regressione di performance da 40 it/s a 14 it/s nella fase `SoloData` (`goal == 2`).
- Identificata la causa nel calcolo intensivo dei gradienti spaziali (`torch.autograd.grad(..., create_graph=True)`) per le 4 condizioni al contorno di Neumann attive ($p, \tau_{xx}, \tau_{xy}, \tau_{yy}$) in `ViscoelasticPhysics.boundary_loss` quando `active_bcs` è `None`.
- Documentato il meccanismo architetturale e la strategia di risoluzione (passaggio di un `active_bcs` esplicito per bypassare l'autograd) in [[Viscoelastic_Training]].

## [2026-05-18] update | Ottimizzazione VRAM per Viscoelastic PINN (OOM Prevention)
- Analizzati e risolti i frequenti errori `CUDA out of memory` durante l'addestramento dei modelli viscoelastici su GPU con VRAM limitata (es. GTX 1050 Ti 4GB).
- Ottimizzato il calcolo del Dynamic Weighting in `Viscoelastic_PINN.py`, eliminando i forward pass ridondanti sull'intero dataset e riutilizzando le componenti di loss già presenti in `loss_dict`.
- Abilitata l'allocazione avanzata PyTorch `PYTORCH_ALLOC_CONF=expandable_segments:True` in `Viscoelastic_main.py` per mitigare la frammentazione della memoria video.
- Analizzato il footprint di memoria di L-BFGS in FP64, scoprendo l'enorme impatto di `history_size=300` (~1.68 GB di VRAM allocati per ~350k parametri) e riducendolo dinamicamente a `50` per le GPU con $\le 4.5$ GB di VRAM.
- Implementata la tecnica del Chunking (Gradient Accumulation) all'interno della closure di L-BFGS, dividendo i 5000 punti di collocazione in frammenti da 500 punti per calcolare il gradiente esatto in FP64 senza saturare la VRAM.
- Ottimizzato il controllo finale della loss post L-BFGS rimpiazzando il ricalcolo full-batch con il riutilizzo dell'ultima loss valutata all'interno della closure.
- Creato il metodo tecnico [[VRAM_Optimization]] e aggiornato l'indice della wiki.

## [2026-05-18] lint | Controllo di coerenza globale e allineamento strutturale
- Eseguito un controllo completo sull'allineamento gerarchico e la coerenza dei contenuti della wiki.
- Individuate e risolte le mancanze di indicizzazione per le pagine [[Log_Conformation_Tensor]] e [[Staged_Training_Procedure]], ora correttamente aggiunte sotto i metodi tecnici in `Wiki/00_Index.md`.
- Uniformata la struttura di `Wiki/Topics/Spectral_Bias.md` aggiungendo l'header H1 principale.
- Ottimizzata la formattazione matematica in `Wiki/Systems/Viscoelastic_Fluids.md`, rimpiazzando i comandi testuali (`\text{tau}`, `\text{lambda}`, `\text{mu}_p`) con le corrette notazioni greche in LaTeX ($\boldsymbol{\tau}$, $\lambda$, $\mu_p$) per la massima chiarezza e coerenza con il resto del vault.
- Corretta un'incongruenza di numerazione nelle sezioni di `Wiki/Systems/Viscoelastic_Training.md`.
- Verificata l'integrità dei link tramite comando LIST: 100% di integrità confermata e zero broken links.

## [2026-05-18] update | Risoluzione Bug di Scaling Varianza e Ottimizzazione Training Parametri
- Analizzata la mancata convergenza delle reti neurali nei Goal 0 (PurePhys) e 1 (Phys+Data).
- Individuato un grave bug di scaling nella `boundary_loss`: a causa di `VARIANCE_EPS = 1e-8`, i campi con varianza analitica nulla nel flusso di Poiseuille ($v$ e $\tau_{yy}$) venivano divisi per `1e-8`, generando una loss sproporzionata (moltiplicata per 100 milioni) che schiacciava i gradienti di $u, p, \tau_{xx}$.
- Impostato `VARIANCE_EPS = 1.0` in `Viscoelastic_main.py` per disabilitare lo scaling aggressivo ed equalizzare i pesi al contorno.
- Modificata la strategia di addestramento inverso in `Viscoelastic_PINN.py`: i parametri fisici vengono ora mantenuti completamente congelati durante tutta la fase Adam (FP32) per evitare *gradient drift*, delegando l'identificazione di precisione esclusivamente alla fase L-BFGS (FP64) su campi neurali ormai stabili.

## [2026-05-18] update | Allineamento Architetturale Staged Training (Goal 1)
- Analizzata la divergenza di $\tau_{xy}$ e la mancata convergenza dei parametri nel Goal 1 (Phys+Data).
- Individuato un conflitto fisico intrinseco allo Staged Training in presenza di dati di velocità: con i parametri congelati, $\psi$ si ancora ai dati esatti ($\mu=0.005$) mentre $\tau$ viene forzato dalle equazioni costitutive a imparare lo stress sui parametri di guess ($\mu=0.004$). Congelando $\tau$ in Fase 2, l'errore del 20% viene cristallizzato, impedendo a L-BFGS di trovare la convergenza globale.
- Ripristinato lo sblocco mirato dei parametri per il Goal 1 in `Viscoelastic_PINN.py`: in Fase 1 si sbloccano $\mu_p$ e $\lambda$ (Reologia), in Fase 2 si sblocca $\mu_s$ (Dinamica). Grazie alla precedente correzione di `VARIANCE_EPS = 1.0`, Adam è ora stabile e in grado di far convergere dolcemente i parametri guidato dalla stabilità della rete $\psi$.

## [2026-05-19] update | GPU Training Speed Optimizations
- Implemented `use_compile` and `use_amp` toggles in `Viscoelastic_PINN.py` to accelerate training on high-end GPUs.
- Centralized hardware detection (1050 Ti vs modern GPUs) to conditionally enable `torch.compile` and tune L-BFGS memory parameters seamlessly.
- Optimized `torch.isnan` checks to run every 100 epochs instead of every epoch, eliminating severe CPU-GPU implicit synchronization overheads.
- Updated [[GPU_Optimization]] with JIT compilation and AMP considerations for PINNs.

## [2026-05-19] update | Integrazione CUDA Graphs e Stato Ottimizzazioni
- Completata con successo l'integrazione di **CUDA Graphs** per l'eliminazione dell'overhead CPU-GPU sul loop Adam.
- Corretto l'errore `cudaErrorStreamCaptureImplicit` sincronizzando il *warmup* e la *cattura* dello stream sotto lo stesso stream secondario in `CUDAGraphManager.capture`.
- Vettorizzata la `boundary_loss` rimuovendo l'allocazione dinamica di tensori come `active_mask` in `Viscoelastic_physics.py` durante la cattura del grafo.
- Risolto il `RuntimeError` del backward pass multiplo introducendo un **passo di fallback standard** per le epoche in cui sono calcolati i gradienti per la pesatura dinamica (`dynamic_weighting`) o per i log.
- Centralizzato il rilevamento dell'hardware tramite la costante `IS_1050TI`: i CUDA Graphs vengono disabilitati per la GTX 1050 Ti per scongiurare OOM in VRAM, e la stessa variabile viene riutilizzata sia per i grafi sia per il tuning di L-BFGS.
- Documentato lo stato di tutte le ottimizzazioni (attive vs scartate come `torch.compile` su Python 3.14+ e AMP per ragioni di precisione) nella pagina [[GPU_Optimization]].

## [2026-05-21] update_wiki | Aggiunta Sobolev Regularization
- Creata la pagina di metodo [[Sobolev_Regularization]] per documentare l'uso della supervisione dei gradienti (Sobolev training) come rimedio per il degradamento della velocità nel Goal 2 (`SoloData`) su addestramenti lunghi.
- Aggiornato l'indice generale [[00_Index]] inserendo il nuovo metodo tra i metodi tecnici.

## [2026-05-25] update_wiki | Inserimento disaccoppiamento Pressione-Stress (ViscoelasticNet)
- Creata la pagina Topic [[Pressure_Stress_Decoupling]] per documentare l'aspetto fisico e matematico della scomposizione del tensore degli sforzi di Cauchy ($\boldsymbol{\sigma} = -p\mathbf{I} + \mathbf{T}$) e del disaccoppiamento tra la pressione idrostatica (moltiplicatore di Lagrange per l'incomprimibilità) e l'extra-stress polimerico $\boldsymbol{\tau}$ nelle equazioni costitutive.
- Documentato come questo disaccoppiamento venga tradotto in ViscoelasticNet a livello di architettura neurale (output e loss costitutiva separati) e come avvenga il loro accoppiamento tramite la conservazione della quantità di moto.
- Aggiornate le pagine [[00_Index]], [[Viscoelasticity]], [[Viscoelastic_Fluids]] e [[ViscoelasticNet]] per integrare e linkare il nuovo argomento.

## [2026-05-25] update_wiki | Inserimento dettagli su splittaggio Dirichlet/Neumann BC e staged masking
- Aggiornata la pagina [[Viscoelastic_Training]] documentando la separazione matematica e computazionale tra le condizioni al contorno di Dirichlet e Neumann.
- Descritto il meccanismo di mascheramento basato su `NaN` e il funzionamento del filtraggio dinamico con `active_bcs` durante le fasi dello Staged Training per ottimizzare l'esecuzione del codice ed evitare calcoli autograd superflui.

## [2026-05-26] ingest | Generazione Dataset Flusso di Oldroyd-B (Poiseuille)
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

## [2026-05-16] update | Spiegazione Architetturale Loss sui Contorni (Orchestrator vs Delegation)
- Chiarito il pattern architetturale di Strategy/Delegation che lega `compute_pinn_loss` in `func/history_tracker.py` (l'orchestratore generico) e `ViscoelasticPhysics.boundary_loss` in `Viscoelastic_physics.py` (l'implementazione fisica specializzata).
- Documentati i motivi fisici e matematici per cui il dominio viscoelastico richiede una logica dedicata al contorno (calcolo di $u,v$ da $\psi$, condizioni miste Dirichlet/Neumann, mascheramento `NaN`, filtraggio per `active_bcs` e normalizzazione della varianza).
- Aggiunta la sezione di specifica architetturale con diagramma Mermaid e snippet di interazione del codice in `Wiki/Systems/Viscoelastic_Training.md`.

## [2026-05-16] update | Analisi Performance SoloData (Neumann BCs Overhead)
- Analizzata la regressione di performance da 40 it/s a 14 it/s nella fase `SoloData` (`goal == 2`).
- Identificata la causa nel calcolo intensivo dei gradienti spaziali (`torch.autograd.grad(..., create_graph=True)`) per le 4 condizioni al contorno di Neumann attive ($p, \tau_{xx}, \tau_{xy}, \tau_{yy}$) in `ViscoelasticPhysics.boundary_loss` quando `active_bcs` è `None`.
- Documentato il meccanismo architetturale e la strategia di risoluzione (passaggio di un `active_bcs` esplicito per bypassare l'autograd) in [[Viscoelastic_Training]].

## [2026-05-18] update | Ottimizzazione VRAM per Viscoelastic PINN (OOM Prevention)
- Analizzati e risolti i frequenti errori `CUDA out of memory` durante l'addestramento dei modelli viscoelastici su GPU con VRAM limitata (es. GTX 1050 Ti 4GB).
- Ottimizzato il calcolo del Dynamic Weighting in `Viscoelastic_PINN.py`, eliminando i forward pass ridondanti sull'intero dataset e riutilizzando le componenti di loss già presenti in `loss_dict`.
- Abilitata l'allocazione avanzata PyTorch `PYTORCH_ALLOC_CONF=expandable_segments:True` in `Viscoelastic_main.py` per mitigare la frammentazione della memoria video.
- Analizzato il footprint di memoria di L-BFGS in FP64, scoprendo l'enorme impatto di `history_size=300` (~1.68 GB di VRAM allocati per ~350k parametri) e riducendolo dinamicamente a `50` per le GPU con $\le 4.5$ GB di VRAM.
- Implementata la tecnica del Chunking (Gradient Accumulation) all'interno della closure di L-BFGS, dividendo i 5000 punti di collocazione in frammenti da 500 punti per calcolare il gradiente esatto in FP64 senza saturare la VRAM.
- Ottimizzato il controllo finale della loss post L-BFGS rimpiazzando il ricalcolo full-batch con il riutilizzo dell'ultima loss valutata all'interno della closure.
- Creato il metodo tecnico [[VRAM_Optimization]] e aggiornato l'indice della wiki.

## [2026-05-18] lint | Controllo di coerenza globale e allineamento strutturale
- Eseguito un controllo completo sull'allineamento gerarchico e la coerenza dei contenuti della wiki.
- Individuate e risolte le mancanze di indicizzazione per le pagine [[Log_Conformation_Tensor]] e [[Staged_Training_Procedure]], ora correttamente aggiunte sotto i metodi tecnici in `Wiki/00_Index.md`.
- Uniformata la struttura di `Wiki/Topics/Spectral_Bias.md` aggiungendo l'header H1 principale.
- Ottimizzata la formattazione matematica in `Wiki/Systems/Viscoelastic_Fluids.md`, rimpiazzando i comandi testuali (`\text{tau}`, `\text{lambda}`, `\text{mu}_p`) con le corrette notazioni greche in LaTeX ($\boldsymbol{\tau}$, $\lambda$, $\mu_p$) per la massima chiarezza e coerenza con il resto del vault.
- Corretta un'incongruenza di numerazione nelle sezioni di `Wiki/Systems/Viscoelastic_Training.md`.
- Verificata l'integrità dei link tramite comando LIST: 100% di integrità confermata e zero broken links.

## [2026-05-18] update | Risoluzione Bug di Scaling Varianza e Ottimizzazione Training Parametri
- Analizzata la mancata convergenza delle reti neurali nei Goal 0 (PurePhys) e 1 (Phys+Data).
- Individuato un grave bug di scaling nella `boundary_loss`: a causa di `VARIANCE_EPS = 1e-8`, i campi con varianza analitica nulla nel flusso di Poiseuille ($v$ e $\tau_{yy}$) venivano divisi per `1e-8`, generando una loss sproporzionata (moltiplicata per 100 milioni) che schiacciava i gradienti di $u, p, \tau_{xx}$.
- Impostato `VARIANCE_EPS = 1.0` in `Viscoelastic_main.py` per disabilitare lo scaling aggressivo ed equalizzare i pesi al contorno.
- Modificata la strategia di addestramento inverso in `Viscoelastic_PINN.py`: i parametri fisici vengono ora mantenuti completamente congelati durante tutta la fase Adam (FP32) per evitare *gradient drift*, delegando l'identificazione di precisione esclusivamente alla fase L-BFGS (FP64) su campi neurali ormai stabili.

## [2026-05-18] update | Allineamento Architetturale Staged Training (Goal 1)
- Analizzata la divergenza di $\tau_{xy}$ e la mancata convergenza dei parametri nel Goal 1 (Phys+Data).
- Individuato un conflitto fisico intrinseco allo Staged Training in presenza di dati di velocità: con i parametri congelati, $\psi$ si ancora ai dati esatti ($\mu=0.005$) mentre $\tau$ viene forzato dalle equazioni costitutive a imparare lo stress sui parametri di guess ($\mu=0.004$). Congelando $\tau$ in Fase 2, l'errore del 20% viene cristallizzato, impedendo a L-BFGS di trovare la convergenza globale.
- Ripristinato lo sblocco mirato dei parametri per il Goal 1 in `Viscoelastic_PINN.py`: in Fase 1 si sbloccano $\mu_p$ e $\lambda$ (Reologia), in Fase 2 si sblocca $\mu_s$ (Dinamica). Grazie alla precedente correzione di `VARIANCE_EPS = 1.0`, Adam è ora stabile e in grado di far convergere dolcemente i parametri guidato dalla stabilità della rete $\psi$.

## [2026-05-19] update | GPU Training Speed Optimizations
- Implemented `use_compile` and `use_amp` toggles in `Viscoelastic_PINN.py` to accelerate training on high-end GPUs.
- Centralized hardware detection (1050 Ti vs modern GPUs) to conditionally enable `torch.compile` and tune L-BFGS memory parameters seamlessly.
- Optimized `torch.isnan` checks to run every 100 epochs instead of every epoch, eliminating severe CPU-GPU implicit synchronization overheads.
- Updated [[GPU_Optimization]] with JIT compilation and AMP considerations for PINNs.

## [2026-05-19] update | Integrazione CUDA Graphs e Stato Ottimizzazioni
- Completata con successo l'integrazione di **CUDA Graphs** per l'eliminazione dell'overhead CPU-GPU sul loop Adam.
- Corretto l'errore `cudaErrorStreamCaptureImplicit` sincronizzando il *warmup* e la *cattura* dello stream sotto lo stesso stream secondario in `CUDAGraphManager.capture`.
- Vettorizzata la `boundary_loss` rimuovendo l'allocazione dinamica di tensori come `active_mask` in `Viscoelastic_physics.py` durante la cattura del grafo.
- Risolto il `RuntimeError` del backward pass multiplo introducendo un **passo di fallback standard** per le epoche in cui sono calcolati i gradienti per la pesatura dinamica (`dynamic_weighting`) o per i log.
- Centralizzato il rilevamento dell'hardware tramite la costante `IS_1050TI`: i CUDA Graphs vengono disabilitati per la GTX 1050 Ti per scongiurare OOM in VRAM, e la stessa variabile viene riutilizzata sia per i grafi sia per il tuning di L-BFGS.
- Documentato lo stato di tutte le ottimizzazioni (attive vs scartate come `torch.compile` su Python 3.14+ e AMP per ragioni di precisione) nella pagina [[GPU_Optimization]].

## [2026-05-21] update_wiki | Aggiunta Sobolev Regularization
- Creata la pagina di metodo [[Sobolev_Regularization]] per documentare l'uso della supervisione dei gradienti (Sobolev training) come rimedio per il degradamento della velocità nel Goal 2 (`SoloData`) su addestramenti lunghi.
- Aggiornato l'indice generale [[00_Index]] inserendo il nuovo metodo tra i metodi tecnici.

## [2026-05-25] update_wiki | Inserimento disaccoppiamento Pressione-Stress (ViscoelasticNet)
- Creata la pagina Topic [[Pressure_Stress_Decoupling]] per documentare l'aspetto fisico e matematico della scomposizione del tensore degli sforzi di Cauchy ($\boldsymbol{\sigma} = -p\mathbf{I} + \mathbf{T}$) e del disaccoppiamento tra la pressione idrostatica (moltiplicatore di Lagrange per l'incomprimibilità) e l'extra-stress polimerico $\boldsymbol{\tau}$ nelle equazioni costitutive.
- Documentato come questo disaccoppiamento venga tradotto in ViscoelasticNet a livello di architettura neurale (output e loss costitutiva separati) e come avvenga il loro accoppiamento tramite la conservazione della quantità di moto.
- Aggiornate le pagine [[00_Index]], [[Viscoelasticity]], [[Viscoelastic_Fluids]] e [[ViscoelasticNet]] per integrare e linkare il nuovo argomento.

## [2026-05-25] update_wiki | Inserimento dettagli su splittaggio Dirichlet/Neumann BC e staged masking
- Aggiornata la pagina [[Viscoelastic_Training]] documentando la separazione matematica e computazionale tra le condizioni al contorno di Dirichlet e Neumann.
- Descritto il meccanismo di mascheramento basato su `NaN` e il funzionamento del filtraggio dinamico con `active_bcs` durante le fasi dello Staged Training per ottimizzare l'esecuzione del codice ed evitare calcoli autograd superflui.

## [2026-05-26] ingest | Generazione Dataset Flusso di Oldroyd-B (Poiseuille)
- Processed the document teorico sulla generazione del dataset sintetico per il flusso di Poiseuille di Oldroyd-B.
- Estratti i profili analitici di velocità, sforzi e funzione di corrente, e documentate le metodologie di campionamento (Grid e Sobol) con rumore Gaussiano.
- Creata la pagina Literature [[Generazione_Dataset_Poiseuille]].
- Aggiornato l'indice [[00_Index]] e la pagina di sistema [[Viscoelastic_Fluids]].

## [2026-05-26] update_wiki | Aggiunta ViscoelasticNet Unified Model
- Creata la pagina di metodo [[ViscoelasticNet_Full model]] per documentare l'equazione costitutiva unificata 2D (che unisce Oldroyd-B, Giesekus e Linear PTT).
- Documentata l'estensione del sistema di monitoraggio, logging CSV ed evoluzione dei parametri fisici su 5 subplots per tracciare la convergenza di $\epsilon$ e $\alpha$ in modalità inversa.
- Documentate le ottimizzazioni computazionali (fattorizzazione algebrica dei coefficienti e calcolo selettivo di Navier-Stokes in Fase 1).
- Aggiornato l'indice generale [[00_Index]].

## [2026-06-01] update_wiki | Aggiunta procedura estrazione boundary COMSOL
- Creata la pagina di metodo [[COMSOL_Boundary_Extraction]] per documentare la procedura di definizione e assegnazione dei nomi ai vari boundary (inlet, outlet, walls) in COMSOL tramite Selezioni Esplicite (nodo *Explicit* sotto *Definitions*).
- Aggiornato l'indice generale [[00_Index]].

## [2026-06-01] update_wiki | Inserimento analisi dimensionale e ottimizzazione di Schwarz
- Aggiornata la pagina Topic [[Nondimensionalization]] con l'analisi dimensionale dettagliata e il riscalamento viscoso (Viscous Scaling) delle equazioni di Navier-Stokes e del modello costitutivo PTT-Giesekus.
- Aggiornata la pagina Method [[VRAM_Optimization]] inserendo l'ottimizzazione basata sul Teorema di Schwarz per il calcolo di $v_{yy} = -u_{yx}$, riducendo il branching del grafo di autograd per le derivate di terzo ordine.
- Aggiornata la pagina Method [[ViscoelasticNet_Full model]] per referenziare l'analisi dimensionale e descrivere l'ottimizzazione del calcolo delle derivate.

## [2026-06-16] update_wiki | Inserimento Cosine Annealing LR
- Creata la pagina di metodo [[Cosine_Annealing_LR]] per documentare l'apprendimento non lineare cosinusoidale del learning rate.
- Documentate la formula chiusa da paper SGDR e la formula ricorsiva ottimizzata in PyTorch, evidenziando il comportamento senza restart.
- Aggiornato l'indice generale [[00_Index]].

## [2026-06-29] update_wiki | Aggiunta normalizzazione coerente residui viscoelastici
- Creata la pagina di metodo [[Viscoelastic_Residual_Scaling]] per documentare l'incoerenza strutturale delle scale delle derivate spaziali tra la Momentum e l'equazione Costitutiva.
- Documentato l'effetto dell'operatore differenziale sui gradienti ripidi (es. rulli) e l'impatto sul bilancio dei pesi della loss.
- Presentato l'approccio euristico basato sulla sola velocità per stimare `tau_scale` e `momentum_scale` (strain rate massimo) nel caso in cui non siano disponibili i dataset di stress/pressione.
- Aggiornato l'indice generale [[00_Index]] e la pagina Topic [[Nondimensionalization]] per referenziare il nuovo metodo.

## [2026-07-01] update_wiki | Aggiunta documentazione su problematiche scaling pressione
- Creata la pagina Topic [[Pressure_Scaling_Issues]] per documentare l'aspetto fisico e matematico del perché la Momentum non può essere scalata tramite p_scale (fenomeno del Gradient Starvation).
- Chiarito il ruolo del `momentum_scale` basato su `tau_scale * shear_max` come corretta normalizzazione fisica per portare la Momentum loss a O(1) senza distruggere i gradienti della velocità.
- Aggiornato l'indice generale [[00_Index]] inserendo il nuovo topic.


## [2026-07-02] update_wiki | Analisi convergenza pressione e ottimizzazioni Staged Training
- Aggiornata la pagina Topic [[Pressure_Stress_Decoupling]] introducendo il **Limite teorico di Helmholtz-Hodge per l'inferenza di pressione**. Spiegato perché il rumore numerico amplificato nelle derivate di alto ordine delle velocità e degli stress congelati crei una componente rotazionale incompatibile che impedisce alla pressione di convergere in assenza di co-adattamento della velocità.
- Aggiornata la pagina Method [[Staged_Training_Procedure]] con i dettagli della Fase 2: Dynamics, documentando il fenomeno del **Vanishing Gradient Cascade** dovuto alla zero-inizializzazione dell'ultimo layer di `model_p`, e l'ottimizzazione basata sul **Precalcolo della divergenza dello stress ($\nabla \cdot \boldsymbol{\tau}$)** che riduce la VRAM e velocizza la Fase 2 di circa il 25%-30%.

## [2026-07-02] update_wiki | Integrazione doppia scala di adimensionalizzazione per raggio roll
- Aggiornata la pagina Topic [[Nondimensionalization]] descrivendo la **Double Length-Scale Adimensionalization** (Doppia scala di adimensionalizzazione).
- Spiegato in dettaglio come si mantiene la compatibilità al 100% con i vecchi checkpoint riscalando le coordinate nel dataset a $H_{\text{coord}} = 0.05\text{ m}$ (range $[0, 1]$), mentre i parametri adimensionali ($Re, Wi$) e i riferimenti fisici vengono calcolati sul raggio del rullo ($H_{\text{ref}} = 0.005\text{ m}$).
- Documentato matematicamente come il fattore 10 di scala (rapporto $H_{\text{coord}}/H_{\text{ref}}$) si compensi esattamente nel calcolo delle velocità a causa dell'accoppiamento tra stream function e derivate spaziali, evitando spike di loss all'avvio.

## [2026-07-08] update_wiki | Aggiunta documentazione sul meccanismo di Pressure Point Anchoring
- Creata la pagina Method [[Pressure_Point_Anchoring]] per documentare la necessità numerica dell'ancoraggio Dirichlet per la pressione in fluidodinamica incomprimibile (Navier-Stokes/viscoelastici).
- Documentati i due meccanismi di estrazione: tramite selezione esplicita da file mesh COMSOL (`.mphtxt`) o tramite il fallback automatico sul primo nodo del gruppo `Walls` (es. indice `111092`, coordinate `[1.0000, 0.1348]`, corrispondente a `[0.025, -0.01826]` metri sulla parete verticale destra).
- Aggiornato l'indice generale [[00_Index]].



## [2026-07-09] update_wiki | Documentazione del limite rotazionale di Helmholtz-Hodge e della Vorticity Regularization

### Contesto Sperimentale
- Esperimento sul 4-roll mill con Staged Training (Fase 1: psi+tau senza momentum, Fase 2: solo pressione con velocity congelata).
- L'errore L2 della pressione si stabilizzava a **154%** e la loss di momentum si bloccava a **0.933** indipendentemente dal numero di epoche.
- Verificato sperimentalmente che la correlazione Pearson tra $\nabla^2 u$ (Autograd su PINN) e $\nabla^2 u$ (MLS su COMSOL) era **-0.375** (anticorrelazione), dimostrando che le derivate seconde della velocity apprese senza momentum sono fisicamente incoerenti.
- Ricerca bruteforce su tutte le 16 combinazioni di segni dei termini di momentum ha confermato che il residuo medio minimo (~0.42) non è eliminabile con alcuna combinazione di segni.

### Pagine Create
- **[[Vorticity_Regularization]]** (Methods): Nuova pagina atomica che documenta:
  - Il problema della componente rotazionale nel termine noto $\mathbf{F}$ della momentum equation quando la velocity è congelata.
  - La soluzione tramite l'equazione di trasporto della vorticità $Re\,(\mathbf{u} \cdot \nabla \omega) = \beta\,\nabla^2 \omega + \nabla \times (\nabla \cdot \boldsymbol{\tau})$ come regolarizzatore in Fase 1.
  - Considerazioni implementative (costo computazionale delle derivate quarte, warmup, pesi della loss).
  - Tabella comparativa con dati sperimentali del 4-roll mill.

### Pagine Modificate
- **[[Pressure_Stress_Decoupling]]** (Topics): Aggiunta sottosezione "Alternative Resolution: Vorticity Regularization in Phase 1" con wikilink alla nuova pagina. Aggiunto backlink a [[Vorticity_Regularization]] nei riferimenti.
- **[[00_Index]]**: Aggiunta voce [[Vorticity_Regularization]] nella sezione Technical Methods.



## [2026-07-21] update_wiki | Aggiunta analisi dell'identificabilità dei parametri viscoelastici

### Contesto e Teoria
- Analizzata la differenza di convergenza tra $\lambda$ (tempo di rilassamento) ed $\eta_p$ (viscosità polimerica) nel problema inverso.
- Spiegato il motivo per cui l'equazione costitutiva non-lineare di Oldroyd-B vede $\eta_p$ esclusivamente tramite il parametro adimensionale $\beta_{poly} = \frac{\eta_p}{\eta_s + \eta_p}$.
- Dimostrato che una variazione minima dello sforzo predetto $\mathbf{\tau}$ (~3.7% di errore L2) corrisponde a una pendenza piatta $\frac{d\beta_{poly}}{d\eta_p} \approx 0.039$, provocando un'amplificazione dell'errore su $\eta_p$ fino al 65.6% ($1.49$ vs $0.90$).

### Pagine Create
- **[[Viscoelastic_Parameter_Identifiability]]** (Topics): Creata pagina atomica con l'analisi matematica completa, l'analisi delle derivate parziali e le tre contromisure (LR differenziati, riattivazione Momentum in Fase 2 e ottimizzazione di secondo ordine L-BFGS).

### Pagine Modificate
- **[[Inverse_Problems]]** (Topics): Aggiunto riferimento e wikilink a [[Viscoelastic_Parameter_Identifiability]].
- **[[00_Index]]**: Aggiunta voce [[Viscoelastic_Parameter_Identifiability]] nella sezione Thematic Topics.



## [2026-08-08] update_wiki | Post-Processing Script & Resilient Evaluation Protocol

### Modifiche e Soluzioni nel Codice
- **Rinomina Script**: Rinominato `final_roll/train_for_roll_main_mauri.py` in `final_roll/train_4roll_main_mauri.py` rispettando la convenzione del progetto.
- **Risoluzione NameError**: Risolto il bug `NameError: name 'builtins' is not defined` in `src/train.py` (`plot_params`) e `src/physics.py` introducendo un recupero sicuro dei parametri di riferimento ($\eta_s, \eta_p, \lambda, \epsilon, \alpha, \beta$) tramite `builtins` e fallback sui moduli globali.
- **Nuovo Script `postprocess_run.py`**: Creata la suite autonoma di post-processing in `final_roll/postprocess_run.py`. Supporta auto-detection dell'ultima run in `output_4rollmill/`, ripristino di `model_state_dict`, `physics_state_dict` e `history_state_dict`, calcolo metriche L2 e generazione automatica dei plot diagnostici nella sottocartella `postprocess_plots/`.

### Pagine Create
- **[[Postprocessing_and_Evaluation]]** (Methods): Nuova pagina tecnica atomica che descrive il protocollo di post-processing, risoluzione dei checkpoint, gestione fail-safe del contesto e generazione della diagnostica dei campi.

### Pagine Modificate
- **[[00_Index]]**: Aggiunta voce [[Postprocessing_and_Evaluation]] nella sezione Technical Methods.



## [2026-08-18] update_wiki | Lasso Regularization for Parsimonious Viscoelastic Discovery

### Contesto e Teoria
- Documentata la formulazione della penalizzazione $L_1$ (Lasso) per la selezione e discovery autonoma dei modelli costitutivi (PTT / Giesekus / Oldroyd-B).
- Spiegata la differenza fondamentale tra contrazione $L_2$ (Ridge) e sparsità esatta $L_1$ (forza costante di restore verso zero).

### Pagine Create
- **[[Lasso_Regularization]]** (Methods): Nuova pagina metodologica con formulazione matematica, confronto $L_1$ vs $L_2$ e workflow di integrazione nel framework PINN.

### Pagine Modificate
- **[[00_Index]]**: Aggiunta voce [[Lasso_Regularization]] nella sezione Technical Methods.


## [2026-08-21] update_wiki | Full-Blind Inverse PINN Paradigm, Offline SVD Identifiability & Helmholtz-Hodge Resolution

### Contesto e Teoria
- **Autopsia Fallimento Run 010**: Spiegato il motivo per cui legare $Re = \frac{\rho U H}{\eta_{\text{tot}}}$ a una viscosità totale trainabile $\eta_{\text{tot}}$ creava una degenerazione numerica ($\eta_{\text{tot}} \downarrow \implies Re \uparrow \implies \beta \to 0, \eta_{\text{tot}} \to 0.027\ \text{Pa}\cdot\text{s}, L_2(p) \approx 258\%$).
- **Nuovo Paradigma di Adimensionalizzazione Decoupled**: Separata la scala numerica $\eta_0$ ($Re_{\text{scale}} = \frac{\rho U H}{\eta_0}$) dalle viscosità fisiche primarie indipendenti $\eta_s > 0, \eta_p > 0$ in coordinate logaritmiche ($e^r$). I valori composti $\eta_{\text{tot}}, \beta, Re_{\text{phys}}$ sono calcolati esclusivamente a posteriori.
- **Risoluzione Limite Helmholtz-Hodge**: Dimostrato che l'hard freeze di $\psi$ in Fase 2 amplifica il rumore sulle derivate 2ᵉ e 3ᵉ, creando una componente solenoidale non integrabile in $\nabla p$. Risolto mediante sblocco di $\psi$ a basso LR ($10^{-4}$) vincolato dalla loss **Soft Anti-Drift** $\mathcal{L}_{\text{drift}}$.
- **Studi di Identificabilità Offline (COMSOL High-Fidelity)**:
  - Condizionamento SVD su legge costitutiva: $\sigma_1 = 385.1, \sigma_2 = 286.5 \implies \kappa(J_{\text{con}}) = 1.34$. Stima least-squares recupera $\lambda$ allo $0.10\%$ ed $\eta_p$ allo $0.01\%$.
  - Isolamento di $\eta_s$: Il Direct Momentum su dati COMSOL raggiunge correlazione $0.8929$ ed errore su $\eta_s$ dell'$1.03\%$, mentre la formulazione a rotore (Curl) collassa al $95\%$ di errore a causa dell'amplificazione del rumore nelle differenze finite ($1/\Delta x^2 \approx 1250$).
  - Tabella di robustezza al rumore sintetico (0% - 2%): La reologia $(\lambda, \eta_p)$ rimane stabile anche con $1\%$ di rumore ($\approx 7\%$ err), mentre il collasso di $\eta_s$ a differenze finite dimostra il vantaggio fondamentale della PINN (rappresentazione liscia globale e derivate esatte tramite Autograd).
- **Deprecazione Fase 3**: Rimossa la fase di unfreezing globale congiunto, confermando la pipeline rigorosa a 2 Fasi (ciascuna con ciclo Adam FP32 + L-BFGS FP64).

### Pagine Create
- **[[Soft_Anti_Drift]]** (Methods): Metodo di regolarizzazione cinematica $\mathcal{L}_{\text{drift}}$ per risolvere il limite di Helmholtz-Hodge in Fase 2.
- **[[Adaptive_Nondimensionalization]]** (Methods): Protocollo di aggiornamento a blocchi EMA ($K=2000$ epoche, $\alpha=0.1$, clamping $[0.5, 2.0]$) per la scala numerica $\eta_0$ con gradiente detached.

### Pagine Modificate
- **[[Viscoelastic_Parameter_Identifiability]]** (Topics): Riscritta e aggiornata con analisi SVD, tabella di robustezza al rumore, confronto Direct Momentum vs Curl e parametrizzazione log-space.
- **[[Staged_Training_Procedure]]** (Methods): Aggiornata all'architettura a 2 Fasi, deprecando formalmente la Fase 3 congiunta.
- **[[Nondimensionalization]]** (Topics): Integrata la distinzione tra formulazione forward e inverse decoupled scaling.
- **[[Pressure_Stress_Decoupling]]** (Topics): Integrata la risoluzione del limite Helmholtz-Hodge via Soft Anti-Drift.
- **[[Viscoelastic_Training]]** (Systems): Aggiornata l'autopsia del Run 010, le metriche di monitoraggio diagnostico e la roadmap dei test sperimentali (Multi-start, Scale sweep, Noise-aware PINN, Ablation study).
- **[[00_Index]]**: Registrate le nuove pagine metodologiche.

## [2026-08-21] update_wiki | Comprehensive Wiki Audit, Index Alignment & Technical Enhancements

### Azioni di Manutenzione e Bonifica Eseguite
- **Scansione Completa del Vault**: Eseguito audit sistematico su tutti i 67 file markdown della Wiki (`Wiki/` e sottocartelle `Literature/`, `Methods/`, `Systems/`, `Topics/`).
- **Verifica e Indicizzazione 100%**: Aggiunte le pagine mancanti in `Wiki/00_Index.md`:
  - `[[Analisi geometria in tubo semplice]]` sotto la sezione `## Physical Systems`.
  - `[[Upper-convected time derivative]]` sotto la sezione `## Thematic Topics`.
  - Confermato che tutte le 65 pagine della Wiki sono ora perfettamente catalogate e linkate nell'indice.
- **Risoluzione Incongruenze e Deprecazione Fase 3**:
  - In `Wiki/Methods/ViscoelasticNet.md` e `Wiki/Methods/Cosine_Annealing_LR.md`, rimossi i residui storici che descrivevano una Fase 3 attiva, uniformando la documentazione al protocollo consolidato a **2 Fasi disaccoppiate** (Fase 1: Cinematica & Reologia; Fase 2: Idrodinamica & Pressione) con cicli Adam FP32 + L-BFGS FP64.
  - In `Wiki/Methods/ViscoelasticNet_Full model.md`, chiarita la log-parametrizzazione non vincolata ($\lambda = \lambda_{\text{guess}} e^{r_\lambda}$, $\eta_p = \eta_{p,\text{guess}} e^{r_p}$, $\eta_s = \eta_{s,\text{guess}} e^{r_s}$) in sostituzione di `torch.abs` e clamping in-place.
  - In `Wiki/Methods/Sobolev_Regularization.md`, allineata la convenzione dei target alla politica semi-inversa di progetto (supervisione interna pura sui soli campi di velocità $u_{\text{obs}}, v_{\text{obs}}$, senza mai passare dati interni di stress o $\psi$).
- **Integrazione "Buchi" Tecnici e Ottimizzazioni**:
  - In `Wiki/Methods/GPU_Optimization.md`: Documentata la precomputazione statica della divergenza dello extra-stress (`precompute_stress_divergence`) in Fase 2 per dimezzare i tempi computazionali di autograd.
  - In `Wiki/Methods/VRAM_Optimization.md`: Integrata la documentazione sull'algoritmo di autotuning dinamico della dimensione dei blocchi (`get_optimal_chunk_size`).
  - In `Wiki/Methods/Viscoelastic_Metrics.md`: Formalizzata la definizione dell'errore relativo $L2$ mascherato al 5% per la valutazione accurata dello stress in regioni ad alto gradiente / singolarità geometriche.
- **Link Integrity Linting**: Eseguito script di verifica: **0 link rotti** e **100% integrità confermata** su tutto il grafo della Wiki.

## [2026-08-29] update_wiki | Formulazione a Rotore per eta_s, Analisi Gauge Feedback Loop Pressione-Solvente e Protocollo Warmup Fase 2

### Analisi Teorica e Diagnostica Sperimentale
- **Autopsia Drift Monotono di $\eta_s$**: Spiegato analiticamente il fenomeno per cui $\eta_s$ (solvente) attraversa transitoriamente il valore reale ($0.100\text{ Pa}\cdot\text{s}$) intorno a 3.500 epoche di Fase 2 Adam, per poi accumulare una deriva crescente ininterrotta ($0.138$ ad epoca 40k, esplosione $> 1.8$ in L-BFGS non vincolato).
- **Dimostrazione della Gauge Degeneracy Pressione-Solvente**:
  - Nel bilancio $\mu_s^* \nabla^2 \mathbf{u} + \nabla\cdot\boldsymbol{\tau} - \nabla p = \mathbf{0}$, la pressione ha solo 1 punto Dirichlet al bordo ($p(x_0,y_0)=0$) che fissa la costante $+C$ ma lascia libera la scala di ampiezza di $\nabla p$.
  - All'inizio di Fase 2, `model_p` parte da zero. Man mano che impara, $|\nabla p|$ cresce e trascina con sé $\mu_s^*$ tramite la condizione stazionaria $\mu_s^* \approx \frac{\int (\nabla p - \nabla\cdot\boldsymbol{\tau})\cdot\nabla^2 \mathbf{u}}{\int \|\nabla^2 \mathbf{u}\|^2}$.
  - Verificato empiricamente sui tensori: a fine Adam Fase 2, $\mu_s$ è sovrastimato del $+38\%$ e la deviazione standard (ampiezza) di $\nabla p$ è sovrastimata esattamente del $+40\%$, confermando l'inflazione simbiotica non fisica.
- **Formulazione a Rotore / Vorticità per l'Inversione Decoupled di $\eta_s$**:
  - Applicando il rotore ($\nabla \times$) all'equazione di quantità di moto, $\nabla \times \nabla p \equiv \mathbf{0}$, la pressione scompare del tutto.
  - L'equazione di trasporto della vorticità $\mu_s^* \nabla^2 \omega_z = -\nabla \times (\nabla\cdot\boldsymbol{\tau}) + Re \dots$ contiene $\mu_s^*$ come **unica incognita**, generando un problema quadratico unidimensionale strettamente convesso con minimo globale analitico unico.

### Pagine Create
- **[[Vorticity_Inversion_Solvent]]** (Methods): Documentazione completa della formulazione a vorticità/rotore per isolare $\eta_s$, analisi del gauge feedback loop e specifica del protocollo a sotto-fasi (Warmup Fase 2A/2B/2C).

### Pagine Modificate
- **[[00_Index]]**: Registrata la pagina metodologica [[Vorticity_Inversion_Solvent]].

## [2026-08-30] run | Record Storico Assoluto Fase 2: Warmup 5k + L-BFGS 500
- **Run target**: `[INV][STAGED][Ph2_10k+0.5k_Warmup5k][2026-08-29_19-21]`
- **Configurazione**: Checkpoint Mauri (40k+10k) + Fase 2 (10k Adam con 5k Warmup pressione + 500 L-BFGS FP64 con $\mu_s$ trainable).
- **Traguardi raggiunti**:
  - **Pressione $L_2(p)$**: Crollata al **$25.40\%$** ($0.253999$), minimo assoluto di sempre per la pressione (precedente best era $60.9\%$).
  - **Viscosità Solvente $\mu_s$**: Identificata a **$0.1148\text{ Pa}\cdot\text{s}$** (true: $0.1000\text{ Pa}\cdot\text{s}$, $+14.8\%$). L'esplosione incontrollata ($> 1.83$) è stata completamente sradicata!
  - **Viscosità Totale $\mu_{\text{tot}}$**: **$1.0197\text{ Pa}\cdot\text{s}$** (true: $1.0000\text{ Pa}\cdot\text{s}$, errore **$+1.96\%$**).
  - **Reologia**: $\mu_p = 0.9049\text{ Pa}\cdot\text{s}$ ($+0.54\%$), $\lambda = 0.05020\text{ s}$ ($+0.40\%$).
  - **Cinematica $L_2(u,v)$**: $L_2(u) = 0.257\%$, $L_2(v) = 0.252\%$ (precisione scientifica $< 0.26\%$).
  - **Sforzi $L_2(\tau)$**: $L_2(\tau_{xx}) = 0.767\%$, $L_2(\tau_{xy}) = 0.677\%$, $L_2(\tau_{yy}) = 0.698\%$.
- Aggiornato `SUMMARY_RUNS.md` con il nuovo record storico.



## [2026-08-30] update_wiki | Dimostrazione Invarianza di Scala (Gauge Freedom) in Fase 1 e Mappatura Hardcap FP32 per eta_0

### Analisi Teorica, Bug Fix e Diagnostica Sperimentale
- **Dimostrazione Analitica Invarianza di Gauge**:
  - Dimostrato che con output dello stress normalizzato $\boldsymbol{\tau}^* = \mathbf{N}_\tau \cdot \tau_{\text{scale}}$, il residuo costitutivo diviso per $\tau_{\text{scale}}$ contiene il termine di viscosità $2 \left(\frac{\eta_p}{\eta_0 \cdot \tau_{\text{scale}}}\right) \mathbf{D}^*$.
  - Poiché $\tau_{\text{scale}} = \max |\boldsymbol{\tau}^*| = \frac{\tau_{d,\max}}{\eta_0 U_{\text{ref}}/H_{\text{ref}}}$, il prodotto $\eta_0 \cdot \tau_{\text{scale}} = \frac{\tau_{d,\max}}{U_{\text{ref}}/H_{\text{ref}}}$ è una costante fisica invariante rispetto alla scelta arbitraria di $\eta_0$.
  - Di conseguenza, sia il target della rete $\mathbf{N}_\tau \in [-1, 1]$, sia le derivate dell'equazione costitutiva e i gradienti di loss rispetto ai parametri dimensionali $(\mu_p, \lambda)$ risultano analiticamente identici per ogni $\eta_0$.
- **Risoluzione Bug Clamping Asimmetrico**:
  - Individuato in `load_data()` (`src/utils.py`) il clamping difensivo errato `tau_scale = max(float(max_tau_nd), 1.0)`.
  - Poiché per il 4-roll mill $\tau_{d,\max} \approx 4.09\text{ Pa}$ e $U_{\text{ref}}/H_{\text{ref}} \approx 1.667\text{ s}^{-1}$, per $\eta_0 > 2.45\text{ Pa}\cdot\text{s}$ il valore di $\max |\boldsymbol{\tau}^*|$ scendeva sotto $1.0$ e $\tau_{\text{scale}}$ veniva bloccato ad $1.0000$, rompendo la cancellazione di $\eta_0$ e indebolendo la loss costitutiva di un fattore $(2.45/\eta_0)^2$.
  - Corretto con salvaguardia infinitesima: `tau_scale = max(float(max_tau_nd), 1e-6)` e `p_scale = max(float(max_p_nd), 1e-6)`.
- **Suite di Benchmark Sperimentale ed Estensione Hardcap**:
  - Creati script dedicati: `train_4roll_suite.py` (senza alterare `train_4roll_main.py`) e orchestratore `run_suite_eta0.py`.
  - **Check Analitico a Step 0**: Verificato su $\eta_0 \in [0.05, 10.0]\text{ Pa}\cdot\text{s}$ che loss e gradienti coincidono a livello di singolo bit ULP ($10^{-16}$ o entro fluttuazione $1.19 \times 10^{-7}$).
  - **Suite di Training a 2500 Epoche Adam**: Eseguiti run su $\eta_0 \in [0.5, 1.0, 2.0, 5.0]$.
    - Per $\eta_0 \in [0.5, 1.0, 2.0]$: convergenza dei parametri dimensionali identica fino alla sedicesima cifra decimale ($\mu_p = 0.7466121912002563\text{ Pa}\cdot\text{s}$, $\lambda = 0.04028252139687538\text{ s}$).
    - Per $\eta_0 = 5.0$: convergenza coerente allo stesso bacino d'attrazione con scarto $\le 0.36\%$ ($\mu_p = 0.7439\text{ Pa}\cdot\text{s}$).
  - **Esplorazione Hardcap (500 Epoche Adam)**:
    - A salire ($\eta_0 = 3.0$): la transizione cinematica è ritardata di $\sim 150$ epoche per via della scala di stress ridotta, ma converge verso lo stesso asintoto.
    - A scendere ($\eta_0 = 0.20$): comportamento analogo e simmetrico dovuto all'amplificazione di scala ($\tau_{\text{scale}} = 12.27$).
  - **Classificazione dei Limiti Operativi FP32**:
    1. **Core Invariante Bit-for-Bit**: $\eta_0 \in [0.50, 2.00]\text{ Pa}\cdot\text{s}$ (precisione esatta $10^{-16}$).
    2. **Bacino di Convergenza Asintotica**: $\eta_0 \in [0.20, 5.00]\text{ Pa}\cdot\text{s}$ (precisione FP32 $\Delta \le 0.36\%$).
    3. **Hardcap Numerico da Evitare in FP32**: $\eta_0 < 0.10$ e $\eta_0 > 5.00\text{ Pa}\cdot\text{s}$ (rischio di gradient explosion o quantization underflow).

### Pagine Modificate
- **[[Adaptive_Nondimensionalization]]** (Methods): Aggiunta la sezione completa sull'invarianza esatta di gauge in Fase 1, la risoluzione del bug di clamping asimmetrico e la tabella dei regimi operativi e hardcap FP32.
- **[[Viscoelastic_Training]]** (Systems): Aggiornato il Test 3 della roadmap di validazione sperimentale (completato e convalidato per la Fase 1).

### Integrazione Baseline a 500 Epoche e Verifica Bit-Perfect
- Completato il run a 500 epoche su $\eta_0 = 1.00\text{ Pa}\cdot\text{s}$ a parità di scheduler ($T_{\max} = 500$).
- **Risultato di Coincidenza Bit-Perfect**:
  - Tra $\eta_0 = 1.00$ e $\eta_0 = 2.00$, $\mu_p = 0.6963310838\text{ Pa}\cdot\text{s}$ e $\lambda = 0.0391521752\text{ s}$ sono **identici al 100% bit-for-bit** (16 cifre decimali, deviazione $0.000000\%$).
  - Tra $\eta_0 = 0.10$ e $\eta_0 = 0.20$, $\mu_p = 0.6962456703\text{ Pa}\cdot\text{s}$ e $\lambda = 0.0391401388\text{ s}$ sono identici fino alla 10ª cifra decimale.
- **Accuratezza alla Seconda Cifra Decimale**:
  - Su tutta la finestra $\eta_0 \in [0.10, 3.00]\text{ Pa}\cdot\text{s}$, i parametri convergono rigorosamente a $\mu_p = \mathbf{0.70}\text{ Pa}\cdot\text{s}$ e $\lambda = \mathbf{0.04}\text{ s}$ (e fino alla 3ª cifra: $\mu_p = \mathbf{0.696}\text{ Pa}\cdot\text{s}$, deviazione massima $\le 0.03\%$).
  - Confermato che le curve di convergenza in Fase 1 sono matematicamente e operativamente equivalenti su tutto il dominio di scale analizzato.
