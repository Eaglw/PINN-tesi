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

