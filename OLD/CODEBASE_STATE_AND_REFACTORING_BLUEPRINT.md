# PINN Viscoelastic Solver (4-Roll Mill)
## Stato Attuale del Codice & Guida al Refactoring

Questo documento fornisce il **quadro esatto, formale e verticale** dello stato attuale del codice PINN viscoelastico (Four-Roll Mill), focalizzandosi esclusivamente sulle componenti **attualmente in uso e funzionanti**, seguito da una sezione dedicata alle proposte di pulizia dei residui legacy per un futuro refactoring minimale.

---

# PARTE 1: STATO ATTUALE DEL CODICE (Cosa c'è e cosa funziona)

```
final_roll/
├── train_4roll_main.py       # Main controller: definisce iperparametri/flag e lancia il training
├── postprocess_run.py        # Script standalone per ricaricare checkpoint e rifare plot/report
└── src/
    ├── train.py              # Architettura CombinedModel e ciclo di training a 2 Fasi (Adam + L-BFGS)
    ├── physics.py            # Equazioni PDE (Momentum, Costitutiva), log-space params, BCs e metriche L2
    ├── utils.py              # Caricamento dati COMSOL, VRAM safe probe, plotting e TensorBoard
    └── debug.py              # Helper di diagnostica rapida su punti casuali e magnitudo termini PDE
```

---

## 1. Pipeline & Architettura Attiva

### 1.1 Struttura a Teste Separate (`CombinedModel`)
Per evitare interferenze distruttive tra i gradienti di pressione, velocità e stress durante la retropropagazione, il modello utilizza 3 reti neurali (FCN) indipendenti:
- **`model_psi`**: Rete $2 \to [128 \times 8] \to 1$ (predice la Stream Function $\psi$).
- **`model_p`**: Rete $2 \to [128 \times 8] \to 1$ (predice la pressione $p$).
- **`model_tau`**: Rete $2 \to [128 \times 8] \to 3$ (predice le componenti di extra-stress $\tau_{xx}, \tau_{xy}, \tau_{yy}$).

### 1.2 Formulazione Stream-Function ($\psi$)
La velocità $(u, v)$ è calcolata tramite differenziazione automatica di $\psi(x, y)$:
$$u = \frac{\partial \psi}{\partial y}, \qquad v = -\frac{\partial \psi}{\partial x}$$
**Proprietà fondamentale**: L'equazione di continuità per fluidi incomprimibili è soddisfatta in modo analitico esatto ($\nabla \cdot \mathbf{u} = \frac{\partial^2 \psi}{\partial x \partial y} - \frac{\partial^2 \psi}{\partial y \partial x} \equiv 0$). La loss di continuità non compare nella funzione di costo.

### 1.3 Inizializzazione & Scaling
- **Hidden Layers**: Inizializzazione Xavier Normal calibrata sull'attivazione (`SiLU` o `Tanh`).
- **Zero-Init Ultimo Layer**: L'ultimo layer lineare di `model_p` e `model_tau` viene inizializzato con pesi e bias a zero (`initialize_last_layer_zero`). All'epoca 0, le reti predicono $p=0$ e $\boldsymbol{\tau}=\mathbf{0}$, evitando shock numerici alle PDE.
- **Output Scaling**: Le uscite di `model_p` e `model_tau` sono moltiplicate rispettivamente per $p_{\text{scale}}$ e $\tau_{\text{scale}}$ per mantenere i pesi interni su scala $O(1)$.

### 1.4 Setup Hardware & Workflow Multi-Dispositivo
Il progetto opera su un'infrastruttura a ruoli ben definiti:
1. **Dispositivo di Sviluppo (macOS)**:
   - **Ruolo**: Utilizzato **esclusivamente per la scrittura del codice, refactoring, ispezione e pianificazione**.
   - **Regola Tassativa**: **NON eseguire mai training o calcoli pesanti in locale su macOS**. Il codice viene preparato qui e sincronizzato/eseguito sulle macchine dedicate.
2. **PC Principale Personale (Windows / GPU CUDA)**:
   - **Ruolo**: Macchina di calcolo primaria per l'addestramento e la generazione dei risultati principali.
   - **Script di Riferimento**: `train_4roll_main.py`.
3. **PC Maurizio all'Università (Windows / GPU CUDA)**:
   - **Ruolo**: Postazione remota/universitaria dedicata all'esecuzione di run, benchmark e test paralleli.
   - **Script di Riferimento**: `train_4roll_main_mauri.py` (da preservare intatto).
4. **Cloud / Google Colab / Kaggle (Opzionale)**:
   - **Ruolo**: Ambiente ausiliario per test leggeri o verifiche rapide su GPU cloud.

---

## 2. Formulazione Fisica, Adimensionalizzazione & Log-Space

### 2.1 Scale di Riferimento & $\eta_0$ Costante
Date le scale geometriche $H = 0.005\ \text{m}$ (raggio rulli), $H_{\text{coord}} = y_{\max} - y_{\min}$ e la velocità di riferimento $U_{\text{ref}} = \max \sqrt{u^2 + v^2}$:
- Si fissa una **scala di viscosità costante** $\eta_0 = 2.0\ \text{Pa}\cdot\text{s}$ (arbitraria e non addestrabile).
- Le scale derivate sono:
  $$p_{\text{ref}} = \tau_{\text{ref}} = \frac{\eta_0 U_{\text{ref}}}{H}, \quad Re_{\text{scale}} = \frac{\rho U_{\text{ref}} H}{\eta_0}, \quad Wi = \frac{\lambda U_{\text{ref}}}{H}$$

> [!NOTE]
> **Risoluzione del Degeneracy Loop**: Mantenere $\eta_0$ costante e non dipendente dai parametri incogniti impedisce all'ottimizzatore di ridurre artificialmente la viscosità totale a zero per far esplodere il Reynolds e minimizzare la loss senza fisica.

### 2.2 Parametrizzazione Log-Space dei Parametri Inversi
I parametri materiali incogniti sono parametrizzati in spazio logaritmico per garantire la stretta positività fisica:
$$\lambda = \lambda_{\text{guess}} \cdot \exp(r_{\lambda}), \quad \mu_p = \mu_{p,\text{guess}} \cdot \exp(r_{\mu_p}), \quad \mu_s = \mu_{s,\text{guess}} \cdot \exp(r_{\mu_s})$$
dove $r_{\lambda}, r_{\mu_p}, r_{\mu_s}$ sono i parametri `nn.Parameter` ottimizzati (inizializzati a `0.0`).

Le grandezze adimensionali e derivate sono calcolate come:
$$\mu_s^* = \frac{\mu_s}{\eta_0}, \quad \mu_p^* = \frac{\mu_p}{\eta_0}, \quad \mu_{\text{tot}} = \mu_s + \mu_p, \quad \beta = \frac{\mu_s}{\mu_{\text{tot}}}, \quad Re_{\text{phys}} = \frac{\rho U_{\text{ref}} H}{\mu_{\text{tot}}}$$

### 2.3 Equazioni Differenziali Adimensionali
1. **Momentum Equation**:
   $$f_u = Re_{\text{scale}} \left( u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} \right) + \frac{\partial p}{\partial x} - \mu_s^* \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) - \left( \frac{\partial \tau_{xx}}{\partial x} + \frac{\partial \tau_{xy}}{\partial y} \right) = 0$$
   $$f_v = Re_{\text{scale}} \left( u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} \right) + \frac{\partial p}{\partial y} - \mu_s^* \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right) - \left( \frac{\partial \tau_{xy}}{\partial x} + \frac{\partial \tau_{yy}}{\partial y} \right) = 0$$

2. **Equazione Costitutiva (Oldroyd-B / PTT / Giesekus)**:
   $$f_{\boldsymbol{\tau}} = f_{\text{PTT}} \boldsymbol{\tau}^* + Wi \stackrel{\triangledown}{\boldsymbol{\tau}^*} + \frac{\alpha Wi}{\mu_p^*} (\boldsymbol{\tau}^* \cdot \boldsymbol{\tau}^*) - 2 \mu_p^* \mathbf{D}^* = \mathbf{0}$$
   dove $\stackrel{\triangledown}{\boldsymbol{\tau}^*}$ è la derivata convettiva superiore di Oldroyd, $f_{\text{PTT}} = 1 + \frac{\epsilon Wi}{\mu_p^*} \text{tr}(\boldsymbol{\tau}^*)$, e $\mathbf{D}^*$ è il tensore di velocità di deformazione.

---

## 3. Strategia di Training a 2 Fasi Disaccoppiate (Staged Training)

Il training è suddiviso in due fasi strettamente sequenziali:

### 3.1 Fase 1: Cinematica & Reologia (Adam FP32 $\to$ L-BFGS FP64)
- **Obiettivo**: Imparare il campo di velocità $\psi$, la distribuzione topologica dello stress $\boldsymbol{\tau}$, e identificare $\lambda$ e $\mu_p$.
- **Reti & Parametri Attivi**: `model_psi`, `model_tau`, $\lambda$, $\mu_p$.
- **Reti & Parametri Congelati**: `model_p` (off), $\mu_s$ (off).
- **Equazioni Attive**: Costitutiva accesa ($w_{\text{con}}=1.0$), Momentum **spenta** ($w_{\text{mom}}=0.0$).
- **Boundary Conditions**: Velocità $u, v$ su pareti esterne e rulli + **Stress BC sui 4 rulli** (`USE_ROLL_STRESS_BC = True`, pesata 1:1 per componente con $W_{\text{roll\_stress}} = 1.0$).
- **Ottimizzazione**: Adam (FP32, ~20k epoche) $\to$ L-BFGS (FP64, ~5k iterazioni, `strong_wolfe`).

### 3.2 Transizione & Precalcolo
Al termine della Fase 1:
1. Viene eseguito il **precalcolo della divergenza dello stress** $\nabla \cdot \boldsymbol{\tau}$ tramite `precompute_stress_divergence()`, staccandola con `.detach()`.
2. Viene salvata la cache cinematica $(u_{\text{ph1}}, v_{\text{ph1}})$ come riferimento.

### 3.3 Fase 2: Idrodinamica & Pressione (Adam FP32 $\to$ L-BFGS FP64)
- **Obiettivo**: Imparare il campo di pressione $p$ e identificare la viscosità del solvente $\mu_s$.
- **Reti & Parametri Attivi**: `model_p` (LR pieno $10^{-3}$), `model_psi` (micro-LR $10^{-4}$ bilanciato da $W_{\text{data}}=35.0$), $\mu_s$.
- **Reti & Parametri Congelati**: `model_tau` (off), $\lambda$ (off), $\mu_p$ (off).
- **Equazioni Attive**: Momentum **accesa** ($w_{\text{mom}}=1.0$), Costitutiva spenta ($w_{\text{con}}=0.0$). Il termine $\nabla \cdot \boldsymbol{\tau}$ entra come tensore statico precalcolato.
- **Boundary Conditions**: $u, v$ + **PressurePoint (1 nodo Dirichlet)** per eliminare il grado di libertà di gauge della pressione.
- **Ottimizzazione**: Adam (FP32, ~15k epoche) $\to$ L-BFGS (FP64, ~5k iterazioni, `strong_wolfe`).

---

## 4. Ottimizzazioni Hardware, Calcolo VRAM & Autograd

1. **Dynamic Incremental Safe Probe (`get_optimal_chunk_size`)**:
   - Calcola il target di sicurezza pari all'**80% della VRAM GPU fisica totale**.
   - Esegue un probe a passi decrescenti ($5000 \to 2000 \to 1000$ punti) con forward/backward di prova monitorando `torch.cuda.memory_reserved()`.
   - Determina il chunk massimo ottimale senza rischiare OOM o swapping su RAM di sistema su Windows.

2. **Gradient Accumulation Chunking**:
   - Suddivide i 125k punti di collocazione in sotto-blocchi di dimensione ottimale.
   - Esegue `loss.backward()` su ogni chunk e **distrugge istantaneamente il grafo di autograd del blocco**, accumulando i gradienti senza picchi di memoria.

3. **Precalcolo Divergenza Stress ($\nabla \cdot \boldsymbol{\tau}$)**:
   - Poiché `model_tau` è congelato in Fase 2, la sua divergenza viene valutata una sola volta con `create_graph=False` e memorizzata staccata (`.detach()`), azzerando le chiamate autograd sullo stress durante tutta la Fase 2.

4. **Switch di Precisione FP32 $\leftrightarrow$ FP64**:
   - Le funzioni `convert_to_fp64` e `convert_to_fp32` in `src/utils.py` eseguono il cast ricorsivo su modello, fisica e tensori di input prima e dopo i blocchi L-BFGS.

---

### 5. Logging, Tracciamento & Visualizzazioni

1. **Console & File Logging**: Tutti i `print()` vengono intercettati e salvati in tempo reale nel file `train_log.txt` della cartella run.
2. **Server TensorBoard Live**: `launch_tensorboard_server()` avvia automaticamente `tensorboard.exe` su porta 6006 e apre il browser all'inizio dell'addestramento. Traccia:
   - Componenti di Loss: `Total`, `Data`, `BC`, `PDE`, `Momentum`, `Constitutive`.
   - Parametri Fisici: `beta`, `mu_s`, `mu_p`, `lam`, `mu_tot`, `Re_phys`.
   - Errori L2 relativi: per $u, v, p, \tau_{xx}, \tau_{xy}, \tau_{yy}$.
   - Norme dei Gradienti per sottorete: `GradNorm/Psi`, `GradNorm/Pressure`, `GradNorm/Stress`, `GradRatio/Mom_over_Data`.
3. **Pipeline Grafica Unificata (Data Provider Pattern)**:
   - `generate_all_diagnostics()` esegue un'unica passata di inferenza a chunk su GPU.
   - Genera i grafici finali:
     - `global_fields.png`: Matrice 6x5 con isolivelli COMSOL vs PINN, Errore Assoluto, Errore Relativo Classico (cutoff 5%) ed Errore Relativo Dynamics-Scaled.
     - `high_stress.png`: Scatter plot ad alta risoluzione dedicato alle sole zone ad alto sforzo ($|\boldsymbol{\tau}| > 50\%\max$).
     - `loss_history.png`, `params_evolution.png`, `l2_errors_history.png`.
4. **Post-Processing Standalone (`postprocess_run.py`)**:
   - Permette di riaprire qualsiasi run o l'ultima run completata partendo dal checkpoint `.pth` per ricalcolare tutte le metriche e rigenerare tutti i grafici senza riaddestrare.

---

# PARTE 2: RESIDUI LEGACY & PROPOSTE DI REFACTORING

## 1. Censimento dei Residui Legacy da Rimuovere

### 1.1 File Obsoleti nella cartella `final_roll/`
- `train_4roll_PressureOnly.py`: Obsoleto (incorporato nella Fase 2).
- `train_4roll_StressOnly.py`: Obsoleto (incorporato nella Fase 1).
- `train_4roll_clamp_lam.py` / `train_4roll_clamp_lam_kaggle.py`: Obsoleti (clamping manuale superato dalla log-space).

> [!IMPORTANT]
> **File da Preservare**: `train_4roll_main_mauri.py` **NON deve essere rimosso**, in quanto attivamente utilizzato per eseguire test e benchmark in parallelo su una seconda postazione/macchina.

### 1.2 Costrutti di Codice Deprecati
- **Iniezione Dinamica `globals()` / `builtins`**: Il blocco in testa al main che inietta le variabili maiuscole nei moduli `src` (`module.__dict__[name] = val`) è un residuo che crea accoppiamento implicito.
- **Logiche $\beta$-Sigmoid storiche**: Riferimenti storici a `_raw_beta` e `inverse_sigmoid`, superati dalla parametrizzazione indipendente $\mu_p, \mu_s$.

---

## 2. Proposta di Refactoring Pulito & Minimale

La struttura cartelle rimane **esattamente quella attuale** in `src/`, senza aggiungere sotto-cartelle o file YAML esterni:

```
final_roll/
├── train_4roll_main.py       # Configurazione visibile in testa + chiamata pipeline
├── postprocess_run.py        # Post-processing da checkpoint
└── src/
    ├── train.py              # CombinedModel + ciclo di training a 2 fasi
    ├── physics.py            # Equazioni differenziali, log-space params, BCs
    ├── utils.py              # Caricamento CSV/mesh, VRAM probe, plotting, TensorBoard
    └── debug.py              # Diagnostica rapida
```

### 2.1 Come strutturare `train_4roll_main.py`
Mantenere in testa a `train_4roll_main.py` la lista chiara e completa di tutti i parametri, iperparametri e flag spuntabili/modificabili, raggruppati in modo leggibile:

```python
# ============================================================================
# CONFIGURAZIONE & IPERPARAMETRI (Modificabili / Spuntabili)
# ============================================================================
STAGED_TRAINING = True
INVERSE_PROBLEM = True
DEBUG_MODE = False
USE_ROLL_STRESS_BC = True
W_ROLL_STRESS = 1.0

# Parametri Fisici Reali (Ground Truth)
MU_S_TRUE = 0.1
MU_P_TRUE = 0.9
LAM_TRUE = 0.05
RHO = 1000.0

# Scala costante e Guess Iniziali
ETA_0 = 2.0
GUESS_FACTOR = 0.80

# Architettura Neurale
HIDDEN_LAYERS = [128] * 8
ACTIVATION = nn.SiLU

# Iperparametri Training Staged
ADAM_EPOCHS_PHASE1 = 20000
USE_LBFGS_PHASE1 = True
LBFGS_MAX_ITERS_PHASE1 = 5000

ADAM_EPOCHS_PHASE2 = 15000
USE_LBFGS_PHASE2 = True
LBFGS_MAX_ITERS_PHASE2 = 5000

BASE_LR = 1e-3
W_BC = 5.0
W_PHYSICS = 3.0
W_DATA = 35.0
W_MOMENTUM = 1.0
W_CONSTITUTIVE = 1.0
```

### 2.2 Miglioria di Passaggio Parametri (Esplicito)
Nel refactoring, invece di iniettare le variabili con `module.__dict__`:
- Raggruppare i parametri definiti in testa in un semplice dizionario `config = { ... }` o passarli come argomenti con valori di default chiari a `load_data(config)`, `Physics(config)`, `train(model, physics, data, config)`.
- In questo modo:
  1. Il main rimane l'**unico punto di controllo** dove puoi modificare qualsiasi valore o flag.
  2. I file in `src/` diventano funzioni pure, leggibili, testabili e prive di variabili globali fantasma.
