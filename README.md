# PINN Viscoelastic Fluid Solver

Questo repository è dedicato alla ricerca e all'applicazione di **Physics-Informed Neural Networks (PINNs)** per lo studio e la modellazione di **fluidi viscoelastici**. Utilizzando una formulazione adimensionale, il framework permette di risolvere problemi diretti (calcolo dei campi di velocità, pressione e sforzo) e problemi inversi (stima dei parametri reologici a partire da dati sperimentali/COMSOL).

## Modelli Fisici e Formulazione

Il solutore implementa le equazioni di Navier-Stokes per flussi incomprimibili accoppiate con equazioni costitutive viscoelastiche in forma adimensionale. 

### 1. Cinematica (Stream Function)
Per garantire l'incompressibilità ($\nabla \cdot \mathbf{u} = 0$), il modello predice la funzione di corrente $\psi$ al posto dei campi di velocità $u$ e $v$ diretti:
$$u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x}$$

### 2. Equazioni del Moto (Momento)
La conservazione del momento lineare include il contributo della pressione $p$, la viscosità del solvente $\beta$ (viscosità adimensionale) e la divergenza del tensore extra-sforzo polimerico $\boldsymbol{\tau}$:
$$\operatorname{Re} (\mathbf{u} \cdot \nabla \mathbf{u}) + \nabla p - \beta \nabla^2 \mathbf{u} - \nabla \cdot \boldsymbol{\tau} = 0$$

### 3. Equazioni Costitutive
Il tensore extra-sforzo $\boldsymbol{\tau} = (\tau_{xx}, \tau_{xy}, \tau_{yy})$ viene ricavato risolvendo le equazioni costitutive che supportano i seguenti modelli:
- **Oldroyd-B**
- **PTT (Phan-Thien-Tanner)** (tramite il parametro $\epsilon$)
- **Giesekus** (tramite il parametro $\alpha$)

Le equazioni costitutive sono regolate dai numeri adimensionali di **Weissenberg ($Wi$)** e dal rapporto delle viscosità ($\beta_p$).

---

## Architettura e Strategia di Training (Staged Training)

Per garantire la convergenza ottimale su equazioni differenziali non lineari altamente accoppiate (come quelle viscoelastiche), il repository adotta il framework **ViscoelasticNet**:

1. **Rete Multi-Testa**: Tre reti distinte (`FCN`) predicono rispettivamente la stream-function $\psi$, la pressione $p$, e i tre componenti dello sforzo $\tau$.
2. **Staged Training**:
   - **Fase 1 (Adam @ FP32)**: La pressione $p$ è congelata; il modello ottimizza solo la stream function $\psi$ e il tensore extra-sforzo $\tau$.
   - **Fase 2 (Adam @ FP32)**: La pressione $p$ viene sbloccata e ottimizzata congiuntamente a $\psi$.
   - **Fase 3 (L-BFGS @ FP64)**: Raffinamento finale di tutte le reti a precisione doppia (`float64`) per garantire accuratezza fisica microscopica.
3. **Identificazione dei Parametri (Problema Inverso)**: È possibile attivare la stima automatica dei parametri incogniti del fluido ($\mu_s$, $\mu_p$, $\lambda$, $\epsilon$, $\alpha$).

---

## Struttura del Progetto

- **`Viscoelastic/`**:
  - `Viscoelastic_main.py`: Script principale per l'esecuzione di grid search ed esperimenti fisici.
  - `results.csv`: File di log in cui vengono salvati i risultati di ciascuna configurazione di training.
  - `src/`:
    - `Viscoelastic_physics.py`: Definizione matematica dei residui PDE e delle condizioni al contorno (BC) con meccanismo di controllo e alert dei gruppi di mesh.
    - `models.py`: Definizione delle architetture MLP (`FCN`) e del modello coordinatore.
    - `config.py`: Gestione delle configurazioni (`TrainingConfig`), learning rate schedulers e freezing/unfreezing dei parametri.
    - `load_comsol.py`: Utility per caricare, pulire e adimensionalizzare i dati provenienti da esportazioni COMSOL.
    - `trainer.py`: Cicli di ottimizzazione customizzati per gestire la transizione Adam $\rightarrow$ L-BFGS e il passaggio di precisione.
- **`COMSOL/`**: Cartella destinata a ospitare i dataset di riferimento (es. `Oldroyd.csv`).
- **`func/`**: Funzioni condivise per il tracciamento della loss (`history_tracker.py`) e visualizzazioni 2D (`graphic_func.py`).

---

## Setup & Installazione (Windows)

Si raccomanda l'uso di un ambiente virtuale dedicato.

1. **Creazione ambiente virtuale**:
   ```powershell
   python -m venv venv
   ```

2. **Attivazione dell'ambiente**:
   ```powershell
   .\venv\Scripts\activate
   ```

3. **Installazione dipendenze**:
   ```powershell
   .\venv\Scripts\pip install -r requirements.txt
   ```

## Esecuzione dei Test

Per lanciare la suite di training e grid-search sul dataset viscoelastico:
```powershell
.\venv\Scripts:python Viscoelastic/Viscoelastic_main.py
```

I risultati, le metriche aggregate ed i grafici delle comparazioni (confronto campi di velocità, pressione e sforzo rispetto a COMSOL) saranno generati all'interno della cartella `Viscoelastic/experiments_weighted/`.