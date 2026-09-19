# Pressure Point Anchoring

## Overview
Nelle Physics-Informed Neural Networks (PINNs) applicate alla fluidodinamica incomprimibile (Navier-Stokes o flussi viscoelastici), la pressione $p$ è determinata matematicamente solo a meno di una costante additiva, in quanto nelle equazioni di bilancio della quantità di moto compare unicamente sotto forma di gradiente ($\nabla p$). 

Per rendere il problema ben posto dal punto di vista numerico ed evitare che il campo di pressione fluttui o si allontani dai valori reali durante l'ottimizzazione, è necessario ancorare la pressione ad un valore noto in almeno un punto del dominio (condizione di Dirichlet puntuale). Questo punto di ancoraggio prende il nome di **Pressure Point**.

## Technical Implementation & Physical Details

Nel framework del progetto, il caricamento e la definizione del `PressurePoint` avvengono all'interno della pipeline di pre-processing dei dati (in particolare nella funzione `_extract_boundary_groups` di `utils.py`). Il codice segue due strategie in cascata:

### 2. Meccanismo di Ancoraggio Algebrico Rigoroso (Hard Anchor)
Nel modello in produzione (`CombinedModel` in `final_roll/src/train.py`), la condizione di ancoraggio al punto $\mathbf{x}_0$ **non è imposta come penalizzazione debole (soft loss)**, ma è implementata algebricamente per costruzione nella forward pass della rete:

$$ p(\mathbf{x}) = p_{\text{scale}} \left( p_{\text{raw}}(\mathbf{x}) - p_{\text{raw}}(\mathbf{x}_0) \right) + p_{\text{ref}} $$

dove:
* $\mathbf{x}_0 = \text{x\_anchor}$ è il buffer registrato con le coordinate del punto di gauge.
* $p_{\text{raw}}(\mathbf{x})$ è l'output non scalato di `model_p`.
* $p_{\text{ref}}$ è il valore di pressione prescritto al punto di riferimento (default $0.0\,\mathrm{Pa}$).
* $p_{\text{scale}}$ è il fattore di scala globale della pressione.

#### Vantaggi Rispetto alla Soft Loss:
1. **Soddisfazione Esatta $\forall \boldsymbol{\theta}$**: $p(\mathbf{x}_0) \equiv p_{\text{ref}}$ vale identicamente a qualsiasi iterazione e per qualsiasi configurazione dei pesi, azzerando la varianza di gauge.
2. **Eliminazione di Pesi Iperparametrici**: Rimuove la necessità di bilanciare un peso addizionale $W_{BC, p}$ nella loss, evitando gradient conflicts tra l'ancoraggio puntuale e il campo gradiente di Navier-Stokes $\nabla p$.

### 3. Selezione del Punto di Ancoraggio (COMSOL vs Fallback)
Il punto $\mathbf{x}_0$ viene estratto in `load_data()` (`final_roll/src/utils.py`):
1. **Da Selezione Esplicita**: se nel file mesh `.mphtxt` è presente un'etichetta `"PressurePoint"`, ne viene mappato il nodo esatto via `cKDTree`.
2. **Meccanismo di Fallback Automatico**: qualora non sia presente una selezione etichettata, il codice seleziona il primo nodo della parete esterna (`Walls`):
   * Indice nel Four-Roll Mill ($12\text{k}$): nodo sul bordo esterno con coordinate fisiche note.

### 4. Validazione Sperimentale dell'Ancoraggio Singolo
La run `[2026-09-08_15-49][DIR][PHASE2_MLS_SCALED][Ph2_20k+2k]` (Run Kaggle #22) ha dimostrato empiricamente che **1 solo PressurePoint è pienamente sufficiente a vincolare la costante di gauge e a far convergere l'intero campo di pressione 2D** fino a un errore minimo $L_2(p) = 4.91\%$, a patto che il membro destro di Navier-Stokes sia calcolato con regolarità numerica appropriata (vedi **[[MLS_Derivatives_Pressure]]**).

## References & Back-links
- [[00_Index]]
- [[COMSOL_Boundary_Extraction]]
- [[Viscoelastic_Training]]
- [[MLS_Derivatives_Pressure]]
