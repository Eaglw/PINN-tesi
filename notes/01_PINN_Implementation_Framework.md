# PINN Implementation Framework: Theory & Techniques

Questo documento costituisce il riferimento tecnico centrale per l'implementazione delle Physics-Informed Neural Networks (PINNs) in questo progetto. Raccoglie le strategie di ottimizzazione, i paradigmi di campionamento e le analisi numeriche generali applicabili a diversi problemi fisici.

---

## 1. Architetture e Attivazioni

### 1.1 Architetture Tapered (Imbuto)
Invece di utilizzare un numero costante di neuroni per ogni strato (es. 80x6), l'architettura segue una struttura a "imbuto" (es. `[120, 100, 80, 60, 40, 20]`).
**Razionale:** Permette alla rete di apprendere feature complesse negli strati iniziali e di "condensarle" progressivamente verso l'output, riducendo il rischio di overfitting e migliorando la convergenza su problemi con domini regolari.

### 1.2 Learnable Adaptive Activations (LAA)
Le funzioni di attivazione adattive introducono parametri scalabili addestrati insieme ai pesi della rete:
$f(x) = \sigma(a \cdot x)$
Dove $\sigma$ è la funzione di attivazione e $a$ è un parametro scalare (spesso inizializzato a 1.0 o 1.1).
**Razionale:** Permette alla rete di cambiare la pendenza della funzione di attivazione localmente, aiutando a catturare gradienti ripidi o variazioni lente nel campo fisico senza aumentare eccessivamente il numero di parametri.

### 1.3 Analisi Comparativa: Tanh vs GELU
La scelta della funzione di attivazione è critica per la stabilità dei gradienti.

| Caratteristica | Tanh (Tangente Iperbolica) | GELU (Gaussian Error Linear Unit) |
| :-- | :-- | :-- |
| **Output Range** | Limitato tra **[-1, 1]**. | **Non limitato** superiormente. |
| **Derivata** | Satura a 0 per input grandi. | Non satura per input positivi. |
| **Stabilità** | "Freno naturale" in reti piccole. | Favorisce gradienti in reti profonde. |

**Nota sulla stabilità:** Tanh tende a stabilizzare reti più piccole limitando fortemente i valori (effetto "freno"), mentre GELU favorisce l'apprendimento in reti profonde grazie a un flusso di gradienti più ricco. Tuttavia, per PDE di secondo ordine, **SiLU (Swish)** si è spesso dimostrata superiore grazie alla regolarità della sua derivata seconda.

---

## 2. Strategie di Campionamento e Punti

### 2.1 Campionamento Quasi-Monte Carlo (Sobol)
A differenza del campionamento casuale uniforme, le sequenze **Sobol** sono Low-Discrepancy Sequences progettate per coprire il dominio in modo più uniforme.
**Razionale:** Riduce i "buchi" nel campionamento e previene l'addensamento casuale di punti, portando a una stima più accurata del residuo della PDE (Loss Fisica).

### 2.2 Gestione degli Overlap e Margini
Sovrapposizioni o cluster di punti possono causare instabilità.
- **Prevenzione Duplicati:** Applicazione di `torch.unique` sui punti di bordo per eliminare duplicati nei corner.
- **Validazione (Distance Check):** Calcolo della matrice delle distanze (`torch.cdist`) per verificare che la distanza minima sia superiore a una soglia di sicurezza (es. $10^{-4}$).
- **Margine di Sicurezza:** Sincronizzazione del margine di generazione dei punti interni con la distanza minima dai bordi per evitare conflitti con le BC.

### 2.3 Spatially Adaptive Refinement (SAR)
Identifica le aree del dominio dove il residuo della PDE è più alto e aggiunge dinamicamente una densità maggiore di punti in quelle zone.

---

## 3. Bilanciamento della Loss e Ponderazione

### 3.1 Learning Rate Annealing (Dynamic Weighting)
Proposta da **Wang et al. (2021)**, questa strategia mira a mitigare le *gradient pathologies* bilanciando i pesi delle diverse componenti della loss ($\lambda_{bc}, \lambda_{pde}, \lambda_{data}$).

**Logica di Aggiornamento:**
$$ \hat{\lambda}_k^{(n)} = \frac{\max_{\theta} \left| \nabla_{\theta} (\lambda_{bc} \mathcal{L}_{bc}) \right|}{\overline{\left| \nabla_{\theta} \mathcal{L}_k \right|}} $$
Si utilizza il massimo dei gradienti della BC come ancora, aggiornando gli altri pesi tramite una media mobile esponenziale per evitare oscillazioni brusche.

---

## 4. Precisione Numerica e Performance

### 4.1 Staged Precision Strategy (Hybrid FP32/FP64)
Per ottimizzare il rapporto tra velocità di convergenza e precisione fisica finale, si adotta un approccio in due fasi:

1.  **Fase 1: Esplorazione Veloce (Adam @ FP32)**
    *   Sfrutta l'accelerazione hardware (TF32 su Ampere).
    *   Speedup misurato: **10x-12x** rispetto a FP64.
2.  **Fase 2: Raffinamento Fisico (L-BFGS @ FP64)**
    *   Conversione del modello e dei dati in precisione doppia (`float64`).
    *   Garantisce precisione "scientific grade" eliminando i residui ad alta frequenza.

**Conclusioni:** L'uso indiscriminato del `float64` è inefficiente. La strategia ibrida permette di ridurre i tempi di ricerca degli iperparametri mantenendo l'accuratezza finale richiesta.
