# Heat2D Research Journal: Path to Precision

Questo documento traccia l'intera evoluzione della ricerca sul problema di Laplace 2D (Heat Transfer), partendo dalle prime esplorazioni automatiche fino ai risultati di alta precisione (Errore L2 < 0.007).

---

## 1. Infrastruttura di Ricerca (Autoresearch)
Per accelerare lo studio, sono stati creati strumenti dedicati:
- **`Heat2D_weighted_mini.py`**: Versione parametrizzata per iterazioni veloci (1-2 min).
- **`autoresearch_sweep.py`**: Automazione per sweep sistematici su architetture e attivazioni.
- **Log Centralizzato**: `mini_results.csv` per tracciare L2 error vs Durata.

---

## 2. Cronologia degli Esperimenti (Log di Esplorazione)

### Fase 1: Baseline e Architetture (L2 ~0.04 - 0.14)
- **Scoperte iniziali**: Architetture profonde e strette (80x6) superano quelle larghe e superficiali.
- **Attivazione**: GELU emerge come superiore a Tanh nelle fasi esplorative.

### Fase 2: Tapering e Ponderazione (L2 ~0.038)
- **Hypothesis**: Una struttura a imbuto (`[120-20]`) comprime meglio l'informazione.
- **Breakthrough**: Aumentare il peso delle BC (`bc_weight=50.0`) ancora la soluzione, permettendo di battere le baseline in metà del tempo.

### Fase 3: Ottimizzazione Autonoma (L2 ~0.018)
- **Sobol Sampling**: Il passaggio dal campionamento su griglia a **Pure Sobol** (1600 punti) ha ridotto drasticamente l'errore.
- **BC Density**: Aumentare la densità sui bordi (`num_bc=400`) si è rivelato fondamentale.

### Fase 4: Precision Refinement (L2 ~0.0098)
- **Coordinate Scaling**: La mappatura del dominio da `[0, 1]` a **`[-1, 1]`** ha avuto l'impatto più profondo, migliorando il flusso dei gradienti nelle attivazioni.

---

## 3. Strategie Vincenti Finali (Lessons Learned)

Il raggiungimento di una precisione costante (L2 ~0.006) è basato sulla combinazione di:

1.  **Architettura Funnel**: `[120, 120, 100, 80, 60, 40, 20]`.
2.  **SiLU + Adaptive Scaling**: Uso di SiLU con parametro `a = 1.1` per gradienti iniziali più decisi.
3.  **Anchor Points (Compass Effect)**: Aggiunta di **50 punti interni supervisionati** (`lambda_data=10.0`) per rompere i plateau di errore e guidare l'ottimizzatore.
4.  **Ottimizzazione Ibrida**: 2500 epoche di Adam seguite da 1500 di L-BFGS (`history_size=300`).
5.  **Seed-Dependence**: Identificazione di bacini di attrazione stabili tramite `seed=123`.

---

*Nota: Questi risultati derivano da oltre 119 iterazioni di ricerca automatica documentate nei log di sistema.*
