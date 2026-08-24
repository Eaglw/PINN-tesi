# Viscoelastic Metrics

## Overview
A differenza dei problemi scalari (es. Heat2D), i sistemi viscoelastici richiedono la valutazione simultanea di 5 campi fisici interconnessi: $u, p, \tau_{xx}, \tau_{xy}, \tau_{yy}$. Le metriche aggregate sono necessarie per fornire un indice sintetico della performance del modello.

## Technical Implementation

### 1. Errori Individuali
Per ogni campo $f \in \{u, p, \tau_{xx}, \tau_{xy}, \tau_{yy}\}$, calcoliamo:
- **L2 Relative Error**: $L2_f = \frac{\|f_{pred} - f_{exact}\|_2}{\|f_{exact}\|_2}$
- **Max Relative Error**: $Max_f = \max \left( \frac{|f_{pred} - f_{exact}|}{|f_{exact}| + \epsilon} \cdot 100 \right)$

### 2. Metriche Aggregate
Introdotte nel log del [[01_Log#2026-05-10|2026-05-10]]:
- **L2_avg (Performance Globale)**: Media aritmetica degli errori L2 relativi di tutti i campi non-nulli.
  $$ L2_{avg} = \frac{1}{N_{fields}} \sum L2_f $$
  *Nota: I campi con soluzione esatta nulla (es. $\tau_{yy}$ nel Poiseuille) vengono esclusi per evitare divisioni per zero.*
- **Max_global (Affidabilità Worst-case)**: Il valore massimo assoluto tra tutti i `Max_f`. Rappresenta il punto di massima deviazione fisica dell'intero sistema.

### 3. Metriche L2 Mascherate per lo Stress ad Alti Gradienti (Masked L2 > 5%)
Nelle geometrie complesse (es. Four-Roll Mill), vaste porzioni del dominio presentano campi di extra-stress vicini allo zero, mentre gradienti estremamente ripidi e singolarità locali si concentrano negli stretti meati tra i rulli controrotanti. L'errore $L2$ relativo calcolato sull'intero dominio rischia di essere distorto da denominatori quasi nulli.

Per valutare fedelmente la qualità fisica della ricostruzione reologica nelle regioni di effettivo sforzo, viene calcolato l'**errore L2 mascherato** (`tau_masked`) isolando i nodi in cui l'ampiezza dello stress supera la soglia del 5% del valore massimo:
$$
\Omega_{\text{high}} = \left\{ \mathbf{x} \in \Omega \;\middle|\; |\tau_{\text{exact}}(\mathbf{x})| \ge 0.05 \cdot \max_{\mathbf{x}} |\tau_{\text{exact}}(\mathbf{x})| \right\}
$$
$$
L2_{\tau,\text{masked}} = \frac{\|\tau_{\text{pred}} - \tau_{\text{exact}}\|_{2, \Omega_{\text{high}}}}{\|\tau_{\text{exact}}\|_{2, \Omega_{\text{high}}}}
$$
Questo indicatore viene calcolato separatamente per $\tau_{xx}, \tau_{xy}, \tau_{yy}$ e per la prima differenza delle tensioni normali $N_1 = \tau_{xx} - \tau_{yy}$, garantendo una diagnostica quantitativa priva di artefatti numerici dovuti a zone asintoticamente a riposo.

## Application
Queste metriche vengono calcolate da `compute_l2_errors` in `final_roll/src/physics.py` e registrate sia nei log di addestramento che nei report finali di post-processing (vedi [[Postprocessing_and_Evaluation]]). Permettono un confronto rigoroso tra architetture ([[Tapered_Architectures]], SiLU vs Tanh) e tra i diversi regimi di elasticità ($Wi$).

## References
- [[Viscoelastic_Plotting_Updates]]: Documentation of visualization overhaul.
- [[Postprocessing_and_Evaluation]]: Standalone metric evaluation and diagnostic plots.
- [[Viscoelastic_Training]]: Main experiment guide for viscoelastic fluid flows.
- [[Viscoelastic_Fluids]]: Non-Newtonian stress discovery physics.
