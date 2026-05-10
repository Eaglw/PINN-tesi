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

## Application
Queste metriche vengono salvate automaticamente in `results.csv` e permettono di confrontare diverse architetture (es. [[Tapered_Architectures]]) in termini di bilanciamento tra accuratezza della velocità e delle componenti di stress.

## References
- [[Viscoelastic_Plotting_Updates]]: Documentation of visualization overhaul.
- Implementazione in `func/logging_utils.py`.
- Utilizzato per il benchmarking del sistema [[Viscoelastic_Fluids]].
