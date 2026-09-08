---
date: 2026-09-08T15:49:00
run_number: 22
inverse_problem: false
dataset: 4_roll_mill.csv
epochs: 22000
Computer: Kaggle
staged: false
inverse: false
status: completed
notes: "Breakthrough convergenza pressione diretta (L2 = 4.91%) con MLS scalato [-1,1], W_data=0 e 1 solo PressurePoint"
---

# Run Kaggle #22 — Direct Pressure Integration with Scaled MLS

## Info Generali
- **ID Cartella Run**: `[2026-09-08_15-49][DIR][PHASE2_MLS_SCALED][Ph2_20k+2k]`
- **Data**: 2026-09-08 15:49
- **Piattaforma**: Kaggle GPU (Tesla T4 / P100)
- **Script**: `final_roll/train_phase2_direct_comsol.py`
- **Problema**: Diretto (Fase 2 Pura: ricostruzione del campo di pressione scalare $p$ da campi noti di velocità e sforzo $\mathbf{u}, \boldsymbol{\tau}$ campionati da COMSOL)
- **Supervisione Dati Interni**: $W_{\text{data}} = 0.0$ (Zero supervisione interna di pressione)
- **Condizioni al Contorno**: $W_{\text{bc}} = 2.0$ su **1 solo nodo Dirichlet** (`PressurePoint` a coordinate adimensionali $x=1.0, y=0.1348$)

---

## Risultati e Metriche

| Fase Ottimizzazione | Budget | Errore Relativo $L_2(p)$ Iniziale | Errore Relativo $L_2(p)$ Minimo | Errore Relativo $L_2(p)$ Finale |
| :--- | :--- | :--- | :--- | :--- |
| **Adam** (FP32, `lr=1e-3`, `eps=1e-7`) | 20,000 epoche | $92.03\%$ | $19.57\%$ (epoca 3,000) | $19.82\%$ |
| **L-BFGS** (FP64, Strong Wolfe, `hist=300`) | 2,000 iterazioni | $19.82\%$ | **$4.91\%$** (iter 1,700) | **$13.52\%$** |

---

## Aspetti Tecnici e Determinanti del Successo

1. **Local Coordinate Scaling $[-1, 1]$ per MLS**:
   Le coordinate relative dei vicini $dx = x_i - x_0$ sono state scalate per la distanza massima del cluster $h = \max_i \|x_i - x_0\|_2$, mantenendo il condizionamento della matrice di Gram $X^T W X$ dell'ordine di $\mathcal{O}(1)$ ed eliminando il rumore nelle derivate spaziali.
2. **Polinomio di Grado 2 ($K=25$)**:
   L'interpolazione quadratica locale con 6 monomi previene i fenomeni di oscillazione e instabilità tipici del 3° grado (10 monomi), preservando l'irrotazionalità teorica del gradiente $-\nabla p$.
3. **Rigido Gradient Clipping**:
   `GRAD_CLIP_NORM = 5.0` protegge i buffer di Adam dai residui puntuali estremi.
4. **L-BFGS Monolitico**:
   Esecuzione con singola chiamata `optimizer.step(closure)` su 2.000 iterazioni con storico a 300 e line-search Strong Wolfe.

---

## Riferimenti Wiki
- [[MLS_Derivatives_Pressure]]
- [[Pressure_Point_Anchoring]]
- [[Viscoelastic_Training]]
- `final_roll/output_4rollmill/SUMMARY_RUNS.md`
