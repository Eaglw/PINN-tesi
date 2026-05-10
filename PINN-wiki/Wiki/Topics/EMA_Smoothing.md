# EMA Smoothing

## Overview
L'**Exponential Moving Average (EMA)** è un filtro statistico applicato alle serie temporali (in questo caso, la cronologia delle loss) per attenuare il rumore ad alta frequenza e rendere leggibili i trend di convergenza a lungo termine.

Nelle PINN, l'EMA è particolarmente utile per gestire gli **spike di loss** causati da:
1. Cambi repentini del Learning Rate (es. scheduler a gradino o plateau).
2. Reset dell'ottimizzatore.
3. Passaggio tra diverse fasi dello [[Staged_Precision_Strategy]] (es. da Adam a L-BFGS).

## Technical Implementation
La formula ricorsiva utilizzata è:
$$ EMA_t = \alpha \cdot EMA_{t-1} + (1 - \alpha) \cdot L_t $$

Dove:
- $L_t$: valore della loss all'epoca $t$.
- $\alpha \in [0, 1]$: parametro di inerzia (smoothing factor). 
  - Un valore di $\alpha = 0.95$ significa che il nuovo valore smoothed è composto al 95% dalla storia precedente e solo al 5% dall'ultimo dato.

Nel progetto, l'EMA viene calcolato dinamicamente durante la generazione dei grafici in `history_tracker.py` e sovrapposto come linea tratteggiata alla curva originale.

## Benefits
- **Leggibilità**: Permette di distinguere se il modello sta ancora convergendo o se è in stallo, anche in presenza di forti oscillazioni.
- **Diagnostica**: Aiuta a identificare se i pesi PDE/Data sono bilanciati correttamente osservando il trend della `total_loss` filtrata.

## References
- Integrato nel sistema [[Loss_History_Tracking]] (2026-05-10).
- Utilizzato per l'analisi dei risultati in [[ViscoelasticNet]].
