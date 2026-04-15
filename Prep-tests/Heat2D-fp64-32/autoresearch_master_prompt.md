# 🧠 Master Prompt: PINN Precision & Speed Research

Copia e incolla il prompt seguente per avviare una sessione di **Autoresearch** focalizzata sulla velocità e ottimizzazione multi-GPU.

---

## Prompt di Avvio

> "Agisci come un Ricercatore Senior in Calcolo Scientifico e Deep Learning. Il tuo obiettivo è ottimizzare il progetto PINN nella cartella `Heat2D-fp64-32` per massimizzare la velocità di esecuzione senza sacrificare drasticamente la precisione fisica, considerando due scenari hardware: **GTX 1050Ti (Pascal)** e **RTX 3080 (Ampere)**.
>
> ### Fasi della Ricerca:
> 1. **Baseline Speed Analysis**: Esegui `exhaustive_precision_benchmark.py` e analizza il nuovo file `speed_benchmark_results.csv`. Identifica la 'Hybrid Config' (es. Pesi FP32 e Fisica FP64) che offre il miglior compromesso in termini di Epoche al Secondo.
> 2. **Architetture Scalabili**: Testa come cambia il throughput scalando la rete da `[2, 50x4, 1]` a `[2, 256x8, 1]`. Verifica se l'uso di BF16 sulla 3080 permette di usare reti più grandi mantenendo la velocità di una rete piccola in FP64.
> 3. **L-BFGS Refinement**: Implementa un test dove le prime N epoche sono in FP32 (per velocità) e le ultime M epoche (raffinamento L-BFGS) sono in FP64. Quantifica quanto errore viene recuperato rispetto a un training FP32 puro.
> 4. **Hardware Selection**: Se rilevi una 1050Ti, forza il preset `HYBRID_FP32`. Se rilevi una 3080, abilita i Tensor Cores tramite `BF16` e `TF32`.
>
> ### Risultato Atteso:
> Crea un report `precision_strategy_report.md` che indichi esattamente quale `PrecisionConfig` usare per sessioni di 'Trial and Error' veloce e quale per la 'Final Production Run'."

---

## 🛠️ Come usare gli strumenti pronti

1. **Rilevamento Hardware**: Esegui `python Heat2D-fp64-32/src/hardware_utils.py` per vedere cosa suggerisce il sistema per la tua GPU attuale.
2. **Benchmark Completo**: Esegui `python Heat2D-fp64-32/exhaustive_precision_benchmark.py` per generare i grafici di confronto velocità/errore.
3. **Analisi Risultati**: Controlla la cartella `Heat2D-fp64-32/benchmark_plots` per le visualizzazioni.
