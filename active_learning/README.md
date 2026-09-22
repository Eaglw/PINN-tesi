# Modulo Active Learning & Bayesian Experimental Design (DoE)

Questo modulo sfrutta un **Gaussian Process Regressor** per identificare attivamente e con il minimo numero di run i **limiti di convergenza delle Physics-Informed Neural Networks (PINNs)** nella risoluzione di flussi viscoelastici nel *four-roll mill*.

## Fondamenti Metodologici

1. **Target**: Errore relativo massimo tra i parametri fisici identificati:
   $$E_{param} = \max\Big( |\text{err}_\lambda\%|,\, |\text{err}_{\eta_p}\%|,\, |\text{err}_\alpha\%|,\, |\text{err}_\epsilon\%| \Big)$$
   Modellato in scala logaritmica $z = \log_{10}(E_{param})$.
2. **Soglia Critica di Convergenza**: $\gamma = 10\%$ ($\gamma_{\log} = 1.0$).
3. **Acquisition Function di Frontiera (*Straddle Heuristic*)**:
   $$a(\mathbf{x}) = 1.96 \cdot \sigma(\mathbf{x}) - \big|\mu(\mathbf{x}) - 1.0\big|$$
   - Massimizza l'incertezza epistemica $\sigma(\mathbf{x})$ nelle zone poco esplorate.
   - Si focalizza attorno alla frontiera critica di transizione tra convergenza e divergenza ($|\mu(\mathbf{x}) - 1.0|$ minimo).
4. **Batch Diversificato (*Kriging Believer*)**:
   - Seleziona iterativamente il candidato ottimale, allucina la risposta $\hat{z} = \mu(\mathbf{x})$ e aggiorna la matrice di covarianza per forzare il punto successivo ad esplorare una regione differente e complementare dello spazio dei parametri.

## Parametri e Range di Esplorazione

| Parametro | Descrizione | Range Sensato | Note Reologiche |
|---|---|---|---|
| $\lambda$ | Relaxation time | $[0.05,\, 1.20]$ | $\lambda \ge 0.3 - 0.5$ innesca il regime HWNP |
| $\eta_p$ | Viscosità polimerica | $[0.10,\, 0.95]$ | $\eta_s = 1.0 - \eta_p$ (viscosità tot. standard = 1.0) |
| $\alpha$ | Parametro Giesekus | $[0.00,\, 0.50]$ | $\alpha > 0$ introduce shear-thinning quadratico |
| $\epsilon$ | Parametro PTT | $[0.00,\, 0.50]$ | $\epsilon > 0$ introduce rilassamento esponenziale dello stress |
| `mesh` | Risoluzione spaziale | $[5k,\, 125k]$ | Griglie standard: 5k, 12k, 29k, 52k, 88k, 125k |

## Come Eseguire lo Script

Dalla root del progetto:
```powershell
.\venv\Scripts\python active_learning/suggest_batch.py
```

Opzioni disponibili:
- `--batch-size`: numero di esperimenti consigliati (default: 3).
- `--beta`: peso dell'esplorazione epistemica $\sigma$ (default: 1.96).
- `--threshold`: soglia limite di convergenza percentuale (default: 10.0).

## Grafico Diagnostico

L'esecuzione genera automaticamente il plot in `active_learning/plots/convergence_boundary_doe.png`, visualizzando la mappa 2D degli isolivelli di errore previsti dal GP, la frontiera di transizione e i punti del batch raccomandato.
