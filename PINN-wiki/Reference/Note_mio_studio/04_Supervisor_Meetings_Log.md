# Supervisor Meetings: Log & Feedback

Questo documento raccoglie lo storico degli incontri con il relatore, tracciando domande, risposte e decisioni metodologiche prese durante il percorso di tesi.

---

## 📅 Primo Ricevimento

### Q&A e Discussioni
1.  **Rappresentatività delle ODE semplificate:** L'approccio rimane corretto, ma le difficoltà di convergenza del problema diretto sono meno evidenti nei casi semplici.
2.  **Bilancio di Energia:** Introduzione di una seconda rete per risoluzione simultanea (Coupled approach).
3.  **Caso "No Data":** Le BC sono corrette. La loss fisica (mass balance) tende a dominare inizialmente, seguita dalle BC.
4.  **Dimensione della Rete:** Deve crescere con la complessità del problema (prevenzione).
5.  **Velocità Physics-only:** L'addestramento con sola loss fisica è intrinsecamente più lento del fitting di dati sperimentali.
6.  **Campionamento:** La densità e posizione dei punti influenzano le prestazioni; necessaria analisi a posteriori.
7.  **Ottimizzatori:** Confermata la validità di LBFGS dopo un pretraining con Adam.

---

## 📅 Secondo Ricevimento (Preparazione e Note)

### Temi in Discussione
- **Coupled PINN:**
    - Gestione delle loss altissime senza pretrain sui dati.
    - Metodologie di normalizzazione di concentrazione e temperatura.
    - Impatto del bilanciamento statico vs dinamico dei pesi.
- **Problema Inverso:**
    - Utilità del lavorare sui logaritmi dei valori (`ln`).
    - Problema della normalizzazione nel caso inverso (range [0, 1]).
    - Definizione di overfitting nelle PINNs.
- **Heat2D:**
    - Sensibilità al seed iniziale e instabilità della loss.
    - Coerenza dei trend di loss tra GELU e Tanh.
