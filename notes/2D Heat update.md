# 2D Heat Transfer: Log Esperimenti e Aggiornamenti Codebase

## 1. Identificazione del Problema
Gli esperimenti iniziali hanno mostrato che sia la Rete Neurale standard (NN) che la Physics-Informed Neural Network (PINN) non riuscivano ad apprendere la soluzione per l'equazione del calore 2D (Laplace).
- **Causa Radice:** I modelli venivano addestrati su soli 300 punti interni campionati casualmente. Senza punti espliciti sui bordi, le reti non riuscivano a soddisfare le Condizioni al Contorno (BC), producendo soluzioni "piatte" o fisicamente inconsistenti. La PINN, in particolare, faticava a bilanciare i vincoli della PDE con l'adattamento ai dati.

## 2. Aggiornamenti Architetturali
Per aumentare la capacità rappresentativa dei modelli:
- **Dimensione Rete:** Aumentata da 3 hidden layers di 32 neuroni a **4 hidden layers di 50 neuroni** (`[2, 50, 50, 50, 50, 1]`).
- **Funzione di Attivazione:** Passaggio da `GELU` a **`Tanh`**. La tangente iperbolica è più liscia e infinitamente derivabile, rendendola ideale per le PINN dove sono richieste derivate seconde (Hessian) per la loss fisica.

## 3. Miglioramenti Strategia Dati
- **Campionamento Esplicito dei Bordi:** Modificato il processo di generazione dati per includere punti specifici sui quattro lati del dominio ($x=0, x=L_x, y=0, y=L_y$).
    - **Interni:** 1000 punti casuali.
    - **Bordi:** 50 punti per lato (200 totali).
- **Impatto:** Questa singola modifica ha permesso alla NN standard di convergere a una soluzione corretta (`Loss ~ 6e-5`).

## 4. Strategia Ottimizzazione PINN
Per risolvere i problemi di convergenza della PINN, è stato implementato un approccio multi-stage:
- **Fase Warm-up (Primo 1/3 delle epoche):**
    - La Physics Loss (`PDE`) è disabilitata (`lambda=0`).
    - **Ottimizzazione:** Il calcolo dei gradienti per il termine fisico viene saltato per velocizzare questa fase.
    - **Obiettivo:** Forzare la rete a imparare perfettamente le Condizioni al Contorno e i Dati prima di cercare di soddisfare la PDE.
- **Fase Raffinamento (Epoche rimanenti):**
    - La Physics Loss viene attivata con un peso conservativo (`lambda=0.05`) per raffinare la soluzione internamente.
- **Fine-tuning L-BFGS:**
    - Dopo il ciclo di training Adam, viene eseguito un ottimizzatore del secondo ordine (**L-BFGS**). Questo ha ridotto significativamente la loss finale (da `~1e-3` a `~3e-4`).

## 5. Refactoring del Codice
- **Gestione Dati:** `Heat2D_main.py` e `Heat2D_PINN.py` ora gestiscono "Dati Interni" e "Dati al Bordo" come dataset separati.
- **Tracciamento Loss:** Aggiornato `func/history_tracker.py` per calcolare e plottare esplicitamente la `bc_loss` (Boundary Condition Loss) separatamente dalla `data_loss` standard. Questo offre migliore visibilità sul rispetto delle BC.

## 6. Esperimenti Hardware e Precisione (MPS vs CUDA)
Abbiamo esplorato l'utilizzo dell'accelerazione **MPS (Metal Performance Shaders)** per ambienti macOS.
- **Problema:** MPS non supporta nativamente la precisione `float64` (double precision), che è standard e critica per la stabilità numerica nelle applicazioni scientifiche (PINN).
- **Tentativi:** Abbiamo provato a lavorare in `float32` sperimentando diverse funzioni di attivazione alternative per compensare la perdita di precisione, ma senza successo (la loss non convergeva a livelli accettabili).
- **Conclusione:** Abbiamo sospeso gli esperimenti su MPS e siamo tornati all'utilizzo di **CUDA** (o CPU) mantenendo la precisione `float64` per garantire risultati affidabili.
