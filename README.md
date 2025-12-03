# PINN Tesi Project

Questo progetto si concentra sulla ricerca e l'applicazione di Physics-Informed Neural Networks (PINNs) per risolvere equazioni differenziali che modellano sistemi fisici. Supporta sia problemi diretti (trovare la soluzione) che problemi inversi (identificazione dei parametri).

## Panoramica e Aggiornamenti Recenti

Il repository è stato recentemente aggiornato per migliorare la stabilità del training, l'organizzazione del codice e l'accuratezza dei risultati. Sono state implementate strategie avanzate di ottimizzazione (come l'uso ibrido di Adam e L-BFGS) e tecniche di "Warm-up" per gestire condizioni al contorno complesse.

### Risultati CSTR Irreversibile
Il modulo `IrreversibleCSTR` è ora completo e robusto. I principali traguardi includono:
*   **Framework Sperimentale:** Implementazione di un sistema automatizzato per testare diverse configurazioni (Grid Search) su ottimizzatori (Adam, L-BFGS, Ibrido) e funzioni di attivazione (Tanh, GELU, SiLU).
*   **Problemi Diretti e Inversi:** Gli script attuali permettono di risolvere con successo sia la simulazione dell'andamento di concentrazione e temperatura, sia la stima dei parametri fisici sconosciuti a partire dai dati osservati.
*   **Ottimizzazione:** L'approccio ibrido (Adam seguito da raffinamento L-BFGS) si è dimostrato particolarmente efficace nel ridurre la loss finale e aumentare la precisione dei parametri stimati.

### Risultati Scambio Termico 2D (Heat Equation)
Il modulo `Heat2D` ha superato le iniziali difficoltà di convergenza grazie a mirati interventi architetturali e algoritmici:
*   **Campionamento Esplicito dei Bordi:** L'introduzione di punti di training specifici sui 4 lati del dominio ha risolto il problema delle soluzioni "piatte", garantendo il rispetto delle Boundary Conditions.
*   **Strategia di Warm-up:** È stata implementata una fase iniziale di training in cui la fisica è disattivata (o pesata a 0), permettendo alla rete di imparare prima la geometria della soluzione dai dati.
*   **Raffinamento L-BFGS:** L'aggiunta di una fase finale di ottimizzazione del secondo ordine ha portato la loss globale a valori estremamente bassi (**~3e-4**), rendendo la soluzione della PINN competitiva e visivamente indistinguibile da quella della NN supervisionata classica.
*   **Architettura:** Il passaggio a una rete più profonda (4 layer da 50 neuroni) e l'uso dell'attivazione `Tanh` hanno migliorato significativamente la capacità rappresentativa del modello.

## Setup & Installazione

### Virtual Environment
È altamente raccomandato l'uso di un ambiente virtuale.

**MacOS/Linux**:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows**:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Dipendenze Principali
*   `torch`: Core del framework di deep learning.
*   `numpy`, `matplotlib`: Calcolo numerico e visualizzazione.
*   `tqdm`: Barre di progresso per il monitoraggio del training.

## Struttura del Progetto

- **`Heat2D/`**: Modulo per l'equazione del calore 2D (Laplace). Include script per NN classica e PINN con strategie avanzate.
- **`IrreversibleCSTR/`**: Modulo per il reattore CSTR. Include script per problemi diretti, inversi e ottimizzazione iperparametri.
- **`func/`**: Funzioni di utilità condivise (plotting, tracking della loss).
- **`notes/`**: Documentazione tecnica e log degli esperimenti.