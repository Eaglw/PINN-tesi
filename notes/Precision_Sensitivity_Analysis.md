# Analisi della Sensibilità alla Precisione e Strategia Ibrida (FP32/FP64)

## 1. Obiettivo della Sperimentazione
L'obiettivo principale è stato valutare l'impatto della precisione numerica (`float32` vs `float64`) sulle prestazioni computazionali e sull'accuratezza fisica delle Physics-Informed Neural Networks (PINNs) applicate all'equazione di Laplace 2D.

## 2. Analisi Combinatoria (Benchmarking)
Abbiamo refattorizzato il codice per permettere il controllo indipendente della precisione sui seguenti componenti:
- **Net/Opt**: Pesi della rete neurale e stati dell'ottimizzatore.
- **Data**: Calcolo della loss sui dati sperimentali/analitici.
- **Physics**: Calcolo dei residui PDE (derivate automatiche).
- **BC**: Calcolo della loss sulle condizioni al contorno.

### Risultati del Benchmark (GPU 1050 Ti)
Il benchmark esaustivo su 16 combinazioni ha rivelato dati fondamentali:
- **Dominanza Computazionale**: Lo speedup è quasi esclusivamente legato alla precisione di **Rete e Ottimizzatore**. Passare da FP64 a FP32 in questi componenti garantisce uno **speedup di ~10x** (da ~500s a ~50s per 2000 epoche con rete larga).
- **Sensibilità della Fisica**: Il calcolo del residuo fisico in FP32 ha un impatto trascurabile sul tempo totale, ma può introdurre piccoli errori di troncamento nel calcolo delle derivate seconde.
- **Accuratezza**: La configurazione Full FP32 ha mostrato un aumento dell'errore MAE analitico del **~28%** rispetto al Gold Standard (Full FP64), passando da 1.9% a 2.4%.

## 3. Implementazione Strategia Ibrida (Hybrid Precision)
Per coniugare la velocità del `float32` con la precisione del `float64`, abbiamo implementato una strategia di training in due fasi:

### Fase 1: Esplorazione Globale (Adam @ FP32)
- La rete e i dati vengono castati in **precisione singola**.
- L'ottimizzatore Adam esegue il lavoro pesante (es. 40.000 epoche).
- In questa fase si ottiene il massimo vantaggio computazionale, permettendo alla rete di "capire" la forma della soluzione fisica molto rapidamente.

### Fase 2: Raffinamento Chirurgico (L-BFGS @ FP64)
- Al termine della fase Adam, il modello e tutti i dati vengono convertiti in **precisione doppia** (`model.to(torch.float64)`).
- Viene lanciato l'ottimizzatore **L-BFGS**. Essendo un metodo del secondo ordine estremamente sensibile ai gradienti, beneficia enormemente dell'alta precisione per "limare" l'errore e scendere in zone della loss non raggiungibili in FP32.

## 4. Struttura del Codice Implementata
- **`src/precision_utils.py`**: Utility per il casting dei componenti e gestione delle configurazioni.
- **`src/Heat2D_hybrid.py`**: Cuore del training ibrido, gestisce la transizione di dtype tra Adam e L-BFGS.
- **`Heat2D_main.py`**: Script per la Grid Search automatizzata che confronta diverse architetture (50, 80, 100 neuroni) usando la strategia ibrida e salvando i log in `results.csv`.

## 5. Conclusioni Metodologiche per la Tesi
Questa ricerca dimostra che l'uso indiscriminato del `float64` nelle PINN è spesso inefficiente. La strategia ibrida permette di:
1. Ridurre i tempi di ricerca degli iperparametri di un ordine di grandezza.
2. Mantenere l'accuratezza finale richiesta dal calcolo scientifico grazie al raffinamento finale in FP64.
3. Ottimizzare l'uso della memoria video (VRAM), permettendo potenzialmente l'uso di reti più grandi o dataset più densi.
