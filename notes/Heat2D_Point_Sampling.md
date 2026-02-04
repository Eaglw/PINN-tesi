# Gestione dei Punti nel Dominio (Heat2D)

In un problema PINN 2D, la distribuzione dei punti di addestramento (collocazione, dati e bordo) è critica. Sovrapposizioni o cluster di punti possono causare instabilità nel calcolo della loss e dei gradienti.

## 1. Strategia di Generazione

Per garantire una copertura uniforme del dominio ed evitare "buchi" informativi:
- **Punti Interni (Random):** Utilizzano un margine di sicurezza dai bordi per evitare conflitti con le condizioni al contorno.
- **Punti Griglia (Grid):** Generati in modo equidistante, anch'essi con un margine di sicurezza.
- **Punti di Bordo (BC):** Campionati sui quattro lati del dominio.

## 2. Prevenzione e Gestione Overlap

Per rendere il sistema robusto, sono state implementate le seguenti misure:

### Rimozione Duplicati nei Corner
La generazione dei bordi per lati indipendenti crea duplicati nei 4 angoli (dove i lati si intersecano). 
- **Soluzione:** Applicazione di `torch.unique(..., dim=0)` al set dei punti di bordo.

### Validazione tramite Distanza Euclidea
È stata introdotta la funzione `check_overlaps(points, threshold)` che:
- Calcola la matrice delle distanze tra tutti i punti usando `torch.cdist`.
- Verifica che la distanza minima sia superiore a una soglia di sicurezza (es. $10^{-7}$ o $10^{-4}$ a seconda del set).
- Fornisce un feedback immediato (`✅ No overlaps`) all'inizio di ogni esperimento.

### Consistenza Margine-Distanza
Per garantire che i punti interni non "tocchino" mai i bordi:
- Il `margin` di generazione e la distanza minima di filtraggio `d_min` sono stati sincronizzati a $10^{-4}$.

### Filtraggio a Cascata (Disgiunzione)
I punti supervisionati (Data) vengono filtrati rispetto ai punti di collocazione (Fisica) tramite la funzione `filter_and_refill`, garantendo che la rete non riceva informazioni contrastanti nello stesso punto.

## 3. Organizzazione del Codice

Le utility di campionamento sono state centralizzate per favorire il riuso tra diversi script (Main, Inverse, Reduced):
- **File:** `func/sampling_utils.py`
- **Funzioni principali:** `generate_internal_points`, `generate_grid_points`, `filter_and_refill`, `check_overlaps`.
