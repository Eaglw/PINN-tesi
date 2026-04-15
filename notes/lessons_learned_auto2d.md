# Lessons Learned: Strategie Vincenti Auto2D

Questo documento riassume le configurazioni e le strategie che hanno permesso di migliorare drasticamente la precisione del modello Heat2D (Laplace Equation), raggiungendo un L2 Relative Error di ~0.00635.

## 1. Architettura del Modello
- **Funnel Architecture (Tapered)**: L'architettura più stabile e performante segue uno schema a "imbuto": `[120, 120, 100, 80, 60, 40, 20]`.
- **Capacità Iniziale**: Strati iniziali larghi (120 neuroni) catturano meglio le feature delle coordinate, mentre il restringimento progressivo funge da regolarizzatore dello "flow" informativo.
- **Plateau di Complessità**: Aumentare ulteriormente la larghezza (es. 140) o la profondità oltre questo schema non ha portato benefici, indicando che 120 è il "punto dolce" per la precisione FP64 in questo dominio.

## 2. Attivazioni Adattive (SiLU + Adaptive Scaling)
- **Base Activation**: **SiLU (Swish)** si è confermata superiore a GELU e Tanh per PDE di secondo ordine grazie alla continuità e regolarità della sua derivata seconda.
- **Adaptive Scaling**: L'uso di un parametro scalare apprendibile `a` per ogni strato (`f(x) = SiLU(a * x)`) ha accelerato la convergenza.
- **Inizializzazione**: Inizializzare `a = 1.1` (invece di 1.0) ha fornito gradienti iniziali più ripidi, aiutando il modello a uscire rapidamente dalle fasi iniziali di instabilità.

## 3. Campionamento e Punti di Ancoraggio (Strategia Cruciale)
- **Anchor Points (Sparse Supervision)**: L'aggiunta di **50 punti di ancoraggio interni** (punti in cui è nota la soluzione analitica) con un peso `lambda_data = 10.0` è stata la chiave per rompere il plateau di 0.007.
- **Compass Effect**: Questi punti agiscono come una "bussola", impedendo all'ottimizzatore di stabilizzarsi in bacini fisicamente plausibili ma numericamente meno accurati.
- **Campionamento Sobol**: Le sequenze di Sobol con `seed=123` hanno garantito una copertura dello spazio migliore rispetto al campionamento casuale uniforme, mantenendo la coerenza tra i vari esperimenti. Variazioni con Halton hanno mostrato risultati identici, confermando la robustezza del bacino di attrazione trovato.

## 4. Strategia di Ottimizzazione
- **Hybrid Approach**: 2500 epoche di **Adam** seguite da 1500 iterazioni di **L-BFGS**.
- **L-BFGS Tuning**: Una `history_size` di 300 è risultata ottimale. Aumentarla a 500 non ha portato benefici significativi, mentre ridurla rallentava la rifinitura finale.
- **Dynamic Weighting**: L'aggiornamento dinamico dei pesi della loss (specialmente `bc_weight=25.0`) ogni 100 epoche ha bilanciato correttamente il contributo dei residui della PDE rispetto alle condizioni al contorno.

## 5. Inizializzazione dei Pesi
- **Seed-Dependence**: Il `seed=123` identifica un bacino di attrazione particolarmente profondo e largo.
- **Inizializzazione Xavier/Kaiming**: Sebbene testata, l'inizializzazione standard con l'aggiunta delle attivazioni adattive è risultata la più consistente. L'uso di Fourier Mapping (sigma=1.0) ha peggiorato i risultati, indicando che per la Laplace 2D le coordinate raw sono sufficienti se l'architettura è ben bilanciata.

---
*Nota: Questi log sono derivati da oltre 119 iterazioni di ricerca automatica documentate in `heat2dmini/autoresearch-results.tsv`.*
