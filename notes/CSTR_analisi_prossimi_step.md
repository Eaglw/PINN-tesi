### **Piano di Modifica per Sperimentazione Sistematica**

**Obiettivo:** Eseguire in modo automatico esperimenti incrociati (grid search) per diverse combinazioni di ottimizzatori (Adam, LBFGS, etc.) e funzioni di attivazione (Tanh, GELU, SiLU, etc.), salvando i risultati in modo organizzato.

**1. Centralizzazione della Configurazione (in `IrreversibleCSTR_main.py`)**

*   **Azione:** Invece di avere parametri sparsi, creare in cima al file `IrreversibleCSTR_main.py` una lista di "esperimenti". Ogni esperimento sarà un dizionario (o un oggetto) che definisce una combinazione da testare.
*   **Esempio di struttura:**
    ```python
    experiments_to_run = [
        {'name': 'Adam_Tanh', 'optimizer': 'Adam', 'activation': 'Tanh', 'learning_rate': 1e-3, 'epochs': 20000},
        {'name': 'Adam_GELU', 'optimizer': 'Adam', 'activation': 'GELU', 'learning_rate': 1e-3, 'epochs': 20000},
        {'name': 'LBFGS_Tanh', 'optimizer': 'LBFGS', 'activation': 'Tanh', 'learning_rate': 1.0, 'epochs': 500},
        # ... altre combinazioni
    ]
    ```

**2. Rendere il Modello Flessibile (in `IrreversibleCSTR_nn_pinn.py`)**

*   **Azione:** Modificare il costruttore (`__init__`) della classe `PINN` per accettare un parametro stringa che identifichi la funzione di attivazione (es. `'Tanh'`, `'GELU'`).
*   **Logica Interna:** All'interno del costruttore, una semplice struttura `if/elif/else` o un dizionario mapperà la stringa all'oggetto funzione di attivazione di PyTorch (es. se riceve `'GELU'`, userà `torch.nn.GELU()`). In questo modo, il modello non avrà più una funzione di attivazione "hardcoded".

**3. Generalizzare la Creazione dell'Ottimizzatore (in `IrreversibleCSTR_main.py`)**

*   **Azione:** Creare una funzione helper o un blocco `if/elif/else` che istanzi l'ottimizzatore corretto in base alla stringa presente nella configurazione dell'esperimento (es. `'Adam'` o `'LBFGS'`).
*   **Dettaglio:** Questo è un passaggio cruciale perché l'ottimizzatore LBFGS richiede un approccio diverso per il training (l'uso di una `closure`) rispetto ad Adam. La logica di training dovrà quindi adattarsi a quale ottimizzatore è stato scelto.

**4. Creare un Ciclo di Esecuzione Principale (in `IrreversibleCSTR_main.py`)**

*   **Azione:** Invece di eseguire un singolo script, si dovrà creare un ciclo `for` che itera sulla lista `experiments_to_run` definita al punto 1.
*   **Logica del Ciclo:** Per ogni "esperimento" nella lista:
    1.  Estrae i parametri (nome, ottimizzatore, funzione di attivazione, lr, etc.).
    2.  Crea l'istanza del modello `PINN`, passando la funzione di attivazione scelta.
    3.  Crea l'istanza dell'ottimizzatore scelto.
    4.  Esegue il ciclo di training (adattandosi alla logica richiesta da Adam o LBFGS).
    5.  Salva i risultati (grafici, metriche, modello) usando un path specifico per quell'esperimento.

**5. Organizzare il Salvataggio dei Risultati**

*   **Azione:** Modificare le funzioni di salvataggio dei grafici e dei dati (probabilmente dentro `IrreversibleCSTR_main.py` o in `func/graphic_func.py`) per accettare un "nome esperimento".
*   **Logica:** Tutti gli output di un esperimento (es. `Adam_Tanh`) verranno salvati in una sottocartella dedicata, ad esempio `plots/CSTR/Adam_Tanh/`. Questo eviterà che i risultati di esperimenti successivi sovrascrivano quelli precedenti e renderà l'analisi comparativa molto più semplice.