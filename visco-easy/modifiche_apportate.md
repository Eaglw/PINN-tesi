# Riepilogo Modifiche Apportate - visco-easy/train_4rollmill.py

Questo documento riassume i miglioramenti e il refactoring implementati nello script `visco-easy/train_4rollmill.py` per ottimizzare l'accuratezza, la riproducibilità e la tracciabilità degli esperimenti sulla PINN viscoelastica.

---

## 1. Gestione Output e Log Training

### Configurazione Output Dinamica
L'output dello script non viene più sovrascritto ad ogni esecuzione. Ora viene generata automaticamente una sottocartella specifica basata sull'architettura, le epoche, l'attivazione, lo staged training, il tipo di problema (diretto/inverso) e un timestamp di esecuzione:
* **Formato**: `4_roll_mill_L{Layers}_E{Epochs}_{Activation}_staged{Staged}_inv{Inverse}_{Timestamp}`
* **Percorso**: `visco-easy/output_4rollmill/...`

### Salvataggio Log in file `train_log.txt`
Abbiamo ridefinito la funzione `print` all'interno dello script per intercettare l'output standard e salvarlo contemporaneamente in formato testuale nel file `train_log.txt` all'interno della cartella di output del caso.

### Frequenza di Monitoraggio Dinamica
La frequenza di stampa a terminale (e logging) per l'errore L2 e i residui PDE è stata impostata dinamicamente in base alle epoche totali per stampare circa **4-5 volte** durante il training complessivo:
* `PRINT_EVERY = max(1, ADAM_EPOCHS // 5)`

---

## 2. Refactoring Fisico e Scale di Riferimento

### Scala di Velocità Robusta (`U_ref`)
La scala di velocità di riferimento non è più definita sul massimo valore di una singola componente di velocità. Viene ora calcolata in modo robusto basandosi sul modulo del campo di velocità complessivo (modulo quadratico medio massimo):
* **Formula**: `U_ref = max(np.sqrt(u_raw**2 + v_raw**2))`
Questa scala migliorata viene propagata correttamente per derivare le quantità adimensionali coerenti (`p_ref`, `tau_ref`, Reynolds `Re` e Weissenberg `Wi`).

---

## 3. Unificazione delle Loss (Weighted MSE)

### API Unificata `weighted_mse`
Abbiamo eliminato l'utilizzo di `nn.MSELoss()(pred, target) / var` e centralizzato il calcolo delle loss normalizzate tramite la varianza in un'unica funzione esplicita:
```python
def weighted_mse(pred, target, var):
    return torch.mean(((pred - target) ** 2) / var)
```
Questo garantisce che tutte le valutazioni di `data_loss()`, `boundary_loss()` e `evaluate_final_losses()` utilizzino la medesima formula matematica esplicita e priva di ambiguità di riscalamento.

---

## 4. Cast Centralizzato a FP64 e Asserts pre-L-BFGS

### Cast Centralizzato
Abbiamo rimosso i cast `.double()` sparsi e centralizzato la conversione del modello, del physics problem, delle coordinate e di tutti i dizionari dei boundary groups in una singola funzione prima della fase L-BFGS:
* `convert_to_fp64(model, physics, data)`

### Controlli pre-L-BFGS (Asserts & Debug Report)
Prima di avviare il solutore L-BFGS, vengono eseguiti degli assert stringenti per verificare che modello, dati fisici, coordinate e tutti i singoli boundary data siano effettivamente in formato `float64`. Viene inoltre stampato a terminale un report di debug con i dtypes e le shapes correnti.

---

## 5. Clamping Fisico con Softplus e Logging Variazioni

### Softplus Clamping
* Per i parametri fisici che devono rimanere strettamente positivi (`mu_s`, `mu_p`, `lam`), il clamp rigido a `1e-6` è stato sostituito da un clamping basato su `torch.nn.functional.softplus` applicato se scendono sotto le soglie minime configurate.
* Per i parametri `eps` ed `alpha`, che possono scendere a zero, viene mantenuto il clamp standard a `0.0`.
* Le soglie minime di clamping sono state raggruppate in costanti globali all'inizio del file:
```python
MIN_MU_S = 1e-6
MIN_MU_P = 1e-6
MIN_LAM = 1e-6
```

### Logging di Clamping
La funzione `clamp_params()` monitora i valori prima e dopo lo step. Se un parametro viene modificato dal vincolo di clamping, viene stampato a terminale e salvato nel file di log un report di debug dettagliato (es. `mu_s: old -> new (Softplus clamp)`).

---

## 6. Metriche L2 Mascherate (Zone Attive) e Plotting dell'Errore

### Metriche Mascherate sullo Stress
Nelle zone del dominio in cui il valore esatto di stress è prossimo a zero, l'errore relativo L2 globale può risultare fuorviante. Abbiamo quindi introdotto il calcolo dell'errore L2 relativo per `tau_xx`, `tau_xy` e `tau_yy` calcolato esclusivamente sulle zone in cui lo stress totale COMSOL non è nullo (soglia impostata al **5%** del valore massimo di stress registrato nel dominio).

### Tracciamento e Plotting della storia degli Errori
* Gli errori L2 (globali e mascherati) vengono computati ad ogni intervallo di log e inseriti nella classe `SimpleHistory`.
* Al termine del training, viene generato e salvato nella cartella del caso un grafico logaritmico che mostra l'andamento di tutti gli errori L2 durante il training:
  * File generato: `l2_errors_history.png`
