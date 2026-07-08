# Pressure Point Anchoring

## Overview
Nelle Physics-Informed Neural Networks (PINNs) applicate alla fluidodinamica incomprimibile (Navier-Stokes o flussi viscoelastici), la pressione $p$ è determinata matematicamente solo a meno di una costante additiva, in quanto nelle equazioni di bilancio della quantità di moto compare unicamente sotto forma di gradiente ($\nabla p$). 

Per rendere il problema ben posto dal punto di vista numerico ed evitare che il campo di pressione fluttui o si allontani dai valori reali durante l'ottimizzazione, è necessario ancorare la pressione ad un valore noto in almeno un punto del dominio (condizione di Dirichlet puntuale). Questo punto di ancoraggio prende il nome di **Pressure Point**.

## Technical Implementation & Physical Details

Nel framework del progetto, il caricamento e la definizione del `PressurePoint` avvengono all'interno della pipeline di pre-processing dei dati (in particolare nella funzione `_extract_boundary_groups` di `utils.py`). Il codice segue due strategie in cascata:

### 1. Selezione Esplicita da COMSOL (Principale)
Se nel file della mesh COMSOL (`.mphtxt`) è stata definita una selezione geometrica esplicita per il punto di pressione (es. rinominata in COMSOL come `PressurePoint`):
* Il codice ne effettua il parsing identificando gli ID dei nodi associati.
* Tramite un algoritmo KD-Tree (`cKDTree`), mappa questi nodi ai corrispondenti punti nel dataset CSV della mesh.
* La loss associata a questa condizione al contorno viene calcolata come:
  $$ \mathcal{L}_{BC, p} = \frac{1}{N_{BC}} \sum_{i=1}^{N_{BC}} \frac{\left(p_{\theta}(x_i) - p_{COMSOL}(x_i)\right)^2}{\sigma^2_p} $$
  dove $p_{\theta}$ è la pressione predetta dal modello, $p_{COMSOL}$ è il valore di riferimento, e $\sigma^2_p$ è la varianza della pressione usata come peso di normalizzazione.

### 2. Meccanismo di Fallback Automatico
Qualora nel file mesh non sia presente alcuna selezione etichettata con `"pressure"` (case-insensitive), il codice attiva un fallback per evitare divergenze:
* Preleva il primo gruppo di contorno disponibile (di solito la parete esterna, `Walls`).
* Estrae il **primo nodo** di questo gruppo e lo definisce come `PressurePoint` per l'ancoraggio.

#### Esempio di Fallback (Caso Four-Roll Mill)
Nel dataset standard `4_roll_mill.csv`, in assenza di un `PressurePoint` esplicito da COMSOL, il fallback seleziona il seguente punto:
* **Indice nel dataset**: `111092`
* **Coordinate adimensionali** ($x_{nd}, y_{nd}$): `(1.0000, 0.1348)`
* **Coordinate fisiche (raw)** ($x, y$ in metri): `(0.025, -0.01826)`
* **Posizione geometrica**: Sulla parete destra del box quadrangolare del dominio ($[-0.025, 0.025] \times [-0.025, 0.025]$ m), vicino all'angolo in basso a destra.

## References & Back-links
- [[00_Index]]
- [[COMSOL_Boundary_Extraction]]
- [[Viscoelastic_Training]]
