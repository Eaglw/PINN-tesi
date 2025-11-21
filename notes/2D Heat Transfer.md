# Approccio PINN per il Trasporto di Calore in una Lastra 2D

Questo documento descrive come impostare un modello PINN (Physics-Informed Neural Network) per risolvere l'equazione del calore in regime stazionario su una lastra bidimensionale di spessore infinito (problema 2D).

## 1. Formulazione del Problema Fisico

L'obiettivo è trovare la distribuzione di temperatura `T(x, y)` all'interno di una lastra piana. In condizioni di stato stazionario (senza variazione nel tempo) e senza sorgenti di calore interne, l'equazione che governa il fenomeno è l'**equazione di Laplace**:

```
∂²T/∂x² + ∂²T/∂y² = 0
```

Questa equazione rappresenta il "residuo" della fisica del problema, che il nostro PINN dovrà imparare a minimizzare.

### Condizioni al Contorno (Boundary Conditions - BCs) del Caso Specifico

Analizziamo il caso richiesto:
- **Facce sinistra e destra:** Temperatura fissa (condizione di **Dirichlet**).
  - `T(x_min, y) = T_left`
  - `T(x_max, y) = T_right`
- **Facce superiore e inferiore:** Isolamento termico perfetto (condizione di **Neumann**). L'isolamento implica che il flusso di calore normale alla superficie è nullo. Poiché il flusso è proporzionale al gradiente di temperatura, imponiamo che la derivata della temperatura rispetto alla direzione normale (in questo caso, `y`) sia zero.
  - `∂T/∂y |_(x, y_max) = 0`
  - `∂T/∂y |_(x, y_min) = 0`

## 2. Impostazione del Modello PINN

Per risolvere questo problema, costruiamo una rete neurale e la addestriamo a rispettare sia l'equazione di Laplace sia le condizioni al contorno.

### Architettura della Rete

La rete neurale `T_NN(θ)` è un approssimatore della funzione di temperatura. La sua struttura sarà:
- **Input:** Le coordinate spaziali `(x, y)`. (Dimensione input: 2)
- **Output:** La temperatura predetta `T_pred` in quel punto. (Dimensione output: 1)
- **Struttura:** Una rete fully-connected (FCN) con diversi layer nascosti e una funzione di attivazione (es. `Tanh`, `GELU`), simile a quelle già usate nel progetto.

### Funzione di Loss

La loss totale è il cuore del PINN e guida l'addestramento. È una somma pesata di diverse componenti:

`L_total = w_pde * L_pde + w_bc * L_bc`

1.  **Loss sulla Fisica (`L_pde`)**:
    Questa loss forza la rete a rispettare l'equazione di Laplace all'interno del dominio.
    - Si campionano N punti `(x_pde, y_pde)` all'interno della lastra (chiamati collocation points).
    - Per ogni punto, si calcola il residuo dell'equazione usando l'output della rete e le sue derivate seconde (ottenute con *automatic differentiation*):
      `R = ∂²T_pred/∂x² + ∂²T_pred/∂y²`
    - La loss è l'errore quadratico medio (MSE) di questo residuo: `L_pde = MSE(R)`.

2.  **Loss sulle Condizioni al Contorno (`L_bc`)**:
    Questa loss forza la rete a rispettare le condizioni imposte sui bordi. Si suddivide in quattro parti, una per ogni bordo.
    - **Loss Dirichlet (sinistra/destra)**:
      - Si campionano N punti sui bordi `(x_min, y_bc)` e `(x_max, y_bc)`.
      - Si calcola la differenza tra la predizione della rete e il valore di temperatura imposto.
      - `L_bc_left = MSE(T_pred(x_min, y_bc) - T_left)`
      - `L_bc_right = MSE(T_pred(x_max, y_bc) - T_right)`
    - **Loss Neumann (sopra/sotto)**:
      - Si campionano N punti sui bordi `(x_bc, y_max)` e `(x_bc, y_min)`.
      - Si calcola la derivata `∂T_pred/∂y` in questi punti tramite *automatic differentiation*.
      - La loss è l'errore quadratico medio tra questa derivata e il valore imposto (zero).
      - `L_bc_top = MSE(∂T_pred/∂y |_(x_bc, y_max) - 0)`
      - `L_bc_bottom = MSE(∂T_pred/∂y |_(x_bc, y_min) - 0)`

La loss totale diventa quindi:
`L_total = w_pde*L_pde + w_dirichlet*(L_bc_left + L_bc_right) + w_neumann*(L_bc_top + L_bc_bottom)`
I pesi `w` sono iperparametri che bilanciano l'importanza relativa della fisica e delle condizioni al contorno durante l'addestramento.

## 3. Variazioni al Variare delle BCs

La potenza dei PINN risiede nella loro flessibilità. Modificare le condizioni fisiche del problema si traduce semplicemente nel modificare le componenti della funzione di loss.

- **Tutte facce a T fissa (Dirichlet su tutti i lati)**:
  - Tutte e quattro le componenti della `L_bc` sarebbero di tipo Dirichlet, simili a `L_bc_left` e `L_bc_right`. Non ci sarebbe bisogno di calcolare derivate per la loss sui bordi.

- **Tutte facce isolate (Neumann su tutti i lati)**:
  - Tutte e quattro le componenti della `L_bc` sarebbero di tipo Neumann.
  - **Attenzione**: Un problema con solo condizioni di Neumann può essere mal posto (la soluzione è definita a meno di una costante). Per ottenere una soluzione unica, è solitamente necessario "ancorare" la temperatura in almeno un punto del dominio (ad esempio, imponendo `T(0,0) = T_ref` con un'ulteriore componente di loss).

- **Condizione di Convezione (Robin BC)**:
  - Immaginiamo che la faccia superiore scambi calore per convezione con un fluido a temperatura `T_amb`. La condizione fisica è: `-k * ∂T/∂y = h * (T - T_amb)`, dove `k` è la conducibilità termica e `h` il coefficiente di scambio termico.
  - La componente di loss `L_bc_top` diventerebbe:
    `L_bc_top = MSE(-k * ∂T_pred/∂y - h * (T_pred - T_amb))`
  - Questo dimostra come anche condizioni al contorno più complesse, che legano il valore della funzione e della sua derivata, possano essere modellate elegantemente modificando la formula matematica della loss.

In conclusione, l'architettura della rete rimane la stessa, ma la "funzione obiettivo" (la loss) viene adattata per rispecchiare fedelmente la fisica del problema specifico che si vuole risolvere.
