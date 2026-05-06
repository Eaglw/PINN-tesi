# PINN Heat2D: Tecniche e Paradigmi Implementati

Questo documento fornisce una spiegazione teorica delle tecniche avanzate utilizzate per ottimizzare la Physics-Informed Neural Network (PINN) applicata all'equazione del calore 2D (Laplace).

## 1. Architetture Tapered (Imbuto)
Invece di utilizzare un numero costante di neuroni per ogni strato (es. 80x6), l'architettura segue una struttura a "imbuto" (es. `[120, 100, 80, 60, 40, 20]`). 
**Razionale:** Permette alla rete di apprendere feature complesse negli strati iniziali e di "condensarle" progressivamente verso l'output, riducendo il rischio di overfitting e migliorando la convergenza su problemi con domini regolari come quello di Laplace.

## 2. Learnable Adaptive Activations (LAA)
Le funzioni di attivazione adattive introducono parametri scalabili addestrati insieme ai pesi della rete:
$f(x) = \sigma(a \cdot x)$
Dove $\sigma$ è la funzione di attivazione (es. GELU o Tanh) e $a$ è un parametro scalare (spesso inizializzato a 1.0).
**Razionale:** Permette alla rete di cambiare la pendenza della funzione di attivazione localmente, aiutando a catturare gradienti ripidi o variazioni lente nel campo di temperatura senza aumentare eccessivamente il numero di parametri.

## 3. Campionamento Quasi-Monte Carlo (Sobol/Halton)
A differenza del campionamento casuale uniforme (Pseudo-Random), le sequenze Sobol o Halton sono progettate per coprire il dominio in modo più uniforme (Low-Discrepancy Sequences).
**Razionale:** Riduce i "buchi" nel campionamento del dominio e previene l'addensamento casuale di punti, portando a una stima più accurata del residuo della PDE (Loss Fisica) con lo stesso numero di punti.

## 4. Spatially Adaptive Refinement (SAR)
La tecnica SAR (ispirata alla Residual-based Adaptive Refinement - RAR) consiste nell'identificare, durante l'addestramento, le aree del dominio dove il residuo della PDE è più alto.
**Razionale:** Viene aggiunta una densità maggiore di punti di addestramento in quelle zone specifiche. Questo forza la rete a correggere gli errori dove la fisica non è rispettata rigorosamente, ottimizzando la precisione locale.

## 5. Bilanciamento della Loss (Boundary Weighting)
La loss totale è una somma pesata: $L_{tot} = L_{pde} + \lambda_{bc} \cdot L_{bc}$.
**Razionale:** Nelle fasi iniziali, un peso elevato su $\lambda_{bc}$ (es. 50 o 100) è fondamentale per "ancorare" la soluzione ai valori corretti sui bordi. Senza questo, la rete potrebbe trovare soluzioni che soddisfano la PDE ma violano le condizioni al contorno.

## 6. Transfer Learning (Coarse-to-Fine)
L'addestramento viene diviso in fasi: prima su una griglia grossolana (punti ridotti) per catturare la forma generale della soluzione, poi su una griglia fine o con punti casuali densi.
**Razionale:** Accelera la convergenza iniziale e previene che la rete si perda in minimi locali complessi durante le prime epoche.
