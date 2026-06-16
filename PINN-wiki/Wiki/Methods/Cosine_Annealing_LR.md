# Cosine Annealing LR

## Overview
Il **Cosine Annealing** ("ricottura cosinusoidale") è un meccanismo di *learning rate scheduling* ispirato al processo metallurgico e termodinamico della ricottura: riscaldare un sistema per far muovere liberamente le particelle e poi raffreddarlo lentamente per raggiungere uno stato di energia minima cristallizzato.

Nelle reti neurali, l'idea è esplorare il *loss landscape* (la superficie dell'errore) con passi grandi all'inizio, per poi rallentare dolcemente man mano che ci si avvicina a un minimo, evitando di superarlo a causa dell'eccessiva inerzia.

### 1. La Formula in Forma Chiusa
Dal paper *SGDR (Stochastic Gradient Descent with Warm Restarts)*, la formula in forma chiusa è:

$$ \eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left( 1 + \cos\left(\frac{T_{cur} \pi}{T_{max}}\right) \right) $$

Dove:
- $\eta_{\max}$ è il learning rate massimo iniziale.
- $\eta_{\min}$ è il learning rate minimo desiderato alla fine del ciclo.
- $T_{cur}$ è il numero di epoche/iterazioni correnti dall'inizio del ciclo.
- $T_{max}$ è la lunghezza totale del ciclo (in epoche o iterazioni).

La discesa segue una curva cosinusoidale non lineare: decresce lentamente nelle primissime fasi, accelera a metà del training, e infine decelera nuovamente assestandosi in modo fluido su $\eta_{\min}$.

### 2. La Formula Ricorsiva (Implementazione PyTorch)
In PyTorch lo scheduler `CosineAnnealingLR` è implementato tramite una formula ricorsiva:

$$ \eta_{t+1} = \eta_{\min} + (\eta_t - \eta_{\min}) \cdot \frac{1 + \cos\left(\frac{(T_{cur}+1) \pi}{T_{max}}\right)} {1 + \cos\left(\frac{T_{cur} \pi}{T_{max}}\right)} $$

Questa scelta di design ingegneristico permette all'oggetto scheduler di non dover memorizzare l'$\eta_{\max}$ iniziale di ciascun *parameter group* dell'ottimizzatore (che potrebbero essere diversi e multipli), calcolando $\eta_{t+1}$ basandosi unicamente sullo stato corrente $\eta_t$.

### 3. Il Falso "Restart"
Nonostante il riferimento al paper SGDR, la classe `CosineAnnealingLR` **non effettua i restart**. Esegue un'unica discesa continua lungo l'intero intervallo di $T_{max}$ epoche. Se si desiderano riavvii periodici (in cui il learning rate "salta" nuovamente a $\eta_{\max}$ per sfuggire a minimi locali sub-ottimali), è necessario utilizzare `CosineAnnealingWarmRestarts`.

## Technical Implementation & Physical Details
Nello schema di addestramento del progetto PINN viscoelastico (specialmente in modalità `semi_inverse`), il Cosine Annealing viene utilizzato durante le fasi di ottimizzazione con Adam (Phase 1 e Phase 2) per regolare il learning rate prima della transizione alla Phase 3 di fine-tuning con L-BFGS a doppia precisione (FP64).

### PyTorch Example
```python
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

model = nn.Linear(10, 2)
# lr iniziale (eta_max) impostato a 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) 

# Vogliamo scendere fino a 0.001 in 100 epoche
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)

for epoch in range(100):
    # optimizer.step()
    
    # Aggiorna il LR seguendo la curva coseno
    scheduler.step() 
```

## References & Back-links
- [[Staged_Training_Procedure]]
- [[Dynamic_Weighting]]
- [[01_Log]]
