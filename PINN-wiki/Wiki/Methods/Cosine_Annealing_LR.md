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
Nello schema di addestramento del progetto PINN viscoelastico (specialmente in modalità `semi_inverse` e staged training), il Cosine Annealing viene utilizzato durante le fasi di esplorazione con Adam per regolare in modo continuo il learning rate prima delle rispettive transizioni alle fasi di raffinamento fisico con L-BFGS a doppia precisione (FP64).

### 1. Protocollo Cosine Annealing Sincronizzato (Fase 1)
In **Fase 1** (Cinematica & Reologia: `model_psi`, `model_tau`, e parametri inversi $\lambda, \mu_p, \alpha, \varepsilon$), il framework adotta uno schedule coordinato su due gruppi di parametri:
- **Gruppo Reti Neurali** (`model_psi`, `model_tau`):
  - $\eta_{\max} = \text{BASE\_LR} = 2.5 \times 10^{-3}$
  - $\eta_{\min} = \text{ETA\_MIN} = 2.5 \times 10^{-6}$
- **Gruppo Parametri Fisici Reologici** ($\lambda, \mu_p, \alpha, \varepsilon$):
  - Con $\text{PARAM\_LR\_FACTOR} = 1.0$, il tasso iniziale dei parametri fisici coincide esattamente con quello delle reti:
    $$\eta_{\text{phys, max}} = \text{BASE\_LR} \times \text{PARAM\_LR\_FACTOR} = 2.5 \times 10^{-3}$$
  - $\eta_{\text{phys, min}} = \text{ETA\_MIN} = 2.5 \times 10^{-6}$

#### Vantaggio Fisico della Sincronizzazione
Sincronizzare l'intervallo $[2.5 \times 10^{-3}, 2.5 \times 10^{-6}]$ per entrambi i gruppi impedisce che l'apprendimento delle strutture cinematiche veloci distacchi l'adattamento dei parametri reologici o viceversa. Lungo l'intero orizzonte di $T_{\max}$ epoche Adam, le reti e i parametri scalari viaggiano all'unisono:

$$ \eta_t = 2.5 \times 10^{-6} + \frac{1}{2}(2.5 \times 10^{-3} - 2.5 \times 10^{-6}) \left( 1 + \cos\left(\frac{t \pi}{T_{\max}}\right) \right) $$

### 2. Implementazione PyTorch nel Framework
In `final_roll/src/train.py`:
```python
# Setup param groups dedicati con Adam epsilon differenziato
groups = [
    {"params": net_params, "lr": BASE_LR, "eps": ADAM_EPS},
    {"params": phys_params_ph1, "lr": BASE_LR * PARAM_LR_FACTOR, "eps": ADAM_EPS_PHYS}
]
optimizer = torch.optim.Adam(groups, eps=ADAM_EPS)

eta_min_ph1 = getattr(builtins, "ETA_MIN", 2.5e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(steps_rem, 1), eta_min=eta_min_ph1
)
```

## References & Back-links
- [[Staged_Training_Procedure]]
- [[ViscoelasticNet_Full model]]
- [[Viscoelastic_Training]]
- [[Dynamic_Weighting]]
- [[01_Log]]
