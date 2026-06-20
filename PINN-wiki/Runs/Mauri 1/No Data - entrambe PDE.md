---
epochs: "100000"
staged: false
inverse: false
date: 2026-06-17T12:00:00
Computer: Maurizio
notes: ENDED
---
## Idea
L'idea è stata quella di runnare bruteforce il problema diretto, senza la supervisione dei dati del campo di velocità. In particolare usando dall'inizio solo le PDE, sia momentum che constitutive, per il training. 

# Results
![[Runs/Mauri 1/L2-error.png]]
![[Runs/Mauri 1/Loss.png]]
## Next run
[[entrambe PDE + u,v data]]
