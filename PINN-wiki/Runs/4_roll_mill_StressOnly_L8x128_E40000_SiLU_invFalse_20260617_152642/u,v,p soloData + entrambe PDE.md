---
date: 2026-06-17T15:00:00
staged: false
inverse: false
epochs: "40000"
Computer: Fisso
description:
---

> [!NOTE]+ Nota sul dataset
> Questo era il dataset con lambda=1, ma poi le prove sono cambiate con lambda=0.05

## Idea

Abbiamo provato a runnare 30k epoche di NN solo data driven per imparare i campi di velocità e pressione e poi ì, a partire da quelli, fare il train sullo stress. 
In particolare abbiamo attivato entrambe le PDE, con la loss che è composta dalla somma di momentum e constitutive. 
## Results
![[global_fields.png]]
Ovviamente la rete only data impara molto bene velocità e pressione, mentre non viene trovato il profilo di stress. 
![[high_stress.png]]![[loss_history.png]]
![[l2_errors_history.png]]