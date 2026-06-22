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
![[Runs/Fisso/4_roll_mill_StressOnly_L8x128_E40000_SiLU_invFalse_20260617_152642/global_fields.png]]
Ovviamente la rete only data impara molto bene velocità e pressione, mentre non viene trovato il profilo di stress. 
![[Runs/Fisso/4_roll_mill_StressOnly_L8x128_E40000_SiLU_invFalse_20260617_152642/high_stress.png]]![[Runs/Fisso/4_roll_mill_StressOnly_L8x128_E40000_SiLU_invFalse_20260617_152642/loss_history.png]]
![[Runs/Fisso/4_roll_mill_StressOnly_L8x128_E40000_SiLU_invFalse_20260617_152642/l2_errors_history.png]]
# Next run
[[u,v,p solo data 50+5k - entrambe PDE 30+3k]]: provo ad aumentare le epoche mantenendo lo stesso setup