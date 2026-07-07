---
date: 2026-07-07T18:29:02
inverse_problem: false
dataset: 4_roll_mill.csv
epochs: 10000
Computer: Kaggle
staged: true
inverse: false
---
## Idea
ho provato a runnare solo PDE per la pressione, quindi momentum, a partire dai campi di velocità e stress già dati. Non trainando una rete, ma proprio facendo in modo che le autograd vengano fatte sui dati di comsol. Ovviamente è andato molto velocemente a convergenza. 

### Global Fields
![[Runs/Kaggle/u,tau data no NN -p PDE/global_fields.png]]


### L2 Errors History

![[L2_error.png]]
### Loss History
![[Runs/Kaggle/u,tau data no NN -p PDE/loss_history.png]]
