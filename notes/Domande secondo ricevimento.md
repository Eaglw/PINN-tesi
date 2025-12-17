## Coupled
* Introducendo la seconda rete ho loss altissime se non faccio un pretrain solo su dati e non anche su fisica. Sto manualmente imponendo tot epoche, ci sono modi migliori?
* Ho normalizzato concentrazione e temperatura semplicemente su valori plausibili. E' abbastanza o ci sono approcci migliori?
* bilanciare i pesi della loss in modo "manuale" e statico quanto può penalizzare in termini di convergenza rispetto ad approcci più complessi?

## Problema Inverso 
* Ho letto che conviene lavorare sui ln dei valori per i problemi inversi, è vero? 
* Il problema della normalizzazione è presente anche nel caso inverso? conviene addestrare la rete a lavorare comunque con valori tra 0 e 1? 

* come funziona l'overfitting nelle pinns? è 

## Heat2D
* ho avuto problemi con l'inizializzazione dei dati che poteva portare a loss altissime, cercando un seed specifico funzionante e imponendolo ha funzionato, ma come posso evitare questo comportamento?

- il trend di loss per gelu e tanh è coerente?