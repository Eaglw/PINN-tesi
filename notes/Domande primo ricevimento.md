1. Il caso ancora più semplificato di ODE è comunque rappresentativo o ha intrinsecamente dei problemi o limiti? Nel rappresentare casi più complessi o conndimensionalità diverse?
Essenzialmente il problema è la convergenza, quindi nel caso semplificato non si possono apprezzare le difficoltà del problema diretto dato che converge molto falcilmente, ma concettualmente è tutto corretto

2.  Aggiungendo anche bilancio di energia come sarebbe cambiata l'analisi?
Avrei potuto fare una seconda rete aggiungendo il bilancio di energia, e andrebbero risolte in modo simultaneo(?)
  
3.  Trend del caso no data, la BC è corretta?
Si perchè essendoci più contributi di loss, quindi mass balance e BC, viene minimizzato prima quello di ordine maggiore, che in questo caso probabilmente è il mass balance. Una volta che la loss del mass balance inizia a diventare comparabile con quella della BC allora inizia a cambiare anche quella zona.

4. Perchè abbiamo ingrandito la rete nel problema inverso?
Ovviamente va aumentata la dimensione all'aumentare della complessità del problema, ma non è stato fatto perchè dimostrato necessario strettamente

5.  Normale che solo con loss fisica sia decisamente più lento?
Si perchè rispetto a fittare dei punti sperimentali soddisfare il mass balance è molto più complesso

 6. "single_point = x[0].unsqueeze(1) single_point.requires_grad = True" va dentro o fuori dal loop? (no data vs inverse)
tenicamente va fuori, dato che non serve ridefinirlo ad ogni epoca, ma essendo la BC fissa e non dipendente dal tempo non cambia in questo caso

7. Nell inverse pretraining serve BC?
No ma è comunque qualcosa che aiuta nella convergenza quindi male non fa 
  
8. Campionamento dei punti analitici sia in posizione che in densità come potrebbe influenzare?
questa è una bella domanda che non ha una reale risposta, infatti va fatta un analisi a posteriori per vedere come la densità e la posizione dei punti campionati influenzano sulle performance della rete
9.  Incertezza su parametri fisici?
//
10. Funzioni di attivazione e ottimizzatori?
Rispetto al classico Adams si può usare esclusivamente o successivamente ad un pretraining anche LBFGS per convergere, più lentamente, ma verso una soluzione più precisa.
11. Come bilanciare dinamicamente i pesi delle varie loss
Non so precisamente come farlo dinamicamente ma si possono aggiungere dei pesi per compensare ordini di grandezza diversi.