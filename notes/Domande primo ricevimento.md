1. Il caso di ODE ancora più semplificato è comunque rappresentativo, oppure presenta limiti intrinseci quando si estende a sistemi più complessi o a differenti dimensionalità?
Essenzialmente il problema è la convergenza: nel caso semplificato le difficoltà del problema diretto non sono evidenti, perché il metodo tende a convergere molto facilmente; tuttavia, concettualmente l'approccio rimane corretto.

2. Se si aggiungesse anche il bilancio di energia, come cambierebbe l'analisi?
Si potrebbe introdurre una seconda rete che incorpori il bilancio di energia; in tal caso le equazioni dovrebbero essere trattate e risolte in modo simultaneo.

3. Nel caso "no data", il comportamento imposto dalle condizioni al contorno (BC) è corretto?
Sì. Avendo più contributi nella loss (ad es. mass balance e BC), viene minimizzato per primo il termine dominante — probabilmente il mass balance. Quando la loss del mass balance diventa comparabile a quella delle BC, anche la soluzione nella regione interessata inizia a modificarsi.

4. Perché abbiamo aumentato la dimensione della rete nel problema inverso?
La dimensione della rete va aumentata con la complessità del problema; nel caso specifico non è stato fatto perché non si era dimostrato strettamente necessario, ma spesso si procede per precauzione quando la complessità aumenta.

5. È normale che, usando soltanto la loss fisica, l'allenamento risulti decisamente più lento?
Sì. Rispetto al fitting di punti sperimentali, soddisfare vincoli fisici come il mass balance è generalmente più complesso e richiede più tempo di apprendimento.

6. La riga di codice
```
single_point = x[0].unsqueeze(1)
single_point.requires_grad = True
```
va posta dentro o fuori dal loop (no-data vs inverse)?
Tecnicamente va messa fuori dal loop, perché non è necessario ridefinirla ad ogni epoca; inoltre qui la BC è fissa e non dipende dal tempo.

7. Nel pretraining per il problema inverso è utile includere le condizioni al contorno?
Non è indispensabile, ma le BC possono facilitare la convergenza, quindi inserirle non è svantaggioso.

8. In che modo il campionamento dei punti analitici — sia in posizione che in densità — può influenzare le prestazioni?
È una questione importante senza una risposta unica: occorre un'analisi a posteriori per valutare come la densità e la posizione dei punti campionati influenzino le prestazioni della rete.

9. Come trattare l'incertezza sui parametri fisici?
Da approfondire: va valutata separatamente l'influenza dell'incertezza sui parametri e le possibili strategie (es. stime bayesiane, ensemble, o analisi di sensitività).

10. Quali funzioni di attivazione e quali ottimizzatori usare?
Oltre all'ottimizzatore classico (Adam), è possibile usare LBFGS — esclusivamente o dopo un pretraining con Adam — per ottenere una convergenza più lenta ma, spesso, una soluzione più precisa.
 
11. Come bilanciare dinamicamente i pesi delle varie componenti della loss?
Non ho una procedura precisa definita: una soluzione semplice è introdurre pesi per compensare differenze di ordine di grandezza; esistono anche strategie dinamiche ma richiedono implementazione e test specifici.