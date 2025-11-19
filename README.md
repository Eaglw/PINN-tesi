# PINN Tesi Project

This project focuses on using Physics-Informed Neural Networks (PINNs) for thesis work.

## Domande 

- Il caso ancora più semplificato di ODE è comunque rappresentativo o ha intrinsecamente dei problemi o limiti? Nel rappresentare casi più complessi o conndimensionalità diverse?
- Aggiungendo anche bilancio di energia come sarebbe cambiata l'analisi?

- Trend del caso no data, la BC è corretta?
- Perchè abbiamo ingrandito la rete nel problema inverso?
- Normale che solo con loss fisica sia decisamente più lento?
- "single_point = x[0].unsqueeze(1) single_point.requires_grad = True" va dentro o fuori dal loop? (no data vs inverse)
- Nell inverse pretraining serve BC?

- Campionamento dei punti analitici sia in posizione che in densità come potrebbe influenzare?
- Incertezza su parametri fisici?
- Funzioni di attivazione e ottimizzatori
- Come bilanciare dinamicamente i pesi delle varie loss

## Recent Changes

Here is a summary of the latest commits:
* ddfb546: Ulteriore modifica agli step e al campionamento di punti per il problema inverso
* 53e4a1f: Ottimizzato gli step di training e il salvataggio di grafici
* fe0cec6: readme update
* 28ebb48: Improve readability
* 854ae31: CSTR inverso: Impostato l'addestramento per trovare i parametri fisici

## Virtual environment (Python)

Per creare e usare un virtual environment Python con `venv` su macOS (shell `zsh`):

- Creare l'ambiente virtuale:

```
python3 -m venv .venv
```

- Attivare l'ambiente (zsh):

```
source .venv/bin/activate
```

- Aggiornare `pip` (opzionale ma consigliato):

```
pip install --upgrade pip
```

- Installare i requirements del progetto:

```
pip install -r requirements.txt
```

Per disattivare l'ambiente virtuale, eseguire:

```
deactivate
```

Nota: se preferisci creare l'ambiente con un nome diverso, sostituisci `.venv` con il nome scelto. Se usi un'altra shell (ad es. `bash`), il comando di attivazione è analogo: `source <env>/bin/activate`.