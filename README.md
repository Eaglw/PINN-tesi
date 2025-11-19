# PINN Tesi Project

This project focuses on using Physics-Informed Neural Networks (PINNs) for thesis work.
## Recent Changes

Here is a summary of the latest commits:
* ddfb546: Ulteriore modifica agli step e al campionamento di punti per il problema inverso
* 53e4a1f: Ottimizzato gli step di training e il salvataggio di grafici
* fe0cec6: readme update
* 28ebb48: Improve readability
* 854ae31: CSTR inverso: Impostato l'addestramento per trovare i parametri fisici

## Virtual environment (Python)

Per creare e usare un virtual environment Python con `venv` su macOS (shell `zsh`), puoi seguire questi passaggi consolidati:

```bash
python3 -m venv .venv              # Crea l'ambiente virtuale chiamato '.venv'
source .venv/bin/activate         # Attiva l'ambiente virtuale (per zsh/bash)
pip install --upgrade pip         # Aggiorna pip alla versione più recente (opzionale ma consigliato)
pip install -r requirements.txt   # Installa tutte le dipendenze del progetto
# ... (lavora nel tuo ambiente virtuale) ...
deactivate                        # Disattiva l'ambiente virtuale
```

Nota: se preferisci creare l'ambiente con un nome diverso, sostituisci `.venv` con il nome scelto. Se usi un'altra shell, il comando di attivazione rimane `source <env>/bin/activate`.