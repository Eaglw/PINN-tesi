# COMSOL Boundary Extraction

## Overview
Questa pagina documenta la procedura passo-passo per definire e nominare i diversi confini (Boundary Conditions - BC) all'interno di un modello COMSOL Multiphysics. Questo passaggio è fondamentale per rendere i confini facilmente identificabili ed esportabili in fase di pre-processing dei dati per le Physics-Informed Neural Networks (PINNs).

## Procedura Passo-Passo

1. **Navigare nell'Albero del Modello (Model Builder)**:
   - Aprire il file COMSOL (`.mph`).
   - Nell'albero del modello a sinistra, individuare ed espandere il componente di interesse (es. **Component 1**).

2. **Accedere alle Selezioni**:
   - Espandere la sezione **Definitions** del componente.
   - Fare clic destro su **Definitions** per aprire il menu contestuale.

3. **Creare una Selezione Esplicita (Explicit Selection)**:
   - Dal menu contestuale, selezionare **Selections** e poi cliccare su **Explicit** (oppure utilizzare la scheda *Definitions* nella barra degli strumenti in alto e cliccare sull'icona **Explicit**).

4. **Configurare la Selezione per i Confini**:
   - Nelle impostazioni del nodo *Explicit* appena creato (nella scheda centrale *Settings*):
     - Cambiare la voce **Geometric entity level** da *Domain* a **Boundary**.
   - Selezionare graficamente nella finestra 3D/2D o inserire gli ID dei confini (es. contorni delle pareti, dell'inlet o dell'outlet) che comporranno la condizione al contorno desiderata.

5. **Assegnare un Nome Identificativo**:
   - Rinominare il nodo *Explicit* (cliccando con il tasto destro e selezionando **Rename**, oppure selezionando il nodo e premendo `F2`) con un nome significativo e coerente con la BC da applicare (ad esempio `Inlet`, `Outlet`, `Walls`).
   - Questo nome permetterà di richiamare e isolare facilmente le coordinate di questi confini quando la geometria o i dati della mesh verranno esportati e letti dagli script Python della PINN.

## Riferimenti
- [[00_Index]]
- [[Viscoelastic_Training]]
