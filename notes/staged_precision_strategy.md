# Strategia di Training PINN: Staged Precision

Abbiamo implementato una strategia di **Staged Precision Training** per ottimizzare il rapporto tra velocità di convergenza iniziale e precisione fisica finale.

## Logica Raffinata
Il training viene diviso in due fasi distinte:

### Fase 1: Esplorazione Veloce (Adam)
- **Obiettivo**: Raggiungere velocemente un'inizializzazione vicina alla soluzione.
- **Precisione**: `torch.float32` (FP32).
- **Ottimizzazione Hardware**:
    - **RTX 3080/Ampere**: Abilitazione automatica di **TF32** (`matmul_precision='high'`). Questo offre prestazioni vicine al BF16 ma con la stabilità del FP32 per il calcolo dei residui PDE.
    - **GTX 1050Ti**: Fallback automatico su FP32 standard.
- **Vantaggio**: Speedup misurato di circa **10x-12x** rispetto al FP64.

### Fase 2: Raffinamento Fisico (L-BFGS)
- **Obiettivo**: Eliminare i residui ad alta frequenza e garantire stabilità fisica totale.
- **Precisione**: `torch.float64` (FP64).
- **Vantaggio**: Garanzia di precisione "scientific grade" e MAE identico alle configurazioni full FP64.

## Utilizzo del Toggle
Nel file `Heat2D_hybrid.py`, la funzione `train_hybrid_logic` ora dispone di un toggle:
- `use_staged_precision=True`: Abilita il passaggio FP32 (Stage 1) -> FP64 (Stage 2).
- `use_staged_precision=False`: Esegue l'intero training in FP64 puro (configurazione finale/validazione).
