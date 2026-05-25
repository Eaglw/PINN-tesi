import torch
import sys

def check_is_1050ti():
    """
    Controlla se il sistema utilizza una GPU limitata come la GTX 1050 Ti 
    o ha meno di 4.5 GB di VRAM. Utile per abilitare opzioni di fallback 
    (es. history_size ridotto in L-BFGS, disabilitazione CUDA Graphs).
    """
    if not torch.cuda.is_available():
        return False
    try:
        device_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory
        return "1050" in device_name or total_mem < 4.5 * 1024**3
    except:
        return False

# Inizializziamo subito una variabile globale per comoda importazione
IS_1050TI = check_is_1050ti()

def supports_compile():
    """
    Verifica se torch.compile è utilizzabile sul sistema corrente.
    Requisiti: PyTorch >= 2.0, Python < 3.14, CUDA disponibile,
    GPU con Compute Capability >= 7.0 (Turing/Ampere+).
    La 1050 Ti (Pascal, CC 6.1) non supporta il backend Triton.
    Python 3.14+ non è ancora supportato da torch.compile.
    """
    if int(torch.__version__.split('.')[0]) < 2:
        return False
    if not torch.cuda.is_available():
        return False
    if IS_1050TI:
        return False
    return True

SUPPORTS_COMPILE = supports_compile()
