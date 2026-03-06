import torch
import torch.nn as nn

class PrecisionConfig:
    """
    Configurazione per i livelli di precisione dei componenti della PINN.
    """
    def __init__(self, nn_opt=torch.float64, data=torch.float64, physics=torch.float64, bc=torch.float64):
        self.nn_opt = nn_opt # NN e Optimizer devono avere la stessa precisione per stabilità
        self.data = data
        self.physics = physics
        self.bc = bc

    @classmethod
    def from_bitmask(cls, mask, parts=['nn_opt', 'data', 'physics', 'bc']):
        """
        Crea PrecisionConfig da una bitmask. 0 = float32, 1 = float64
        """
        config = {}
        for i, part in enumerate(parts):
            config[part] = torch.float64 if (mask >> i) & 1 else torch.float32
        return cls(**config)

    def __repr__(self):
        def dname(dt): return "FP64" if dt == torch.float64 else "FP32"
        return f"PC(Net/Opt={dname(self.nn_opt)}, Data={dname(self.data)}, Phys={dname(self.physics)}, BC={dname(self.bc)})"

def cast_to(tensor, dtype):
    if tensor is None: return None
    return tensor.to(dtype)

def compute_data_loss(model, x, y, config: PrecisionConfig):
    # Calcolo loss dati nella precisione desiderata
    x_c = cast_to(x, config.data)
    y_c = cast_to(y, config.data)
    # Cast al volo dell'input per il modello (che è in config.nn_opt)
    pred = model(x_c.to(config.nn_opt))
    loss = nn.MSELoss()(pred, y_c.to(config.nn_opt))
    return loss.to(config.nn_opt)

def compute_bc_loss(model, x_bc, y_bc, config: PrecisionConfig, physics_problem=None):
    x_c = cast_to(x_bc, config.bc)
    y_c = cast_to(y_bc, config.bc)
    if physics_problem:
        # La physics_problem deve gestire il casting interno se necessario
        loss = physics_problem.boundary_loss(model, x_c.to(config.nn_opt), y_c.to(config.nn_opt))
    else:
        pred = model(x_c.to(config.nn_opt))
        loss = nn.MSELoss()(pred, y_c.to(config.nn_opt))
    return loss.to(config.nn_opt)

def compute_physics_loss(model, x_p, config: PrecisionConfig, physics_problem=None):
    x_c = cast_to(x_p, config.physics)
    if not x_c.requires_grad: x_c.requires_grad_(True)
    
    # Se la fisica è richiesta in FP32 ma il modello è FP64, 
    # i gradienti verranno comunque calcolati nella precisione del modello.
    # Per un vero test FP32 della fisica, il modello dovrebbe essere FP32.
    if physics_problem:
        loss = physics_problem.residual(model, x_c.to(config.nn_opt))
    else:
        return torch.tensor(0.0, device=x_p.device, dtype=config.nn_opt)
    return loss.to(config.nn_opt)
