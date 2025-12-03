import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn

class TrainingHistory:
    """
    Una classe per registrare e visualizzare l'andamento delle loss durante il training.
    """
    def __init__(self):
        self.epochs = []
        self.losses = {}

    def update(self, epoch, loss_dict):
        """
        Registra i valori delle loss per un dato'epoch'.

        Args:
            epoch (int): L'epoch corrente.
            loss_dict (dict): Un dizionario con i nomi delle loss e i loro valori.
                              Es: {'total_loss': 1.5, 'pde_loss': 1.2, ...}
        """
        self.epochs.append(epoch)
        for name, value in loss_dict.items():
            # Se è un tensore, estrai il valore numerico
            val = value.item() if hasattr(value, 'item') else value
            
            if name not in self.losses:
                self.losses[name] = []
            self.losses[name].append(val)

    def plot_losses(self, last_adam_epoch=0, save_path=None, experiment_name="", show_plot=True):
        """
        Genera un grafico con l'andamento di tutte le loss registrate.
        """
        plt.figure(figsize=(8, 4))
        ax = plt.gca()
        for name, values in self.losses.items():
            if name == "total_loss":
                plt.plot(self.epochs, values, linewidth=4, label=name)
            else:
                plt.plot(self.epochs, values, label=name)
        
        plt.title(f'Andamento Loss - {experiment_name}', y=0.85)
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        if last_adam_epoch != 0:
            plt.axvline(last_adam_epoch, color="r", linestyle="--", label="Last adam epoch")
        # Stile coerente con plot_result
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        l = plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
        plt.setp(l.get_texts(), color="k")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Grafico delle loss salvato in: {save_path}")
        
        if show_plot:
            plt.show()
        
        plt.close() # Chiude la figura per liberare memoria


def compute_pinn_loss(model, x_data, y_data, x_bc=None, y_bc=None, physics_loss_fn=None, x_physics=None, ic_loss_fn=None, lambda_data=1.0, lambda_bc=1.0, lambda_physics=1.0, **kwargs):
    """
    Calcola le componenti della loss per una PINN in modo generico.
    
    Args:
        model: Il modello PyTorch.
        x_data: Input dei dati di training (punti interni / supervisionati).
        y_data: Target dei dati di training.
        x_bc: Input dei dati al contorno (Boundary Conditions).
        y_bc: Target dei dati al contorno.
        physics_loss_fn: Funzione che accetta (model, x_physics) e restituisce la loss sulla PDE.
        x_physics: Punti di collocazione per la loss fisica.
        ic_loss_fn: Funzione opzionale per condizioni iniziali, accetta (model).
        lambda_data: Peso per la data loss (interna).
        lambda_bc: Peso per la boundary loss.
        lambda_physics: Peso per la physics loss.
        **kwargs: Argomenti extra da passare alle funzioni di loss custom.
        
    Returns:
        total_loss: Somma delle loss.
        loss_dict: Dizionario con i dettagli {'total_loss': ..., 'data_loss': ..., ...}.
    """
    loss_dict = {}
    total_loss = 0.0
    mse_loss = nn.MSELoss()
    
    # 1. Data Loss (Internal Points)
    if x_data is not None and y_data is not None:
        y_pred = model(x_data)
        data_loss = mse_loss(y_pred, y_data)
        loss_dict['data_loss'] = data_loss
        total_loss += lambda_data * data_loss

    # 2. BC Loss (Boundary Points) - Explicitly separated
    if x_bc is not None and y_bc is not None:
        bc_pred = model(x_bc)
        bc_loss_val = mse_loss(bc_pred, y_bc)
        loss_dict['bc_loss'] = bc_loss_val
        total_loss += lambda_bc * bc_loss_val
    
    # 3. Physics Loss (PDE)
    if physics_loss_fn is not None:
        if x_physics is not None:
            # Se x_physics richiede gradiente per differenziazione automatica
            if not x_physics.requires_grad:
                x_physics.requires_grad_(True)
            pde_loss = physics_loss_fn(model, x_physics, **kwargs)
        else:
            # Alcune loss fisiche potrebbero non richiedere x_physics esplicito o gestirlo internamente
            pde_loss = physics_loss_fn(model, **kwargs)
            
        loss_dict['pde_loss'] = pde_loss
        total_loss += lambda_physics * pde_loss
        
    # 4. IC Loss (Initial Conditions - optional/legacy)
    if ic_loss_fn is not None:
        ic_loss = ic_loss_fn(model, **kwargs)
        loss_dict['ic_loss'] = ic_loss
        total_loss += ic_loss
        
    loss_dict['total_loss'] = total_loss
    
    return total_loss, loss_dict

