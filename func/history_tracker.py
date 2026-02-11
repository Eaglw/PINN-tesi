import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import numpy as np

class TrainingHistory:
    """
    Una classe per registrare e visualizzare l'andamento delle loss durante il training.
    """
    def __init__(self):
        self.epochs = []
        self.losses = {} # Dizionario di liste: {'total_loss': [v1, v2...], 'pde_loss': [None, ..., v100...]}

    def update(self, epoch, loss_dict):
        """
        Registra i valori delle loss per un dato'epoch'.
        Gestisce loss opzionali (es. pde_loss durante warmup) mantenendo le liste allineate.

        Args:
            epoch (int): L'epoch corrente.
            loss_dict (dict): Un dizionario con i nomi delle loss e i loro valori.
                              Es: {'total_loss': 1.5, 'pde_loss': 1.2, ...}
        """
        self.epochs.append(epoch)
        
        # 1. Identifica tutte le chiavi di loss viste finora (nel dizionario storico o nel corrente)
        current_keys = set(loss_dict.keys())
        known_keys = set(self.losses.keys())
        all_keys = current_keys.union(known_keys)
        
        for name in all_keys:
            # Inizializza la lista se è una nuova chiave
            if name not in self.losses:
                # Se la chiave appare per la prima volta ma siamo già avanti col training (es. epoch > 0),
                # dobbiamo riempire il passato con None per allinearci a len(self.epochs) - 1
                self.losses[name] = [None] * (len(self.epochs) - 1)
            
            # Estrai il valore corrente o usa None se manca in questo step
            if name in loss_dict:
                val = loss_dict[name]
                val = val.item() if hasattr(val, 'item') else val
            else:
                val = None
            
            self.losses[name].append(val)

    def plot_losses(self, warmup_epoch=0, adam_epochs=None, save_path=None, experiment_name="", show_plot=True):
        """
        Genera un grafico con l'andamento di tutte le loss registrate.
        Divide il grafico in due subplot se viene fornito adam_epochs, separando la fase Adam da L-BFGS.
        """
        # Se non fornito, proviamo a dedurlo o trattiamo tutto come fase unica
        has_lbfgs = adam_epochs is not None and any(e >= adam_epochs for e in self.epochs)
        
        if has_lbfgs:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [3, 1]})
            fig.subplots_adjust(wspace=0.3)
        else:
            plt.figure(figsize=(8, 4))
            ax1 = plt.gca()
            ax2 = None

        def plot_on_ax(ax, epoch_range_indices, title_suffix=""):
            for name, values in self.losses.items():
                if name.startswith('grad_') or name.startswith('weight_'): continue
                
                # Prendi solo il range richiesto
                r_epochs = [self.epochs[i] for i in epoch_range_indices]
                r_values = [values[i] if values[i] is not None else np.nan for i in epoch_range_indices]
                
                # Se tutti i valori sono NaN, salta
                if all(np.isnan(r_values)): continue

                # Specifica nel label se la loss è pesata o pura
                if name == "total_loss":
                    label = f"{name} (weighted)"
                    linewidth = 3
                else:
                    label = f"{name} (pure)"
                    linewidth = 1.5

                ax.plot(r_epochs, r_values, linewidth=linewidth, label=label)
            
            ax.set_title(f'Loss {title_suffix}')
            ax.set_xlabel('Epoch/Iter')
            ax.set_ylabel('Loss')
            ax.set_yscale('log')
            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        if has_lbfgs:
            # Adam: indici dove epoch < adam_epochs
            adam_indices = [i for i, e in enumerate(self.epochs) if e < adam_epochs]
            # Aggiungiamo il primo punto di L-BFGS per dare continuità visiva se possibile
            lbfgs_indices = [i for i, e in enumerate(self.epochs) if e >= adam_epochs]
            
            if adam_indices:
                plot_on_ax(ax1, adam_indices, "(Adam Phase)")
                if warmup_epoch != 0:
                    ax1.axvline(warmup_epoch, color="r", linestyle="--", label="End Warmup")
                ax1.legend(loc='upper right', frameon=False, fontsize="small")

            if lbfgs_indices:
                # Per continuità, includiamo l'ultimo punto di Adam nel plot L-BFGS
                if adam_indices:
                    lbfgs_plot_indices = [adam_indices[-1]] + lbfgs_indices
                else:
                    lbfgs_plot_indices = lbfgs_indices
                
                plot_on_ax(ax2, lbfgs_plot_indices, "(L-BFGS Refinement)")
                ax2.set_xlabel('Iter')
        else:
            # Plot standard
            plot_on_ax(ax1, range(len(self.epochs)), f"- {experiment_name}")
            if warmup_epoch != 0:
                ax1.axvline(warmup_epoch, color="r", linestyle="--", label="End Warmup")
            ax1.legend(loc='upper right', frameon=False, fontsize="large")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Grafico delle loss salvato in: {save_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()

    def plot_gradients(self, save_path=None, experiment_name="", show_plot=True):
        """
        Genera un grafico con l'andamento delle norme dei gradienti.
        """
        grad_keys = [k for k in self.losses.keys() if k.startswith('grad_')]
        if not grad_keys:
            print("No gradient history found to plot.")
            return

        plt.figure(figsize=(8, 4))
        ax = plt.gca()
        
        for name in grad_keys:
            # Filter None/NaN
            values = self.losses[name]
            clean_values = [v if v is not None else np.nan for v in values]
            
            # Check if we have enough data (gradient logging might be sparse)
            valid_indices = [i for i, v in enumerate(clean_values) if not np.isnan(v)]
            valid_epochs = [self.epochs[i] for i in valid_indices]
            valid_vals = [clean_values[i] for i in valid_indices]
            
            if valid_vals:
                plt.plot(valid_epochs, valid_vals, label=name, marker='o', markersize=2)
        
        plt.title(f'Gradient Norms - {experiment_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        l = plt.legend(loc='upper right', frameon=False, fontsize="large")
        plt.setp(l.get_texts(), color="k")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Grafico dei gradienti salvato in: {save_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()

    def plot_weights(self, save_path=None, experiment_name="", show_plot=True):
        """
        Genera un grafico con l'andamento dei pesi delle loss (lambda).
        """
        weight_keys = [k for k in self.losses.keys() if k.startswith('weight_')]
        if not weight_keys:
            return

        plt.figure(figsize=(8, 4))
        ax = plt.gca()
        
        for name in weight_keys:
            values = self.losses[name]
            clean_values = [v if v is not None else np.nan for v in values]
            
            # Filtriamo i punti validi
            valid_indices = [i for i, v in enumerate(clean_values) if not np.isnan(v)]
            valid_epochs = [self.epochs[i] for i in valid_indices]
            valid_vals = [clean_values[i] for i in valid_indices]
            
            if valid_vals:
                plt.plot(valid_epochs, valid_vals, label=name.replace('weight_', 'lambda_'), linewidth=2)
        
        plt.title(f'Evolution of Loss Weights - {experiment_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Weight Value')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        l = plt.legend(loc='best', frameon=False, fontsize="large")
        plt.setp(l.get_texts(), color="k")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Grafico dei pesi salvato in: {save_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()



def compute_pinn_loss(model, x_data, y_data, x_bc=None, y_bc=None, physics_loss_fn=None, x_physics=None, ic_loss_fn=None, physics_problem=None, lambda_data=1.0, lambda_bc=1.0, lambda_physics=1.0, **kwargs):
    """
    Computes the components of the PINN loss.
    
    IMPORTANT: 
    - Individual components (data_loss, bc_loss, pde_loss) in 'loss_dict' are PURE residuals (UNWEIGHTED).
    - 'total_loss' is the WEIGHTED sum: lambda_data*data_loss + lambda_bc*bc_loss + lambda_physics*pde_loss.
    
    This ensures that physics residuals can be monitored independently of their contribution to the gradient.
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

    # 2. BC Loss (Boundary Points)
    if physics_problem is not None and x_bc is not None and y_bc is not None:
        # Use modular physics for BC loss
        bc_loss_val = physics_problem.boundary_loss(model, x_bc, y_bc)
        loss_dict['bc_loss'] = bc_loss_val
        total_loss += lambda_bc * bc_loss_val
    elif x_bc is not None and y_bc is not None:
        # Legacy MSE BC loss
        bc_pred = model(x_bc)
        bc_loss_val = mse_loss(bc_pred, y_bc)
        loss_dict['bc_loss'] = bc_loss_val
        total_loss += lambda_bc * bc_loss_val
    
    # 3. Physics Loss (PDE)
    if physics_problem is not None and x_physics is not None:
        pde_loss = physics_problem.residual(model, x_physics)
        loss_dict['pde_loss'] = pde_loss
        total_loss += lambda_physics * pde_loss
    elif physics_loss_fn is not None:
        if x_physics is not None:
            if not x_physics.requires_grad:
                x_physics.requires_grad_(True)
            pde_loss = physics_loss_fn(model, x_physics, **kwargs)
        else:
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

