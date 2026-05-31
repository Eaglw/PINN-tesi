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
        self.lr_history = []

    def extend(self, other):
        """
        Concatena un'altra TrainingHistory a questa.
        Gestisce l'offset delle epoche per rendere la sequenza continua.
        """
        if not other.epochs:
            return
            
        last_epoch = self.epochs[-1] if self.epochs else -1
        # Offset delle epoche per farle seguire l'ultima registrata
        self.epochs.extend([e + last_epoch + 1 for e in other.epochs])
        self.lr_history.extend(other.lr_history)
        
        # Sincronizza le chiavi di tutte le loss
        all_keys = set(self.losses.keys()).union(set(other.losses.keys()))
        current_len_before = len(self.epochs) - len(other.epochs)
        
        for name in all_keys:
            if name not in self.losses:
                # Se la chiave è nuova per 'self', riempiamo il passato con None
                self.losses[name] = [None] * current_len_before
            
            if name in other.losses:
                self.losses[name].extend(other.losses[name])
            else:
                # Se la chiave manca in 'other', riempiamo la nuova sezione con None
                self.losses[name].extend([None] * len(other.epochs))

    def update(self, epoch, loss_dict, lr=None):
        """
        Registra i valori delle loss per un dato 'epoch'.
        """
        self.epochs.append(epoch)
        
        # Gestione Learning Rate: assicuriamoci che sia un float o None
        if lr is not None:
            lr = lr.item() if hasattr(lr, "item") else lr
        elif 'lr' in loss_dict:
            lr = loss_dict['lr']
            lr = lr.item() if hasattr(lr, 'item') else lr
        self.lr_history.append(lr)
        
        # 1. Identifica tutte le chiavi di loss viste finora
        current_keys = set(loss_dict.keys())
        known_keys = set(self.losses.keys())
        all_keys = current_keys.union(known_keys)
        
        for name in all_keys:
            if name not in self.losses:
                self.losses[name] = [None] * (len(self.epochs) - 1)
            
            if name in loss_dict:
                val = loss_dict[name]
                val = val.item() if hasattr(val, 'item') else val
            else:
                val = None
            
            self.losses[name].append(val)

    def plot_losses(self, warmup_epoch=0, adam_epochs=None, save_path=None, experiment_name="", show_plot=False, skip_epochs=0, phase_markers=None, smoothing_alpha=0.0, active_loss_keys=None):
        """
        Genera un grafico con l'andamento di tutte le loss registrate.
        
        Arguments:
            skip_epochs: Numero di epoche iniziali da non visualizzare nel grafico.
            phase_markers: Lista di dict [{'epoch': N, 'label': 'Fase 2', 'color': 'purple'}]
                per disegnare linee verticali ai cambi di fase (es. Staged Training).
            smoothing_alpha: Float tra 0 e 1. Se > 0, sovrappone una curva EMA smoothed
                alle loss per rendere il trend leggibile. 0 = nessuno smoothing.
            active_loss_keys: Set di chiavi loss che hanno peso > 0. Se fornito,
                le loss non presenti vengono escluse dalla visualizzazione.
        """
        has_lbfgs = adam_epochs is not None and any(e >= adam_epochs for e in self.epochs)
        
        if has_lbfgs:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [3, 1]})
            fig.subplots_adjust(wspace=0.3)
        else:
            plt.figure(figsize=(8, 4))
            ax1 = plt.gca()
            ax2 = None

        def plot_on_ax(ax, epoch_range_indices, title_suffix=""):
            # Filtro per skip_epochs
            epoch_range_indices = [i for i in epoch_range_indices if self.epochs[i] >= skip_epochs]
            if not epoch_range_indices: return

            for name, values in self.losses.items():
                if name.startswith('grad_') or name.startswith('weight_'): continue
                
                # Filtra loss con peso 0 (es. data_loss in PurePhys)
                if active_loss_keys is not None and name != 'total_loss':
                    # Mappa il nome della loss alla chiave nel set
                    loss_key_map = {'data_loss': 'data', 'bc_loss': 'bc', 'pde_loss': 'physics'}
                    mapped_key = loss_key_map.get(name, name)
                    if mapped_key not in active_loss_keys:
                        continue
                
                r_epochs = [self.epochs[i] for i in epoch_range_indices]
                r_values = [values[i] if values[i] is not None else np.nan for i in epoch_range_indices]
                
                if all(np.isnan(r_values)): continue

                # Total loss pesata (linea spessa), componenti pure (linee sottili)
                if name == "total_loss":
                    label = f"{name} (weighted)"
                    linewidth = 2.5
                    alpha = 1.0
                else:
                    label = f"{name} (pure)"
                    linewidth = 1.2
                    alpha = 0.8

                line, = ax.plot(r_epochs, r_values, linewidth=linewidth, label=label, alpha=alpha)
                
                # Smoothing EMA overlay
                if smoothing_alpha > 0 and len(r_values) > 10:
                    ema = []
                    current = None
                    for v in r_values:
                        if np.isnan(v):
                            ema.append(np.nan)
                        elif current is None or np.isnan(current):
                            current = v
                            ema.append(v)
                        else:
                            current = smoothing_alpha * current + (1 - smoothing_alpha) * v
                            ema.append(current)
                    ax.plot(r_epochs, ema, linewidth=linewidth + 0.8, alpha=0.9, 
                            color=line.get_color(), linestyle='-')
            
            # Disegno linee verticali per i cambi di Learning Rate
            if len(self.lr_history) > 0:
                lr_changes = []
                for i in range(1, len(epoch_range_indices)):
                    idx_curr = epoch_range_indices[i]
                    idx_prev = epoch_range_indices[i-1]
                    
                    if idx_curr >= len(self.lr_history) or idx_prev >= len(self.lr_history):
                        continue
                        
                    lr_curr = self.lr_history[idx_curr]
                    lr_prev = self.lr_history[idx_prev]
                    
                    # Confronto robusto per cambi di LR (escludendo i None)
                    if lr_curr is not None and lr_prev is not None and not np.isclose(lr_curr, lr_prev, rtol=1e-8, atol=1e-12):
                        lr_changes.append(self.epochs[idx_curr])
                
                # Disegniamo i cambi di LR SOLO se sono eventi discreti (es. Plateau, StepLR)
                # Se il LR cambia ad ogni epoca (come nel Cosine Annealing), avremmo una linea nera per ogni epoca!
                if 0 < len(lr_changes) <= 50:
                    first_lr_vline = True
                    for ep in lr_changes:
                        label = "LR Change" if first_lr_vline else None
                        ax.axvline(ep, color="black", linestyle="--", alpha=0.3, linewidth=1, label=label)
                        first_lr_vline = False

            ax.set_title(f'Loss {title_suffix}')
            ax.set_xlabel('Epoch/Iter')
            ax.set_ylabel('Loss')
            ax.set_yscale('log')
            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Autoscale y-axis logic to avoid being squashed by initial spikes
            valid_vals = [v for v in r_values if not np.isnan(v) and v > 0]
            if valid_vals:
                vmin, vmax = min(valid_vals), max(valid_vals)
                if vmax / vmin > 1e6:
                    ax.set_ylim(vmin * 0.5, vmin * 1e5) # Focus on converged region
            
            # Phase markers (Staged Training)
            if phase_markers:
                for pm in phase_markers:
                    pm_epoch = pm.get('epoch', 0)
                    pm_label = pm.get('label', 'Phase Change')
                    pm_color = pm.get('color', 'purple')
                    # Solo se il marker è nel range visualizzato
                    displayed_epochs = [self.epochs[i] for i in epoch_range_indices]
                    if displayed_epochs and min(displayed_epochs) <= pm_epoch <= max(displayed_epochs):
                        ax.axvline(pm_epoch, color=pm_color, linestyle='-.', linewidth=1.5, alpha=0.7, label=pm_label)

        if has_lbfgs:
            adam_indices = [i for i, e in enumerate(self.epochs) if e < adam_epochs]
            lbfgs_indices = [i for i, e in enumerate(self.epochs) if e >= adam_epochs]
            
            if adam_indices:
                plot_on_ax(ax1, adam_indices, "(Adam Phase)")
                if warmup_epoch != 0 and warmup_epoch >= skip_epochs:
                    ax1.axvline(warmup_epoch, color="r", linestyle="--", label="End Warmup")
                ax1.legend(loc='upper right', frameon=False, fontsize="x-small")

            if lbfgs_indices:
                lbfgs_plot_indices = [adam_indices[-1]] + lbfgs_indices if adam_indices else lbfgs_indices
                plot_on_ax(ax2, lbfgs_plot_indices, "(L-BFGS Refinement)")
                ax2.set_xlabel('Iter')
        else:
            plot_on_ax(ax1, range(len(self.epochs)), f"- {experiment_name}")
            if warmup_epoch != 0 and warmup_epoch >= skip_epochs:
                ax1.axvline(warmup_epoch, color="r", linestyle="--", label="End Warmup")
            ax1.legend(loc='upper right', frameon=False, fontsize="small")

        if save_path:
            dir_name = os.path.dirname(save_path)
            if dir_name: os.makedirs(dir_name, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        
        if show_plot: plt.show()
        plt.close()

    def plot_gradients(self, save_path=None, experiment_name="", show_plot=False):
        grad_keys = [k for k in self.losses.keys() if k.startswith('grad_')]
        if not grad_keys: return

        plt.figure(figsize=(8, 4))
        for name in grad_keys:
            values = self.losses[name]
            clean_values = [v if v is not None else np.nan for v in values]
            valid_indices = [i for i, v in enumerate(clean_values) if not np.isnan(v)]
            if valid_indices:
                plt.plot([self.epochs[i] for i in valid_indices], [clean_values[i] for i in valid_indices], label=name, marker='o', markersize=2)
        
        plt.title(f'Gradient Norms - {experiment_name}')
        plt.yscale('log')
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend(loc='upper right', frameon=False)
        if save_path:
            dir_name = os.path.dirname(save_path)
            if dir_name: os.makedirs(dir_name, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        if show_plot: plt.show()
        plt.close()

    def plot_weights(self, save_path=None, experiment_name="", show_plot=False):
        weight_keys = [k for k in self.losses.keys() if k.startswith('weight_')]
        if not weight_keys: return

        plt.figure(figsize=(8, 4))
        for name in weight_keys:
            values = self.losses[name]
            clean_values = [v if v is not None else np.nan for v in values]
            valid_indices = [i for i, v in enumerate(clean_values) if not np.isnan(v)]
            if valid_indices:
                plt.plot([self.epochs[i] for i in valid_indices], [clean_values[i] for i in valid_indices], label=name.replace('weight_', 'lambda_'), linewidth=2)
        
        plt.title(f'Evolution of Loss Weights - {experiment_name}')
        plt.yscale('log')
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend(loc='best', frameon=False)
        if save_path:
            dir_name = os.path.dirname(save_path)
            if dir_name: os.makedirs(dir_name, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        if show_plot: plt.show()
        plt.close()

    def plot_physical_parameters(self, true_etas, true_etap, true_lam, true_epsilon=None, true_alpha=None, save_path=None, experiment_name="", show_plot=False):
        has_params = any(k in self.losses for k in ['param_etas', 'param_etap', 'param_lam', 'param_epsilon', 'param_alpha'])
        if not has_params:
            return

        active_params = []
        if 'param_etas' in self.losses: active_params.append({'key': 'param_etas', 'true_val': true_etas, 'title': r'Solvent Viscosity ($\eta_s$)', 'color': 'b'})
        if 'param_etap' in self.losses: active_params.append({'key': 'param_etap', 'true_val': true_etap, 'title': r'Polymer Viscosity ($\eta_p$)', 'color': 'g'})
        if 'param_lam' in self.losses: active_params.append({'key': 'param_lam', 'true_val': true_lam, 'title': r'Relaxation Time ($\lambda$)', 'color': 'r'})
        if 'param_epsilon' in self.losses: active_params.append({'key': 'param_epsilon', 'true_val': true_epsilon, 'title': r'PTT Mobility ($\epsilon$)', 'color': 'm'})
        if 'param_alpha' in self.losses: active_params.append({'key': 'param_alpha', 'true_val': true_alpha, 'title': r'Giesekus Mobility ($\alpha$)', 'color': 'c'})

        n_params = len(active_params)
        if n_params == 0:
            return

        fig, axs = plt.subplots(n_params, 1, figsize=(10, 4*n_params), sharex=True)
        if n_params == 1:
            axs = [axs]
            
        fig.suptitle(f'Inverse Problem: Physical Parameters Evolution\n{experiment_name}', fontsize=16)

        for i, config in enumerate(active_params):
            ax = axs[i]
            key = config['key']
            true_val = config['true_val']
            color = config['color']
            title = config['title']

            if key in self.losses:
                values = self.losses[key]
                clean_values = [v if v is not None else np.nan for v in values]
                valid_indices = [idx for idx, v in enumerate(clean_values) if not np.isnan(v)]
                
                if valid_indices:
                    epochs = [self.epochs[idx] for idx in valid_indices]
                    vals = [clean_values[idx] for idx in valid_indices]
                    
                    ax.plot(epochs, vals, label='Learned Value', color=color, linewidth=2)
            
            # Plot the true value as a dashed line
            if true_val is not None:
                ax.axhline(true_val, color='k', linestyle='--', linewidth=2, label='True Value')
            ax.set_title(title)
            ax.grid(True, ls="--", alpha=0.5)
            ax.legend(loc='best', frameon=True)

        axs[-1].set_xlabel('Epoch/Iter')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path:
            dir_name = os.path.dirname(save_path)
            if dir_name: os.makedirs(dir_name, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        if show_plot: plt.show()
        plt.close()




