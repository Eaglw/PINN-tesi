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
        Genera un grafico con l'andamento di tutte le loss registrate e dei parametri, suddiviso in più subplot verticali:
        1. Core Losses (total_loss, bc_loss, data_loss, pde_loss) in scala logaritmica.
        2. BC Components (loss_bc_Inlet, loss_bc_Walls, loss_bc_Outlet, ecc.) in scala logaritmica.
        3. Physical Parameters (param_etas, param_etap, param_lam, param_epsilon, param_alpha) in scala lineare.
        
        Se è presente L-BFGS, ogni riga ha due colonne (Adam e L-BFGS).
        """
        has_lbfgs = adam_epochs is not None and any(e >= adam_epochs for e in self.epochs)
        
        # Identifica i gruppi di chiavi presenti
        bc_keys = sorted([k for k in self.losses.keys() if k.startswith('loss_bc_')])
        param_keys = sorted([k for k in self.losses.keys() if k.startswith('param_')])
        
        rows = ['core']
        if bc_keys:
            rows.append('bc_comp')
        # Rimossa l'evoluzione dei parametri fisici dal grafico delle loss per evitare ridondanze.
        # if param_keys:
        #     rows.append('params')
            
        num_rows = len(rows)
        num_cols = 2 if has_lbfgs else 1
        
        # Creiamo la figura con subplot verticali condivisi sull'asse x per ciascuna colonna
        gridspec_kw = {'width_ratios': [3, 1]} if num_cols == 2 else None
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12 if num_cols == 2 else 8, 3.5 * num_rows), 
                                gridspec_kw=gridspec_kw, sharex='col', squeeze=False)
        fig.subplots_adjust(hspace=0.25, wspace=0.25)
        
        label_mapping = {
            'total_loss': 'Total Loss (weighted)',
            'bc_loss': 'BC Loss (total)',
            'data_loss': 'Data Loss (pure)',
            'pde_loss': 'PDE Loss (pure)',
            'param_etas': r'Solvent Viscosity ($\eta_s$)',
            'param_etap': r'Polymer Viscosity ($\eta_p$)',
            'param_lam': r'Relaxation Time ($\lambda$)',
            'param_epsilon': r'PTT Mobility ($\epsilon$)',
            'param_alpha': r'Giesekus Mobility ($\alpha$)'
        }

        def plot_on_ax(ax, epoch_range_indices, row_type, col_idx, is_bottom_row):
            # Filtro per skip_epochs
            epoch_range_indices = [i for i in epoch_range_indices if self.epochs[i] >= skip_epochs]
            if not epoch_range_indices: return

            # Selezioniamo le chiavi da plottare in base alla riga
            if row_type == 'core':
                keys_to_plot = ['total_loss', 'bc_loss', 'data_loss', 'pde_loss']
                # Se ci sono altre chiavi generiche, includiamole qui
                for k in self.losses.keys():
                    if k not in keys_to_plot and not k.startswith('loss_bc_') and not k.startswith('param_') and not k.startswith('grad_') and not k.startswith('weight_'):
                        keys_to_plot.append(k)
            elif row_type == 'bc_comp':
                keys_to_plot = bc_keys
            elif row_type == 'params':
                keys_to_plot = param_keys
            else:
                keys_to_plot = []

            r_epochs = [self.epochs[i] for i in epoch_range_indices]
            plotted_any = False

            for name in keys_to_plot:
                if name not in self.losses: continue
                
                # Filtra loss con peso 0 se fornito active_loss_keys
                if row_type == 'core' and active_loss_keys is not None and name != 'total_loss':
                    loss_key_map = {'data_loss': 'data', 'bc_loss': 'bc', 'pde_loss': 'physics'}
                    mapped_key = loss_key_map.get(name, name)
                    if mapped_key not in active_loss_keys:
                        continue
                
                values = self.losses[name]
                r_values = [values[i] if values[i] is not None else np.nan for i in epoch_range_indices]
                
                if all(np.isnan(r_values)): continue
                plotted_any = True

                # Label formatting
                if name in label_mapping:
                    label = label_mapping[name]
                elif name.startswith('loss_bc_'):
                    label = name.replace('loss_bc_', 'BC ')
                else:
                    label = name

                # Total loss pesata (linea spessa), componenti pure (linee sottili)
                if name == "total_loss":
                    linewidth = 2.2
                    alpha = 1.0
                else:
                    linewidth = 1.2
                    alpha = 0.85

                line, = ax.plot(r_epochs, r_values, linewidth=linewidth, label=label, alpha=alpha)
                
                # Smoothing EMA overlay (solo per le loss, non per i parametri fisici)
                if row_type != 'params' and smoothing_alpha > 0 and len(r_values) > 10:
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
            
            # Disegno linee verticali per i cambi di Learning Rate (solo sul core plot per non affollare)
            if row_type == 'core' and len(self.lr_history) > 0:
                lr_changes = []
                for i in range(1, len(epoch_range_indices)):
                    idx_curr = epoch_range_indices[i]
                    idx_prev = epoch_range_indices[i-1]
                    
                    if idx_curr >= len(self.lr_history) or idx_prev >= len(self.lr_history):
                        continue
                        
                    lr_curr = self.lr_history[idx_curr]
                    lr_prev = self.lr_history[idx_prev]
                    
                    if lr_curr is not None and lr_prev is not None and not np.isclose(lr_curr, lr_prev, rtol=1e-8, atol=1e-12):
                        lr_changes.append(self.epochs[idx_curr])
                
                if 0 < len(lr_changes) <= 50:
                    first_lr_vline = True
                    for ep in lr_changes:
                        label = "LR Change" if first_lr_vline else None
                        ax.axvline(ep, color="black", linestyle="--", alpha=0.3, linewidth=1, label=label)
                        first_lr_vline = False

            # Titolo del subplot
            if row_type == 'core':
                title_base = "Core Losses"
            elif row_type == 'bc_comp':
                title_base = "BC Components Loss"
            elif row_type == 'params':
                title_base = "Physical Parameters"
            else:
                title_base = "Loss"
                
            if num_cols == 2:
                title_suffix = " (Adam Phase)" if col_idx == 0 else " (L-BFGS)"
                ax.set_title(title_base + title_suffix, fontsize=10)
            else:
                ax.set_title(f"{title_base} - {experiment_name}", fontsize=11)
                
            if is_bottom_row:
                ax.set_xlabel('Epoch/Iter' if col_idx == 0 else 'Iter')
                
            if row_type == 'params':
                ax.set_ylabel('Value')
                ax.set_yscale('linear')
            else:
                ax.set_ylabel('Loss')
                ax.set_yscale('log')
                
            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Autoscale y-axis logic per i plot in scala logaritmica (evita che i picchi iniziali schiaccino tutto)
            if row_type != 'params' and plotted_any:
                all_plotted_vals = []
                for name in keys_to_plot:
                    if name in self.losses:
                        # Consideriamo solo i valori all'interno degli indici delle epoche plottate su questo asse
                        vals_in_range = [self.losses[name][i] for i in epoch_range_indices]
                        all_plotted_vals.extend([v for v in vals_in_range if v is not None and not np.isnan(v) and v > 0])
                if all_plotted_vals:
                    vmin, vmax = min(all_plotted_vals), max(all_plotted_vals)
                    # Impostiamo i limiti reali per mostrare TUTTO il grafico in scala logaritmica
                    ax.set_ylim(vmin * 0.5, vmax * 2.0)
            
            # Phase markers e warmup (disegnati su tutti i subplot per allineamento verticale)
            if phase_markers:
                for pm in phase_markers:
                    pm_epoch = pm.get('epoch', 0)
                    pm_label = pm.get('label', 'Phase Change')
                    pm_color = pm.get('color', 'purple')
                    displayed_epochs = [self.epochs[i] for i in epoch_range_indices]
                    if displayed_epochs and min(displayed_epochs) <= pm_epoch <= max(displayed_epochs):
                        ax.axvline(pm_epoch, color=pm_color, linestyle='-.', linewidth=1.5, alpha=0.7, label=pm_label)
                        
            if warmup_epoch != 0 and warmup_epoch >= skip_epochs:
                displayed_epochs = [self.epochs[i] for i in epoch_range_indices]
                if displayed_epochs and min(displayed_epochs) <= warmup_epoch <= max(displayed_epochs):
                    ax.axvline(warmup_epoch, color="r", linestyle="--", label="End Warmup")
                    
            if plotted_any:
                ax.legend(loc='upper right', frameon=False, fontsize="x-small" if num_cols == 2 else "small")

        # Popoliamo i subplot per ciascuna riga
        for r_idx, row_type in enumerate(rows):
            is_bottom_row = (r_idx == num_rows - 1)
            if has_lbfgs:
                adam_indices = [i for i, e in enumerate(self.epochs) if e < adam_epochs]
                lbfgs_indices = [i for i, e in enumerate(self.epochs) if e >= adam_epochs]
                
                if adam_indices:
                    plot_on_ax(axs[r_idx, 0], adam_indices, row_type, col_idx=0, is_bottom_row=is_bottom_row)
                if lbfgs_indices:
                    # includiamo l'ultimo di adam per continuazione visiva
                    lbfgs_plot_indices = [adam_indices[-1]] + lbfgs_indices if adam_indices else lbfgs_indices
                    plot_on_ax(axs[r_idx, 1], lbfgs_plot_indices, row_type, col_idx=1, is_bottom_row=is_bottom_row)
            else:
                plot_on_ax(axs[r_idx, 0], range(len(self.epochs)), row_type, col_idx=0, is_bottom_row=is_bottom_row)

        if save_path:
            dir_name = os.path.dirname(save_path)
            if dir_name: os.makedirs(dir_name, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
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




