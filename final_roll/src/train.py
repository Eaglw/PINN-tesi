import os
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils import convert_to_fp64, convert_to_fp32, get_optimal_chunk_size
from src.physics import compute_l2_errors

class SimpleHistory:
    """Tracker minimale per loss e parametri."""
    def __init__(self):
        self.epochs = []
        self.losses = {}

    def state_dict(self):
        return {
            'epochs': self.epochs,
            'losses': self.losses
        }

    def load_state_dict(self, state_dict):
        self.epochs = state_dict['epochs']
        self.losses = state_dict['losses']

    def update(self, epoch, loss_dict):
        self.epochs.append(epoch)
        for k, v in loss_dict.items():
            if k not in self.losses:
                self.losses[k] = [None] * (len(self.epochs) - 1)
            self.losses[k].append(v.item() if isinstance(v, torch.Tensor) else v)
        for k in self.losses:
            if k not in loss_dict:
                self.losses[k].append(None)

    def plot_losses(self, save_path):
        """Plot loss totale/data/bc/pde/momentum/constitutive."""
        fig, ax = plt.subplots(figsize=(10, 5))
        keys_plot = ['total', 'data', 'bc', 'pde', 'loss_momentum', 'loss_constitutive']
        colors = {
            'total': 'black',
            'data': 'blue',
            'bc': 'green',
            'pde': 'red',
            'loss_momentum': 'purple',
            'loss_constitutive': 'orange'
        }
        for k in keys_plot:
            if k not in self.losses:
                continue
            vals = self.losses[k]
            valid = [(e, v) for e, v in zip(self.epochs, vals) if v is not None and v > 0]
            if valid:
                ep, vv = zip(*valid)
                lw = 2.0 if k == 'total' else 1.2
                ax.plot(ep, vv, label=k, color=colors.get(k, None), linewidth=lw, alpha=0.85)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch / Iter')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss History (4rollmill)')
        ax.legend()
        ax.grid(True, ls='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def plot_params(self, save_path):
        """Plot evoluzione parametri fisici."""
        param_keys = [('param_beta', getattr(builtins, "BETA_TRUE", MU_S_TRUE / (MU_S_TRUE + MU_P_TRUE)), r'$\beta = \frac{\eta_s}{\eta_s + \eta_p}$'),
                      ('param_mu_s', MU_S_TRUE, r'$\eta_s$'),
                      ('param_mu_p', MU_P_TRUE, r'$\eta_p$'),
                      ('param_lam', LAM_TRUE, r'$\lambda$'),
                      ('param_eps', EPS_TRUE, r'$\epsilon$'),
                      ('param_alpha', ALPHA_TRUE, r'$\alpha$')]

        active = [(k, t, l) for k, t, l in param_keys if k in self.losses]
        if not active:
            return
        fig, axs = plt.subplots(len(active), 1, figsize=(10, 3.5 * len(active)), sharex=True)
        if len(active) == 1:
            axs = [axs]

        for ax, (k, true_val, label) in zip(axs, active):
            vals = self.losses[k]
            valid = [(e, v) for e, v in zip(self.epochs, vals) if v is not None]
            if valid:
                ep, vv = zip(*valid)
                ax.plot(ep, vv, linewidth=2, label='Learned')

                # Se i valori variano pochissimo (rumore FP32), imposta ylimits intorno al valore vero
                min_v, max_v = min(vv), max(vv)
                if abs(max_v - min_v) < 1e-5:
                    if abs(true_val) > 1e-5:
                        ax.set_ylim(true_val * 0.8, true_val * 1.2)
                    else:
                        ax.set_ylim(-0.02, 0.1)

            ax.axhline(true_val, color='k', linestyle='--', linewidth=2, label='True')
            ax.yaxis.get_major_formatter().set_useOffset(False)
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.set_title(label)
            ax.grid(True, ls='--', alpha=0.5)
            ax.legend()

        axs[-1].set_xlabel('Epoch / Iter')
        fig.suptitle(r'Physical Parameters Evolution ($\beta = \frac{\eta_s}{\eta_s + \eta_p}$)', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=150)
        plt.close()

    def plot_l2_errors(self, save_path):
        """Plot evoluzione errori L2 globali e mascherati."""
        fig, ax = plt.subplots(figsize=(10, 5))
        keys_plot = ['l2_u', 'l2_v', 'l2_p', 'l2_tau_xx', 'l2_tau_xy', 'l2_tau_yy', 'l2_tau_xx_masked', 'l2_tau_xy_masked', 'l2_tau_yy_masked']
        colors = {
            'l2_u': 'blue',
            'l2_v': 'deepskyblue',
            'l2_p': 'green',
            'l2_tau_xx': 'red',
            'l2_tau_xy': 'orange',
            'l2_tau_yy': 'purple',
            'l2_tau_xx_masked': 'brown',
            'l2_tau_xy_masked': 'magenta',
            'l2_tau_yy_masked': 'cyan'
        }
        for k in keys_plot:
            if k not in self.losses:
                continue
            vals = self.losses[k]
            valid = [(e, v) for e, v in zip(self.epochs, vals) if v is not None]
            if valid:
                ep, vv = zip(*valid)
                linestyle = '--' if 'masked' in k else '-'
                label = k.replace('l2_', '')
                ax.plot(ep, vv, label=label, color=colors.get(k, None), linestyle=linestyle, alpha=0.85)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch / Iter')
        ax.set_ylabel('L2 Relative Error')
        ax.set_title('L2 Relative Error History (Global & Masked Stress)')
        ax.legend()
        ax.grid(True, ls='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()


class FCN(nn.Module):
    """Fully Connected Network."""

    def __init__(self, n_input, n_output, hidden_layers):
        super().__init__()
        layers_sizes = [n_input] + hidden_layers + [n_output]

        layers = []
        for i in range(len(layers_sizes) - 1):
            layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
            # Inseriamo l'attivazione globale ovunque tranne che nell'ultimo layer
            if i < len(layers_sizes) - 2:
                layers.append(ACTIVATION())

        # Delegazione del forward a nn.Sequential
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CombinedModel(nn.Module):
    """Combina psi (1), p (1), tau (3) in un unico output con scaling di pressione e stress."""

    def __init__(self, p_scale=1.0, tau_scale=1.0):
        super().__init__()
        # Utilizzo della variabile globale HIDDEN_LAYERS
        self.model_psi = FCN(2, 1, HIDDEN_LAYERS)
        self.model_p = FCN(2, 1, HIDDEN_LAYERS)
        self.model_tau = FCN(2, 3, HIDDEN_LAYERS)

        self.p_scale = p_scale
        self.tau_scale = tau_scale

    def forward(self, x):
        psi = self.model_psi(x)
        p = self.model_p(x) * self.p_scale
        tau = self.model_tau(x) * self.tau_scale
        return torch.cat([psi, p, tau], dim=1)


def initialize_last_layer_zero(model):
    """Azzera l'ultimo layer lineare della rete."""
    last_linear = None
    # Esplora tutti i sottomoduli; l'ultimo che sovrascrive last_linear sarà quello finale
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module

    if last_linear is not None:
        nn.init.zeros_(last_linear.weight)
        if last_linear.bias is not None:
            nn.init.zeros_(last_linear.bias)


def init_weights_xavier(m, activation_name="tanh"):
    """Inizializzazione dei pesi Xavier Normal dinamica basata sull'attivazione."""
    if isinstance(m, nn.Linear):
        # Gestisce classi di attivazione convertendole in stringa
        if not isinstance(activation_name, str):
            activation_name = activation_name.__name__
        
        activation_name = activation_name.lower()
        if activation_name == 'silu':
            activation_name = 'relu'
            
        gain = nn.init.calculate_gain(activation_name)
        nn.init.xavier_normal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def precompute_stress_divergence(model, physics, points):
    """Precalcola la divergenza del tensore degli sforzi tau per model_tau congelato."""
    model.eval()
    chunk_size = 8192
    div_tau_x_list = []
    div_tau_y_list = []
    for i in range(0, points.shape[0], chunk_size):
        xc = points[i : i + chunk_size]
        xph = xc.clone().requires_grad_(True)
        out = model(xph)
        tau = out[:, 2:5]
        tau_xx, tau_xy, tau_yy = tau[:, 0:1], tau[:, 1:2], tau[:, 2:3]
        g_txx = physics._grad(tau_xx, xph, create_graph=False)
        tau_xx_x, tau_xx_y = g_txx[:, 0:1], g_txx[:, 1:2]
        g_txy = physics._grad(tau_xy, xph, create_graph=False)
        tau_xy_x, tau_xy_y = g_txy[:, 0:1], g_txy[:, 1:2]
        g_tyy = physics._grad(tau_yy, xph, create_graph=False)
        tau_yy_x, tau_yy_y = g_tyy[:, 0:1], g_tyy[:, 1:2]
        dt_x = (tau_xx_x + tau_xy_y).detach()
        dt_y = (tau_xy_x + tau_yy_y).detach()
        div_tau_x_list.append(dt_x)
        div_tau_y_list.append(dt_y)
    div_tau_x = torch.cat(div_tau_x_list, dim=0)
    div_tau_y = torch.cat(div_tau_y_list, dim=0)
    return div_tau_x, div_tau_y

def train(model, physics, data, resume_checkpoint=None, save_dir=None, tb_writer=None):
    """
    Training completo PINN: Adam Phase 1 (FP32) -> L-BFGS Phase 1 (FP64) -> Adam Phase 2 (FP32) -> L-BFGS Phase 2 (FP64).
    Implementazione ottimizzata per il contenimento della VRAM.
    """
    global CHUNK_SIZE_ADAM, CHUNK_SIZE_LBFGS
    history = SimpleHistory()

    start_epoch = 0
    loaded_opt_state = None
    loaded_sch_state = None

    if resume_checkpoint is not None:
        if os.path.exists(resume_checkpoint):
            print(f"\n[Checkpoint] Caricamento da: {resume_checkpoint}")
            chk = torch.load(resume_checkpoint, map_location=DEVICE)
            
            model.load_state_dict(chk['model_state_dict'])
            physics.load_state_dict(chk['physics_state_dict'], strict=False)
            history.load_state_dict(chk['history_state_dict'])
            
            loaded_opt_state = chk.get('optimizer_state_dict', None)
            loaded_sch_state = chk.get('scheduler_state_dict', None)
            
            start_epoch = chk.get('epoch', 0) + 1
            print(f"[Checkpoint] Ripresa dall'epoca {start_epoch}")
        else:
            print(f"\n[ATTENZIONE] Il file di checkpoint specificato NON ESISTE: {resume_checkpoint}")
            print("Il training ripartirà da zero!")

    # --- ESTRAZIONE DATI ---
    xy_all = data["coords"]
    uv_all = data["uv_data"]
    var_w = data["var_weights"]
    bc_data = data["boundary_groups"]

    # --- DEFINIZIONE DEI LIMITI DELLE FASI ---
    end_adam1 = ADAM_EPOCHS_PHASE1
    iters_ph1 = int(LBFGS_MAX_ITERS_PHASE1) if USE_LBFGS_PHASE1 else 0
    end_lbfgs1 = end_adam1 + iters_ph1
    end_adam2 = end_lbfgs1 + ADAM_EPOCHS_PHASE2
    iters_ph2 = int(LBFGS_MAX_ITERS_PHASE2) if USE_LBFGS_PHASE2 else 0
    end_lbfgs2 = end_adam2 + iters_ph2

    # Motore di calcolo centralizzato con Accumulazione dei Gradienti in Chunk.
    def compute_and_backward_losses(
        active_bcs, w_mom, w_con, points, labels, is_lbfgs=False, chunk_size_override=None, frozen_velocity=False, precomputed_div_tau=None
    ):
        d_loss_accum, p_loss_accum, loss_m_val, loss_c_val = 0.0, 0.0, 0.0, 0.0
        
        # Selezione dinamica del chunk size (L-BFGS necessita di chunk più piccoli)
        if chunk_size_override is not None:
            chunk_size = chunk_size_override
        else:
            chunk_size = CHUNK_SIZE_LBFGS if is_lbfgs else CHUNK_SIZE_ADAM

        for i in range(0, points.shape[0], chunk_size):
            xc = points[i : i + chunk_size]
            yc = labels[i : i + chunk_size] if labels is not None else None
            w_chunk = xc.shape[0] / points.shape[0]

            # Estrazione cinematica UNIFICATA
            xph = xc.clone().requires_grad_(True)
            u, v, p, tau = physics.get_velocity(model, xph)
            
            chunk_total_loss = 0.0

            # --- 1. DATA LOSS ---
            if yc is not None:
                dl = physics.data_loss(u, v, yc, var_w)
                d_loss_accum += dl.item() * w_chunk
                chunk_total_loss = chunk_total_loss + W_DATA * dl * w_chunk

            # --- 2. PDE LOSS ---
            if precomputed_div_tau is not None:
                div_tau_x_c = precomputed_div_tau[0][i : i + chunk_size]
                div_tau_y_c = precomputed_div_tau[1][i : i + chunk_size]
                chunk_div_tau = (div_tau_x_c, div_tau_y_c)
            else:
                chunk_div_tau = None

            lm, lc = physics.compute_pde_losses(
                xph, u, v, p, tau, w_mom, w_con, frozen_velocity=frozen_velocity, precomputed_div_tau=chunk_div_tau
            )
            loss_m_val += lm.item() * w_chunk
            loss_c_val += lc.item() * w_chunk
            pl = (w_mom * lm) + (w_con * lc)

            p_loss_accum += pl.item() * w_chunk
            chunk_total_loss = chunk_total_loss + W_PHYSICS * pl * w_chunk
            
            # Unico backward per il chunk corrente: distrugge il grafo intermediario
            chunk_total_loss.backward()

        # --- 3. BOUNDARY LOSS ---
        # Le condizioni al contorno hanno solitamente pochi punti, non serve chunking
        b_loss = physics.boundary_loss(model, bc_data, var_w, active_bcs=active_bcs)
        b_loss_val = b_loss.item()
        (W_BC * b_loss).backward()

        tot_loss = (
            (W_DATA * d_loss_accum) + (W_BC * b_loss_val) + (W_PHYSICS * p_loss_accum)
        )
        return tot_loss, d_loss_accum, b_loss_val, p_loss_accum, loss_m_val, loss_c_val

    # ==================================================================
    # FASE 1 ADAM (Cinematica e Reologia)
    # ==================================================================
    if start_epoch < end_adam1:
        print(
            f"\n{'=' * 60}\nFASE 1 ADAM (Cinematica e Reologia): {ADAM_EPOCHS_PHASE1} epoche\n{'=' * 60}"
        )
        
        # Configura requires_grad per Fase 1
        if STAGED_TRAINING:
            # Fase 1: Cinematica e Reologia (Congela Pressione)
            for p in model.parameters():
                p.requires_grad = False
            for p in model.model_psi.parameters():
                p.requires_grad = True
            for p in model.model_tau.parameters():
                p.requires_grad = True
            active_bcs, w_mom, w_con, frozen_vel = ["u", "v", "tau_xx", "tau_xy", "tau_yy"], 0.0, W_CONSTITUTIVE, False
        else:
            for p in model.parameters():
                p.requires_grad = True
            active_bcs, w_mom, w_con, frozen_vel = None, W_MOMENTUM, W_CONSTITUTIVE, False

        # Configura parametri fisici in modalità inversa
        if physics.inverse_mode:
            trainable_names = ["lam", "beta", "eps", "alpha"]
            for pname in ["beta", "mu_s", "mu_p", "lam", "eps", "alpha"]:
                is_tr = (pname in trainable_names) and (start_epoch >= WARMUP_UNLOCK_EPOCH)
                physics.set_trainable(pname, is_tr)

        # Log configurazione Fase 1
        print(f"  Active BCs: {active_bcs}")
        print(f"  Momentum: {'SPENTA (w=0)' if w_mom == 0.0 else f'ACCESA (w={w_mom})'} | Constitutive: {'SPENTA (w=0)' if w_con == 0.0 else f'ACCESA (w={w_con})'}")
        if active_bcs is not None and "p" not in active_bcs:
            print("  Pressure Point BC: DISATTIVATA in Fase 1")

        # Calcola chunk size ottimale
        CHUNK_SIZE_ADAM = get_optimal_chunk_size(
            phase=1, model=model,
            test_closure=lambda c: compute_and_backward_losses(
                active_bcs=active_bcs, w_mom=w_mom, w_con=w_con,
                points=xy_all[:c], labels=(uv_all[:c] if uv_all is not None else None),
                is_lbfgs=False, chunk_size_override=c, frozen_velocity=frozen_vel
            )
        )

        # Costruisci l'ottimizzatore
        steps_rem = end_adam1 - start_epoch
        net_params = [p for p in model.parameters() if p.requires_grad]
        groups = [{"params": net_params, "lr": BASE_LR}]
        if physics.inverse_mode:
            beta_lr_mult = getattr(builtins, "BETA_LR_FACTOR", 1.0)
            other_phys = [p for n, p in physics.named_parameters() if n != "_raw_beta" and p.requires_grad]
            beta_phys = [p for n, p in physics.named_parameters() if n == "_raw_beta" and p.requires_grad]
            if other_phys:
                groups.append({"params": other_phys, "lr": BASE_LR * PARAM_LR_FACTOR})
            if beta_phys:
                groups.append({"params": beta_phys, "lr": BASE_LR * PARAM_LR_FACTOR * beta_lr_mult})
        optimizer = torch.optim.Adam(groups, eps=ADAM_EPS)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps_rem, 1), eta_min=1e-6)

        # Carica stato se ripreso da checkpoint
        opt_loaded = False
        if loaded_opt_state is not None:
            try:
                optimizer.load_state_dict(loaded_opt_state)
                print("[Checkpoint] Optimizer state ripristinato con successo per Adam Fase 1.")
                opt_loaded = True
            except Exception as e:
                print(f"[Checkpoint] Avviso: Impossibile ripristinare optimizer state: {e}")
        if loaded_sch_state is not None and opt_loaded:
            try:
                scheduler.load_state_dict(loaded_sch_state)
            except Exception:
                pass

        pbar = tqdm(range(start_epoch, end_adam1), desc="Adam Fase 1", mininterval=2.0)
        for epoch in pbar:
            # Sblocco parametri fisici (Warmup Problema Inverso)
            if physics.inverse_mode and epoch == WARMUP_UNLOCK_EPOCH and epoch > 0:
                print(f"\n  [Warmup Stage 1] Sblocco parametri fisici (epoca {epoch})")
                trainable_names = ["lam", "beta", "eps", "alpha"]
                for pname in trainable_names:
                    physics.set_trainable(pname, True)
                steps_rem = end_adam1 - epoch
                net_params = [p for p in model.parameters() if p.requires_grad]
                groups = [{"params": net_params, "lr": BASE_LR}]
                beta_lr_mult = getattr(builtins, "BETA_LR_FACTOR", 1.0)
                other_phys = [p for n, p in physics.named_parameters() if n != "_raw_beta" and p.requires_grad]
                beta_phys = [p for n, p in physics.named_parameters() if n == "_raw_beta" and p.requires_grad]
                if other_phys:
                    groups.append({"params": other_phys, "lr": BASE_LR * PARAM_LR_FACTOR})
                if beta_phys:
                    groups.append({"params": beta_phys, "lr": BASE_LR * PARAM_LR_FACTOR * beta_lr_mult})
                optimizer = torch.optim.Adam(groups, eps=ADAM_EPS)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps_rem, 1), eta_min=1e-6)

            model.train()
            optimizer.zero_grad(set_to_none=True)

            tot_loss, d_loss_accum, b_loss_val, p_loss_accum, loss_m_val, loss_c_val = (
                compute_and_backward_losses(
                    active_bcs, w_mom, w_con, xy_all, uv_all,
                    frozen_velocity=frozen_vel, precomputed_div_tau=None
                )
            )

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            if physics.inverse_mode:
                phys_clip = [p for p in physics.parameters() if p.requires_grad]
                if phys_clip:
                    torch.nn.utils.clip_grad_norm_(phys_clip, PARAM_CLIP_NORM)

            optimizer.step()
            scheduler.step()

            log_l2 = ((epoch + 1) % max(1, end_adam1 // 40) == 0) or (epoch == start_epoch) or ((epoch + 1) == end_adam1)
            log_loss = ((epoch + 1) % 10 == 0) or log_l2

            if log_loss:
                params = physics.log_params()
                loss_dict = {
                    "total": tot_loss,
                    "data": d_loss_accum,
                    "bc": b_loss_val,
                    "pde": p_loss_accum,
                    "loss_momentum": loss_m_val,
                    "loss_constitutive": loss_c_val,
                    "param_beta": params["beta"],
                    "param_mu_s": params["mu_s"],
                    "param_mu_p": params["mu_p"],
                    "param_lam": params["lam"],
                    "param_eps": params["eps"],
                    "param_alpha": params["alpha"],
                }
                
                if log_l2:
                    print(f"\n[Epoch {epoch+1}] Loss: {tot_loss:.4e} | Data: {d_loss_accum:.4e} | BC: {b_loss_val:.4e} | PDE: {p_loss_accum:.4e}")
                    model.eval()
                    with torch.no_grad():
                        l2_errs = compute_l2_errors(model, physics, data)
                        print(f"  L2 Errors -> u: {l2_errs['u']:.4e} | v: {l2_errs['v']:.4e} | p: {l2_errs['p']:.4e}")
                        print(f"               tau_xx: {l2_errs['tau_xx']:.4e} | tau_xy: {l2_errs['tau_xy']:.4e} | tau_yy: {l2_errs['tau_yy']:.4e}")
                    model.train()
                    
                    loss_dict.update({
                        "l2_u": l2_errs["u"],
                        "l2_v": l2_errs["v"],
                        "l2_p": l2_errs["p"],
                        "l2_tau_xx": l2_errs["tau_xx"],
                        "l2_tau_xy": l2_errs["tau_xy"],
                        "l2_tau_yy": l2_errs["tau_yy"],
                    })

                    if save_dir is not None:
                        chk_path = os.path.join(save_dir, "checkpoint.pth")
                        state = {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'physics_state_dict': physics.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'history_state_dict': history.state_dict()
                        }
                        torch.save(state, chk_path)
                        print(f"  [Checkpoint] Salvato in: {chk_path}")
                        
                        if (epoch + 1) == end_adam1:
                            phase1_path = os.path.join(save_dir, "checkpoint_phase1_adam.pth")
                            torch.save(state, phase1_path)
                            print(f"  [Checkpoint Phase 1 Adam] Salvato in: {phase1_path}")

                history.update(epoch, loss_dict)

                if tb_writer is not None:
                    # Log Loss Scalars
                    tb_writer.add_scalar('Loss/Total', tot_loss, epoch)
                    tb_writer.add_scalar('Loss/Data', d_loss_accum, epoch)
                    tb_writer.add_scalar('Loss/BC', b_loss_val, epoch)
                    tb_writer.add_scalar('Loss/PDE', p_loss_accum, epoch)
                    tb_writer.add_scalar('Loss/Momentum', loss_m_val, epoch)
                    tb_writer.add_scalar('Loss/Constitutive', loss_c_val, epoch)
                    
                    # Log Physical Parameters (solo quelli effettivamente addestrati)
                    if physics.inverse_mode:
                        for name in ['beta', 'mu_s', 'mu_p', 'lam', 'eps', 'alpha']:
                            raw_p = getattr(physics, f"_raw_{name}", None)
                            if raw_p is not None and isinstance(raw_p, nn.Parameter) and raw_p.requires_grad:
                                tb_writer.add_scalar(f'Params/{name}', params[name], epoch)
                    
                    if log_l2:
                        for k in ['u', 'v', 'p', 'tau_xx', 'tau_xy', 'tau_yy']:
                            tb_writer.add_scalar(f'L2_Error/{k}', l2_errs[k], epoch)
                            
                    # Calculate and Log Gradient Norms (normgrad) for model sub-networks
                    grad_norms = {'Psi': 0.0, 'Pressure': 0.0, 'Stress': 0.0}
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if name.startswith('model_psi'):
                                grad_norms['Psi'] += param.grad.data.norm(2).item() ** 2
                            elif name.startswith('model_p'):
                                grad_norms['Pressure'] += param.grad.data.norm(2).item() ** 2
                            elif name.startswith('model_tau'):
                                grad_norms['Stress'] += param.grad.data.norm(2).item() ** 2
                                
                    for sm, norm in grad_norms.items():
                        if norm > 0:
                            tb_writer.add_scalar(f'GradNorm/{sm}', norm ** 0.5, epoch)

            pbar.set_postfix(
                {
                    "Loss": f"{tot_loss:.2e}",
                    "Data": f"{d_loss_accum:.2e}",
                    "BC": f"{b_loss_val:.2e}",
                    "PDE": f"{p_loss_accum:.2e}",
                    "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )
        pbar.close()
        start_epoch = end_adam1
        loaded_opt_state = None
        loaded_sch_state = None

    # ==================================================================
    # FASE 1.5 L-BFGS (Psi + Tau) - FP64
    # ==================================================================
    if USE_LBFGS_PHASE1 and start_epoch < end_lbfgs1:
        print(f"\n{'=' * 60}\nFASE L-BFGS 1 (Psi + Tau): {iters_ph1} iterazioni (FP64)\n{'=' * 60}")
        
        convert_to_fp64(model, physics, data)
        xy_all = data["coords"]
        uv_all = data["uv_data"]
        bc_data = data["boundary_groups"]

        if STAGED_TRAINING:
            for p in model.parameters():
                p.requires_grad = False
            for p in model.model_psi.parameters():
                p.requires_grad = True
            for p in model.model_tau.parameters():
                p.requires_grad = True
            active_bcs_lbfgs1 = ["u", "v", "tau_xx", "tau_xy", "tau_yy"]
            w_mom_lbfgs1 = 0.0
            w_con_lbfgs1 = W_CONSTITUTIVE
        else:
            for p in model.parameters():
                p.requires_grad = True
            active_bcs_lbfgs1 = None
            w_mom_lbfgs1 = W_MOMENTUM
            w_con_lbfgs1 = W_CONSTITUTIVE

        if physics.inverse_mode:
            trainable_names = ["lam", "beta", "eps", "alpha"]
            for pname in ["beta", "mu_s", "mu_p", "lam", "eps", "alpha"]:
                physics.set_trainable(pname, pname in trainable_names)

        # Log configurazione L-BFGS Fase 1
        print(f"  Active BCs: {active_bcs_lbfgs1}")
        if active_bcs_lbfgs1 is not None and "p" not in active_bcs_lbfgs1:
            print("  Pressure Point BC: DISATTIVATA in L-BFGS Fase 1")

        all_params_ph1 = [p for p in model.parameters() if p.requires_grad] + [p for p in physics.parameters() if p.requires_grad]

        optimizer_lbfgs1 = torch.optim.LBFGS(
            all_params_ph1, lr=1.0, max_iter=iters_ph1,
            tolerance_grad=1e-16, tolerance_change=1e-16, history_size=300, line_search_fn="strong_wolfe",
        )

        local_start = start_epoch - end_adam1 if start_epoch > end_adam1 else 0
        l_it1 = [local_start]
        pbar_lbfgs1 = tqdm(total=iters_ph1, initial=local_start, desc="L-BFGS Phase 1", mininterval=2.0)

        CHUNK_SIZE_LBFGS = get_optimal_chunk_size(
            phase=3, model=model,
            test_closure=lambda c: compute_and_backward_losses(
                active_bcs=active_bcs_lbfgs1, w_mom=w_mom_lbfgs1, w_con=w_con_lbfgs1,
                points=xy_all[:c], labels=(uv_all[:c] if uv_all is not None else None),
                is_lbfgs=True, chunk_size_override=c
            )
        )

        def closure1():
            optimizer_lbfgs1.zero_grad()
            tot_loss, d_loss_accum, b_loss_val, p_loss_accum, m_loss, c_loss = (
                compute_and_backward_losses(
                    active_bcs=active_bcs_lbfgs1, w_mom=w_mom_lbfgs1, w_con=w_con_lbfgs1,
                    points=xy_all, labels=uv_all, is_lbfgs=True,
                )
            )
            loss_tensor = torch.tensor(tot_loss, device=DEVICE)
            
            global_step = end_adam1 + l_it1[0]
            log_l2_lbfgs = (l_it1[0] % max(1, iters_ph1 // 40) == 0) or (l_it1[0] == iters_ph1 - 1)
            
            if log_l2_lbfgs:
                params = physics.log_params()
                with torch.no_grad():
                    l2_errs = compute_l2_errors(model, physics, data)
                
                history.update(
                    global_step,
                    {"total": tot_loss, "data": d_loss_accum, "bc": b_loss_val, "pde": p_loss_accum, "loss_momentum": m_loss, "loss_constitutive": c_loss, "param_beta": params["beta"], "param_mu_s": params["mu_s"], "param_mu_p": params["mu_p"], "param_lam": params["lam"], "param_eps": params["eps"], "param_alpha": params["alpha"], "l2_u": l2_errs["u"], "l2_v": l2_errs["v"], "l2_p": l2_errs["p"], "l2_tau_xx": l2_errs["tau_xx"], "l2_tau_xy": l2_errs["tau_xy"], "l2_tau_yy": l2_errs["tau_yy"], "l2_tau_xx_masked": l2_errs["tau_xx_masked"], "l2_tau_xy_masked": l2_errs["tau_xy_masked"], "l2_tau_yy_masked": l2_errs["tau_yy_masked"]}
                )
                print(f"\n[L-BFGS Phase 1 - Iter {l_it1[0]+1}/{iters_ph1}] Loss: {tot_loss:.4e} | Data: {d_loss_accum:.4e} | BC: {b_loss_val:.4e} | PDE: {p_loss_accum:.4e}")
                print(f"  L2 Errors -> u: {l2_errs['u']:.4e} | v: {l2_errs['v']:.4e} | p: {l2_errs['p']:.4e}")
                print(f"               tau_xx: {l2_errs['tau_xx']:.4e} | tau_xy: {l2_errs['tau_xy']:.4e} | tau_yy: {l2_errs['tau_yy']:.4e}")

                if save_dir is not None:
                    chk_path = os.path.join(save_dir, "checkpoint.pth")
                    torch.save({
                        'epoch': global_step,
                        'model_state_dict': model.state_dict(),
                        'physics_state_dict': physics.state_dict(),
                        'history_state_dict': history.state_dict()
                    }, chk_path)

                if tb_writer is not None:
                    tb_writer.add_scalar('Loss/Total', tot_loss, global_step)
                    tb_writer.add_scalar('Loss/Data', d_loss_accum, global_step)
                    tb_writer.add_scalar('Loss/BC', b_loss_val, global_step)
                    tb_writer.add_scalar('Loss/PDE', p_loss_accum, global_step)
                    tb_writer.add_scalar('Loss/Momentum', m_loss, global_step)
                    tb_writer.add_scalar('Loss/Constitutive', c_loss, global_step)

                    if physics.inverse_mode:
                        for name in ['beta', 'mu_s', 'mu_p', 'lam', 'eps', 'alpha']:
                            raw_p = getattr(physics, f"_raw_{name}", None)
                            if raw_p is not None and isinstance(raw_p, nn.Parameter) and raw_p.requires_grad:
                                tb_writer.add_scalar(f'Params/{name}', params[name], global_step)

                    for k in ['u', 'v', 'p', 'tau_xx', 'tau_xy', 'tau_yy']:
                        tb_writer.add_scalar(f'L2_Error/{k}', l2_errs[k], global_step)

                    tb_writer.flush()

            l_it1[0] += 1
            pbar_lbfgs1.update(1)
            pbar_lbfgs1.set_postfix({"Loss": f"{tot_loss:.2e}"})
            return loss_tensor

        optimizer_lbfgs1.step(closure1)
        pbar_lbfgs1.close()

        if save_dir is not None:
            chk_path = os.path.join(save_dir, "checkpoint_lbfgs_phase1.pth")
            torch.save({'model_state_dict': model.state_dict(), 'physics_state_dict': physics.state_dict(), 'history_state_dict': history.state_dict()}, chk_path)
            print(f"  [Checkpoint Phase 1 L-BFGS] Salvato in: {chk_path}")

        start_epoch = end_lbfgs1
        loaded_opt_state = None
        loaded_sch_state = None

    # ==================================================================
    # FASE 2 ADAM (Dinamica - Pressione)
    # ==================================================================
    if start_epoch < end_adam2:
        print(f"\n{'=' * 60}\nFASE 2 ADAM (Dinamica - Pressione): {ADAM_EPOCHS_PHASE2} epoche\n{'=' * 60}")
        
        # Converte modello/fisica/dati di nuovo in FP32
        convert_to_fp32(model, physics, data)
        xy_all = data["coords"]
        uv_all = data["uv_data"]
        bc_data = data["boundary_groups"]

        # Azzera i gradienti residui accumulati della Fase 1
        model.zero_grad(set_to_none=True)

        if STAGED_TRAINING:
            # Fase 2: Pressione e Psi sbloccati, tau congelato
            for p in model.parameters():
                p.requires_grad = False
            for p in model.model_psi.parameters():
                p.requires_grad = True
            for p in model.model_p.parameters():
                p.requires_grad = True
            active_bcs, w_mom, w_con, frozen_vel = ["u", "v", "p"], W_MOMENTUM, 0.0, False
        else:
            for p in model.parameters():
                p.requires_grad = True
            active_bcs, w_mom, w_con, frozen_vel = None, W_MOMENTUM, W_CONSTITUTIVE, False

        if physics.inverse_mode:
            physics.set_trainable("beta", False)
            physics.set_trainable("mu_s", True)
            for pname in ["mu_p", "lam", "eps", "alpha"]:
                physics.set_trainable(pname, False)

        # Precalcola la divergenza dello stress tau
        print("\n[Optimization] Precalcolo divergenza sforzi in corso per la Fase 2 (Adam)...")
        precomputed_div_tau = precompute_stress_divergence(model, physics, xy_all)
        print("[Optimization] Divergenza sforzi precalcolata.")

        # Calcola chunk size ottimale in FP32
        CHUNK_SIZE_ADAM = get_optimal_chunk_size(
            phase=2, model=model,
            test_closure=lambda c: compute_and_backward_losses(
                active_bcs=active_bcs, w_mom=w_mom, w_con=w_con,
                points=xy_all[:c], labels=(uv_all[:c] if uv_all is not None else None),
                is_lbfgs=False, chunk_size_override=c, frozen_velocity=frozen_vel,
                precomputed_div_tau=(precomputed_div_tau[0][:c], precomputed_div_tau[1][:c]) if precomputed_div_tau is not None else None
            )
        )

        # Costruisci l'ottimizzatore Fase 2 (LR differenziati)
        psi_params = [p for p in model.model_psi.parameters() if p.requires_grad]
        other_net_params = [p for p in model.model_p.parameters() if p.requires_grad]
        
        groups = []
        if psi_params:
            groups.append({"params": psi_params, "lr": 1e-4})
        if other_net_params:
            groups.append({"params": other_net_params, "lr": BASE_LR})

        optimizer = torch.optim.Adam(groups, eps=ADAM_EPS)
        
        local_start = start_epoch - end_lbfgs1 if start_epoch > end_lbfgs1 else 0
        steps_rem = ADAM_EPOCHS_PHASE2 - local_start
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps_rem, 1), eta_min=1e-6)

        # Carica stato se ripreso da checkpoint
        opt_loaded = False
        if loaded_opt_state is not None:
            try:
                optimizer.load_state_dict(loaded_opt_state)
                print("[Checkpoint] Optimizer state ripristinato con successo per Adam Fase 2.")
                opt_loaded = True
            except Exception as e:
                print(f"[Checkpoint] Avviso: Impossibile ripristinare optimizer state: {e}")
        if loaded_sch_state is not None and opt_loaded:
            try:
                scheduler.load_state_dict(loaded_sch_state)
            except Exception:
                pass

        pbar = tqdm(range(local_start, ADAM_EPOCHS_PHASE2), desc="Adam Fase 2", mininterval=2.0)
        for epoch_idx in pbar:
            epoch = end_lbfgs1 + epoch_idx

            model.train()
            optimizer.zero_grad(set_to_none=True)

            tot_loss, d_loss_accum, b_loss_val, p_loss_accum, loss_m_val, loss_c_val = (
                compute_and_backward_losses(
                    active_bcs, w_mom, w_con, xy_all, uv_all,
                    frozen_velocity=frozen_vel, precomputed_div_tau=precomputed_div_tau
                )
            )

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            if physics.inverse_mode:
                phys_clip = [p for p in physics.parameters() if p.requires_grad]
                if phys_clip:
                    torch.nn.utils.clip_grad_norm_(phys_clip, PARAM_CLIP_NORM)

            optimizer.step()
            scheduler.step()

            log_l2 = ((epoch_idx + 1) % max(1, ADAM_EPOCHS_PHASE2 // 40) == 0) or (epoch_idx == local_start) or ((epoch_idx + 1) == ADAM_EPOCHS_PHASE2)
            log_loss = ((epoch_idx + 1) % 10 == 0) or log_l2

            if log_loss:
                params = physics.log_params()
                loss_dict = {
                    "total": tot_loss,
                    "data": d_loss_accum,
                    "bc": b_loss_val,
                    "pde": p_loss_accum,
                    "loss_momentum": loss_m_val,
                    "loss_constitutive": loss_c_val,
                    "param_beta": params["beta"],
                    "param_mu_s": params["mu_s"],
                    "param_mu_p": params["mu_p"],
                    "param_lam": params["lam"],
                    "param_eps": params["eps"],
                    "param_alpha": params["alpha"],
                }
                
                if log_l2:
                    print(f"\n[Epoch {epoch+1}] Loss: {tot_loss:.4e} | Data: {d_loss_accum:.4e} | BC: {b_loss_val:.4e} | PDE: {p_loss_accum:.4e}")
                    model.eval()
                    with torch.no_grad():
                        l2_errs = compute_l2_errors(model, physics, data)
                        print(f"  L2 Errors -> u: {l2_errs['u']:.4e} | v: {l2_errs['v']:.4e} | p: {l2_errs['p']:.4e}")
                        print(f"               tau_xx: {l2_errs['tau_xx']:.4e} | tau_xy: {l2_errs['tau_xy']:.4e} | tau_yy: {l2_errs['tau_yy']:.4e}")
                    model.train()
                    
                    loss_dict.update({
                        "l2_u": l2_errs["u"],
                        "l2_v": l2_errs["v"],
                        "l2_p": l2_errs["p"],
                        "l2_tau_xx": l2_errs["tau_xx"],
                        "l2_tau_xy": l2_errs["tau_xy"],
                        "l2_tau_yy": l2_errs["tau_yy"],
                    })

                    if save_dir is not None:
                        chk_path = os.path.join(save_dir, "checkpoint.pth")
                        state = {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'physics_state_dict': physics.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'history_state_dict': history.state_dict()
                        }
                        torch.save(state, chk_path)
                        print(f"  [Checkpoint] Salvato in: {chk_path}")
                        
                        if (epoch_idx + 1) == ADAM_EPOCHS_PHASE2:
                            phase2_path = os.path.join(save_dir, "checkpoint_phase2_adam.pth")
                            torch.save(state, phase2_path)
                            print(f"  [Checkpoint Phase 2 Adam] Salvato in: {phase2_path}")

                history.update(epoch, loss_dict)

                if tb_writer is not None:
                    # Log Loss Scalars
                    tb_writer.add_scalar('Loss/Total', tot_loss, epoch)
                    tb_writer.add_scalar('Loss/Data', d_loss_accum, epoch)
                    tb_writer.add_scalar('Loss/BC', b_loss_val, epoch)
                    tb_writer.add_scalar('Loss/PDE', p_loss_accum, epoch)
                    tb_writer.add_scalar('Loss/Momentum', loss_m_val, epoch)
                    tb_writer.add_scalar('Loss/Constitutive', loss_c_val, epoch)
                    
                    # Log Physical Parameters (solo quelli effettivamente addestrati)
                    if physics.inverse_mode:
                        for name in ['beta', 'mu_s', 'mu_p', 'lam', 'eps', 'alpha']:
                            raw_p = getattr(physics, f"_raw_{name}", None)
                            if raw_p is not None and isinstance(raw_p, nn.Parameter) and raw_p.requires_grad:
                                tb_writer.add_scalar(f'Params/{name}', params[name], epoch)
                    
                    if log_l2:
                        for k in ['u', 'v', 'p', 'tau_xx', 'tau_xy', 'tau_yy']:
                            tb_writer.add_scalar(f'L2_Error/{k}', l2_errs[k], epoch)
                            
                    # Calculate and Log Gradient Norms (normgrad) for model sub-networks
                    grad_norms = {'Psi': 0.0, 'Pressure': 0.0, 'Stress': 0.0}
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if name.startswith('model_psi'):
                                grad_norms['Psi'] += param.grad.data.norm(2).item() ** 2
                            elif name.startswith('model_p'):
                                grad_norms['Pressure'] += param.grad.data.norm(2).item() ** 2
                            elif name.startswith('model_tau'):
                                grad_norms['Stress'] += param.grad.data.norm(2).item() ** 2
                                
                    for sm, norm in grad_norms.items():
                        if norm > 0:
                            tb_writer.add_scalar(f'GradNorm/{sm}', norm ** 0.5, epoch)

            pbar.set_postfix(
                {
                    "Loss": f"{tot_loss:.2e}",
                    "Data": f"{d_loss_accum:.2e}",
                    "BC": f"{b_loss_val:.2e}",
                    "PDE": f"{p_loss_accum:.2e}",
                    "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )
        pbar.close()
        start_epoch = end_adam2
        loaded_opt_state = None
        loaded_sch_state = None

    # ==================================================================
    # FASE L-BFGS 2 (Psi + P) - FP64
    # ==================================================================
    if USE_LBFGS_PHASE2 and start_epoch < end_lbfgs2:
        print(f"\n{'=' * 60}\nFASE L-BFGS 2 (Psi + Pressione): {iters_ph2} iterazioni (FP64)\n{'=' * 60}")
        
        convert_to_fp64(model, physics, data)
        xy_all = data["coords"]
        uv_all = data["uv_data"]
        bc_data = data["boundary_groups"]

        # Azzera i gradienti prima dell'avvio di L-BFGS
        model.zero_grad(set_to_none=True)

        # Congela psi e tau, sblocca solo p
        for p in model.parameters():
            p.requires_grad = False
        for p in model.model_p.parameters():
            p.requires_grad = True

        if physics.inverse_mode:
            physics.set_trainable("beta", False)
            physics.set_trainable("mu_s", True)
            for pname in ["mu_p", "lam", "eps", "alpha"]:
                physics.set_trainable(pname, False)

        all_params_ph2 = [p for p in model.parameters() if p.requires_grad] + [p for p in physics.parameters() if p.requires_grad]

        optimizer_lbfgs2 = torch.optim.LBFGS(
            all_params_ph2, lr=1.0, max_iter=iters_ph2,
            tolerance_grad=1e-16, tolerance_change=1e-16, history_size=300, line_search_fn="strong_wolfe",
        )

        # Precalcola la divergenza di tau in FP64
        print("\n[Optimization] Precalcolo divergenza sforzi in corso per la Fase 2 (L-BFGS, FP64)...")
        precomputed_div_tau_lbfgs = precompute_stress_divergence(model, physics, xy_all)
        print("[Optimization] Divergenza sforzi precalcolata.")

        # Calcola chunk size ottimale per L-BFGS in FP64
        CHUNK_SIZE_LBFGS = get_optimal_chunk_size(
            phase=3, model=model,
            test_closure=lambda c: compute_and_backward_losses(
                active_bcs=["p"], w_mom=W_MOMENTUM, w_con=0.0,
                points=xy_all[:c], labels=(uv_all[:c] if uv_all is not None else None),
                is_lbfgs=True, chunk_size_override=c, frozen_velocity=True,
                precomputed_div_tau=(precomputed_div_tau_lbfgs[0][:c], precomputed_div_tau_lbfgs[1][:c]) if precomputed_div_tau_lbfgs is not None else None
            )
        )

        local_start = start_epoch - end_adam2 if start_epoch > end_adam2 else 0
        l_it2 = [local_start]
        pbar_lbfgs2 = tqdm(total=iters_ph2, initial=local_start, desc="L-BFGS Phase 2", mininterval=2.0)

        def closure2():
            optimizer_lbfgs2.zero_grad()
            tot_loss, d_loss_accum, b_loss_val, p_loss_accum, m_loss, c_loss = (
                compute_and_backward_losses(
                    active_bcs=["p"], w_mom=W_MOMENTUM, w_con=0.0,
                    points=xy_all, labels=uv_all, is_lbfgs=True, frozen_velocity=True,
                    precomputed_div_tau=precomputed_div_tau_lbfgs
                )
            )
            loss_tensor = torch.tensor(tot_loss, device=DEVICE)
            
            global_step = end_adam2 + l_it2[0]
            log_l2_lbfgs = (l_it2[0] % max(1, iters_ph2 // 40) == 0) or (l_it2[0] == iters_ph2 - 1)
            
            if log_l2_lbfgs:
                params = physics.log_params()
                with torch.no_grad():
                    l2_errs = compute_l2_errors(model, physics, data)
                
                history.update(
                    global_step,
                    {"total": tot_loss, "data": d_loss_accum, "bc": b_loss_val, "pde": p_loss_accum, "loss_momentum": m_loss, "loss_constitutive": c_loss, "param_beta": params["beta"], "param_mu_s": params["mu_s"], "param_mu_p": params["mu_p"], "param_lam": params["lam"], "param_eps": params["eps"], "param_alpha": params["alpha"], "l2_u": l2_errs["u"], "l2_v": l2_errs["v"], "l2_p": l2_errs["p"], "l2_tau_xx": l2_errs["tau_xx"], "l2_tau_xy": l2_errs["tau_xy"], "l2_tau_yy": l2_errs["tau_yy"], "l2_tau_xx_masked": l2_errs["tau_xx_masked"], "l2_tau_xy_masked": l2_errs["tau_xy_masked"], "l2_tau_yy_masked": l2_errs["tau_yy_masked"]}
                )
                print(f"\n[L-BFGS Phase 2 - Iter {l_it2[0]+1}/{iters_ph2}] Loss: {tot_loss:.4e} | Data: {d_loss_accum:.4e} | BC: {b_loss_val:.4e} | PDE: {p_loss_accum:.4e}")
                print(f"  L2 Errors -> u: {l2_errs['u']:.4e} | v: {l2_errs['v']:.4e} | p: {l2_errs['p']:.4e}")
                print(f"               tau_xx: {l2_errs['tau_xx']:.4e} | tau_xy: {l2_errs['tau_xy']:.4e} | tau_yy: {l2_errs['tau_yy']:.4e}")

                if save_dir is not None:
                    chk_path = os.path.join(save_dir, "checkpoint.pth")
                    torch.save({
                        'epoch': global_step,
                        'model_state_dict': model.state_dict(),
                        'physics_state_dict': physics.state_dict(),
                        'history_state_dict': history.state_dict()
                    }, chk_path)

                if tb_writer is not None:
                    tb_writer.add_scalar('Loss/Total', tot_loss, global_step)
                    tb_writer.add_scalar('Loss/Data', d_loss_accum, global_step)
                    tb_writer.add_scalar('Loss/BC', b_loss_val, global_step)
                    tb_writer.add_scalar('Loss/PDE', p_loss_accum, global_step)
                    tb_writer.add_scalar('Loss/Momentum', m_loss, global_step)
                    tb_writer.add_scalar('Loss/Constitutive', c_loss, global_step)

                    if physics.inverse_mode:
                        for name in ['beta', 'mu_s', 'mu_p', 'lam', 'eps', 'alpha']:
                            raw_p = getattr(physics, f"_raw_{name}", None)
                            if raw_p is not None and isinstance(raw_p, nn.Parameter) and raw_p.requires_grad:
                                tb_writer.add_scalar(f'Params/{name}', params[name], global_step)

                    for k in ['u', 'v', 'p', 'tau_xx', 'tau_xy', 'tau_yy']:
                        tb_writer.add_scalar(f'L2_Error/{k}', l2_errs[k], global_step)

                    tb_writer.flush()

            l_it2[0] += 1
            pbar_lbfgs2.update(1)
            pbar_lbfgs2.set_postfix({"Loss": f"{tot_loss:.2e}"})
            return loss_tensor

        optimizer_lbfgs2.step(closure2)
        pbar_lbfgs2.close()

        if save_dir is not None:
            chk_path = os.path.join(save_dir, "checkpoint_lbfgs_phase2.pth")
            torch.save({'model_state_dict': model.state_dict(), 'physics_state_dict': physics.state_dict(), 'history_state_dict': history.state_dict()}, chk_path)
            print(f"  [Checkpoint Phase 2 L-BFGS] Salvato in: {chk_path}")

    return history
