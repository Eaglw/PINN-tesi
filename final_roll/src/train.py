import os
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils import convert_to_fp64
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
        param_keys = [('param_mu_s', MU_S_TRUE, r'$\eta_s$'),
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
            ax.axhline(true_val, color='k', linestyle='--', linewidth=2, label='True')
            ax.set_title(label)
            ax.grid(True, ls='--', alpha=0.5)
            ax.legend()

        axs[-1].set_xlabel('Epoch / Iter')
        fig.suptitle('Physical Parameters Evolution', fontsize=14)
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


def train(model, physics, data, resume_checkpoint=None, save_dir=None):
    """
    Training completo PINN: Adam (staged/non-staged) + L-BFGS (FP64).
    Implementazione ottimizzata per il contenimento della VRAM.
    """
    global CHUNK_SIZE_ADAM, CHUNK_SIZE_LBFGS
    history = SimpleHistory()

    start_epoch = 0
    loaded_opt_state = None
    loaded_sch_state = None

    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        print(f"\n[Checkpoint] Caricamento da: {resume_checkpoint}")
        chk = torch.load(resume_checkpoint, map_location=DEVICE)
        
        model.load_state_dict(chk['model_state_dict'])
        physics.load_state_dict(chk['physics_state_dict'])
        history.load_state_dict(chk['history_state_dict'])
        
        loaded_opt_state = chk.get('optimizer_state_dict', None)
        loaded_sch_state = chk.get('scheduler_state_dict', None)
        
        start_epoch = chk.get('epoch', 0) + 1
        print(f"[Checkpoint] Ripresa dall'epoca {start_epoch}")

    # --- ESTRAZIONE DATI ---
    xy_all = data["coords"]
    uv_all = data["uv_data"]
    var_w = data["var_weights"]
    bc_data = data["boundary_groups"]
    total_points = xy_all.shape[0]

    half_epochs = int(ADAM_EPOCHS * 1.1)
    #half_epochs = 10
    def configure_staged_phase(epoch):
        """Modifica i flag requires_grad dei sottomodelli. Chiamata SOLO ai cambi di fase."""
        if not STAGED_TRAINING:
            for p in model.parameters():
                p.requires_grad = True
            return None, W_MOMENTUM, W_CONSTITUTIVE
        if epoch < half_epochs:
            # Fase 1: Cinematica e Reologia (Congela Pressione)
            for p in model.parameters():
                p.requires_grad = False
            for p in model.model_psi.parameters():
                p.requires_grad = True
            for p in model.model_tau.parameters():
                p.requires_grad = True
            return ["u", "v", "tau_xx", "tau_xy", "tau_yy"], 0.0, W_CONSTITUTIVE
        else:
            # Fase 2: Dinamica (Congela Sforzi)
            for p in model.parameters():
                p.requires_grad = False
            for p in model.model_psi.parameters():
                p.requires_grad = True
            for p in model.model_p.parameters():
                p.requires_grad = True
            return ["u", "v", "p"], W_MOMENTUM, 0.0



    def build_optimizer(steps_remaining):
        """Costruisce Adam e lo Scheduler, gestendo l'inclusione dei parametri fisici e LR differenziati in Fase 2."""
        
        # Rilevamento dinamico Fase 2: solo se STAGED_TRAINING è True e la pressione è sbloccata
        is_phase2 = STAGED_TRAINING and any(p.requires_grad for p in model.model_p.parameters())
        
        if is_phase2:
            psi_params = [p for p in model.model_psi.parameters() if p.requires_grad]
            other_net_params = [p for p in model.model_p.parameters() if p.requires_grad] + \
                               [p for p in model.model_tau.parameters() if p.requires_grad]
            
            groups = [
                {"params": psi_params, "lr": 1e-5},
                {"params": other_net_params, "lr": BASE_LR}
            ]
        else:
            net_params = [p for p in model.parameters() if p.requires_grad]
            groups = [{"params": net_params, "lr": BASE_LR}]

        if physics.inverse_mode:
            phys_params = [p for p in physics.parameters() if p.requires_grad]
            if phys_params:
                groups.append({"params": phys_params, "lr": BASE_LR * PARAM_LR_FACTOR})

        opt = torch.optim.Adam(groups, eps=ADAM_EPS)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(steps_remaining, 1), eta_min=1e-6
        )
        return opt, sch

    def compute_and_backward_losses(
        active_bcs, w_mom, w_con, points, labels, is_lbfgs=False, chunk_size_override=None
    ):
        """
        Motore di calcolo centralizzato con Accumulazione dei Gradienti in Chunk.

        LOGICA DI MEMORIA: PyTorch crea un enorme grafo computazionale durante il forward.
        Se calcolassimo tutta la loss prima di fare `.backward()`, la GPU saturerebbe.
        Chiamando `.backward()` separatamente per i Dati e per la PDE *all'interno* del loop
        dei chunk, accumuliamo iterativamente il gradiente matematico e forziamo la GPU a
        distruggere immediatamente i grafi intermedi, mantenendo la memoria piatta e costante.
        """
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if physics.inverse_mode:
            trainable_params += [p for p in physics.parameters() if p.requires_grad]

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
            lm, lc = physics.compute_pde_losses(xph, u, v, p, tau, w_mom, w_con)
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
    # FASE 1: ADAM
    # ==================================================================
    if STAGED_TRAINING:
        print(
            f"\n{'=' * 60}\nFASE 1 ADAM (Cinematica e Reologia): {half_epochs} epoche\n{'=' * 60}"
        )
    else:
        print(
            f"\n{'=' * 60}\nFASE ADAM UNICA (Tutto Attivo): {ADAM_EPOCHS} epoche\n{'=' * 60}"
        )

    from src.utils import get_optimal_chunk_size
    # Configurazione di base per l'epoca di partenza
    if STAGED_TRAINING and start_epoch >= half_epochs:
        active_bcs, w_mom, w_con = configure_staged_phase(start_epoch)
        if physics.inverse_mode:
            for p in [physics.mu_s, physics.mu_p, physics.lam, physics.eps, physics.alpha]:
                p.requires_grad_(False)
        CHUNK_SIZE_ADAM = get_optimal_chunk_size(
            phase=2, model=model,
            test_closure=lambda c: compute_and_backward_losses(
                active_bcs=active_bcs, w_mom=w_mom, w_con=w_con,
                points=xy_all[:c], labels=(uv_all[:c] if uv_all is not None else None),
                is_lbfgs=False, chunk_size_override=c
            )
        )
    else:
        active_bcs, w_mom, w_con = configure_staged_phase(start_epoch)
        if physics.inverse_mode:
            if start_epoch >= WARMUP_UNLOCK_EPOCH:
                for p in [physics.mu_s, physics.mu_p, physics.lam]:
                    p.requires_grad_(True)
            else:
                for pname in ["mu_s", "mu_p", "lam", "eps", "alpha"]:
                    getattr(physics, pname).requires_grad_(False)
        CHUNK_SIZE_ADAM = get_optimal_chunk_size(
            phase=1, model=model,
            test_closure=lambda c: compute_and_backward_losses(
                active_bcs=active_bcs, w_mom=w_mom, w_con=w_con,
                points=xy_all[:c], labels=(uv_all[:c] if uv_all is not None else None),
                is_lbfgs=False, chunk_size_override=c
            )
        )

    # Calcolo step rimanenti per lo scheduler iniziale
    steps_rem = (half_epochs - start_epoch) if (STAGED_TRAINING and start_epoch < half_epochs) else (ADAM_EPOCHS - start_epoch)
    optimizer, scheduler = build_optimizer(steps_rem)

    if loaded_opt_state is not None:
        try:
            optimizer.load_state_dict(loaded_opt_state)
            print("[Checkpoint] Optimizer state ripristinato con successo.")
        except Exception as e:
            print(f"[Checkpoint] Avviso: Impossibile ripristinare optimizer state (possibile cambio di fase): {e}")

    if loaded_sch_state is not None:
        try:
            scheduler.load_state_dict(loaded_sch_state)
        except Exception:
            pass

    pbar = tqdm(range(start_epoch, ADAM_EPOCHS), desc="Adam", mininterval=2.0)
    for epoch in pbar:
        # Sblocco parametri fisici (Warmup Problema Inverso)
        if physics.inverse_mode and epoch == WARMUP_UNLOCK_EPOCH:
            print(f"\n  [Warmup Stage 1] Sblocco mu_s, mu_p, lam (epoca {epoch})")
            for p in [physics.mu_s, physics.mu_p, physics.lam]:
                p.requires_grad_(True)
            steps_rem = (
                (half_epochs - epoch) if STAGED_TRAINING else (ADAM_EPOCHS - epoch)
            )
            optimizer, scheduler = build_optimizer(steps_rem)

        # Transizione di Fase (Eseguita UNA sola volta all'epoca esatta)
        if STAGED_TRAINING and epoch == half_epochs:
            print(
                f"\n{'=' * 60}\nFASE 2 ADAM (Dinamica): {ADAM_EPOCHS - half_epochs} epoche\n{'=' * 60}"
            )

            active_bcs, w_mom, w_con = configure_staged_phase(
                epoch
            )  # Ricalcolo layer attivi
            if physics.inverse_mode:
                for p in [
                    physics.mu_s,
                    physics.mu_p,
                    physics.lam,
                    physics.eps,
                    physics.alpha,
                ]:
                    p.requires_grad_(False)
                    
            from src.utils import get_optimal_chunk_size
            CHUNK_SIZE_ADAM = get_optimal_chunk_size(
                phase=2, model=model,
                test_closure=lambda c: compute_and_backward_losses(
                    active_bcs=active_bcs, w_mom=w_mom, w_con=w_con,
                    points=xy_all[:c], labels=(uv_all[:c] if uv_all is not None else None),
                    is_lbfgs=False, chunk_size_override=c
                )
            )
            optimizer, scheduler = build_optimizer(ADAM_EPOCHS - half_epochs)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Motore centrale: calcola le loss in chunk evitando OOM
        tot_loss, d_loss_accum, b_loss_val, p_loss_accum, _, _ = (
            compute_and_backward_losses(active_bcs, w_mom, w_con, xy_all, uv_all)
        )

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        if physics.inverse_mode:
            phys_clip = [
                p
                for p in [
                    physics.mu_s,
                    physics.mu_p,
                    physics.lam,
                    physics.eps,
                    physics.alpha,
                ]
                if p.requires_grad
            ]
            if phys_clip:
                torch.nn.utils.clip_grad_norm_(phys_clip, PARAM_CLIP_NORM)

        optimizer.step()
        if physics.inverse_mode:
            physics.clamp_params()
        scheduler.step()

        # Condizioni di logging separate
        log_loss = ((epoch + 1) % 10 == 0) or (epoch == 0) or (STAGED_TRAINING and (epoch + 1) == half_epochs)
        log_l2 = ((epoch + 1) % max(1, ADAM_EPOCHS // 40) == 0) or (epoch == 0) or (STAGED_TRAINING and (epoch + 1) == half_epochs)

        if log_loss:
            params = physics.log_params()
            loss_dict = {
                "total": tot_loss,
                "data": d_loss_accum,
                "bc": b_loss_val,
                "pde": p_loss_accum,
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
                    "l2_tau_xx_masked": l2_errs["tau_xx_masked"],
                    "l2_tau_xy_masked": l2_errs["tau_xy_masked"],
                    "l2_tau_yy_masked": l2_errs["tau_yy_masked"],
                })

                if save_dir is not None:
                    chk_path = os.path.join(save_dir, "checkpoint.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'physics_state_dict': physics.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'history_state_dict': history.state_dict()
                    }, chk_path)
                    print(f"  [Checkpoint] Salvato in: {chk_path}")

            history.update(epoch, loss_dict)

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

    # ==================================================================
    # FASE 2: L-BFGS (Attivazione Tramite Flag)
    # ==================================================================
    if not USE_LBFGS:
        print(
            "\n[!] Fase L-BFGS disabilitata tramite configurazione. Addestramento concluso."
        )
        return history

    print(
        f"\n{'=' * 60}\nFASE L-BFGS: {int(LBFGS_MAX_ITERS)} iterazioni (FP64)\n{'=' * 60}"
    )

    convert_to_fp64(model, physics, data)
    xy_all = data["coords"]
    uv_all = data["uv_data"]
    bc_data = data["boundary_groups"]

    for p in model.parameters():
        p.requires_grad = True
    if physics.inverse_mode:
        for p in [physics.mu_s, physics.mu_p, physics.lam]:
            p.requires_grad_(True)
        all_params = list(model.parameters()) + [
            physics.mu_s,
            physics.mu_p,
            physics.lam,
        ]
    else:
        all_params = list(model.parameters())

    optimizer_lbfgs = torch.optim.LBFGS(
        all_params,
        lr=1.0,
        max_iter=int(LBFGS_MAX_ITERS),
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=300,
        line_search_fn="strong_wolfe",
    )

    l_it = [0]
    pbar_lbfgs = tqdm(total=int(LBFGS_MAX_ITERS), desc="L-BFGS", mininterval=2.0)

    from src.utils import get_optimal_chunk_size
    global CHUNK_SIZE_LBFGS
    CHUNK_SIZE_LBFGS = get_optimal_chunk_size(
        phase=3, model=model,
        test_closure=lambda c: compute_and_backward_losses(
            active_bcs=None, w_mom=W_MOMENTUM, w_con=W_CONSTITUTIVE,
            points=xy_all[:c], labels=(uv_all[:c] if uv_all is not None else None),
            is_lbfgs=True, chunk_size_override=c
        )
    )

    def closure():
        optimizer_lbfgs.zero_grad()

        # is_lbfgs=True assicura che venga usato CHUNK_SIZE_LBFGS per evitare OOM
        tot_loss, d_loss_accum, b_loss_val, p_loss_accum, m_loss, c_loss = (
            compute_and_backward_losses(
                active_bcs=None,
                w_mom=W_MOMENTUM,
                w_con=W_CONSTITUTIVE,
                points=xy_all,
                labels=uv_all,
                is_lbfgs=True,
            )
        )

        loss_tensor = torch.tensor(tot_loss, device=DEVICE)

        log_lbfgs = (l_it[0] % max(1, int(LBFGS_MAX_ITERS) // 100) == 0) or (l_it[0] == int(LBFGS_MAX_ITERS) - 1)
        if log_lbfgs:
            params = physics.log_params()
            
            print(f"\n[L-BFGS Iter {l_it[0]}] Loss: {tot_loss:.4e} | Data: {d_loss_accum:.4e} | BC: {b_loss_val:.4e} | PDE: {p_loss_accum:.4e}")

            with torch.no_grad():
                l2_errs = compute_l2_errors(model, physics, data)
                print(f"  L2 Errors -> u: {l2_errs['u']:.4e} | v: {l2_errs['v']:.4e} | p: {l2_errs['p']:.4e}")
                print(f"               tau_xx: {l2_errs['tau_xx']:.4e} | tau_xy: {l2_errs['tau_xy']:.4e} | tau_yy: {l2_errs['tau_yy']:.4e}")

            if save_dir is not None:
                chk_path = os.path.join(save_dir, "checkpoint.pth")
                torch.save({
                    'epoch': ADAM_EPOCHS + l_it[0],
                    'model_state_dict': model.state_dict(),
                    'physics_state_dict': physics.state_dict(),
                    'optimizer_state_dict': optimizer_lbfgs.state_dict(),
                    'history_state_dict': history.state_dict()
                }, chk_path)
                print(f"  [Checkpoint] Salvato in: {chk_path}")

            history.update(
                ADAM_EPOCHS + l_it[0],
                {
                    "total": tot_loss,
                    "data": d_loss_accum,
                    "bc": b_loss_val,
                    "pde": p_loss_accum,
                    "loss_momentum": m_loss,
                    "loss_constitutive": c_loss,
                    "param_mu_s": params["mu_s"],
                    "param_mu_p": params["mu_p"],
                    "param_lam": params["lam"],
                    "param_eps": params["eps"],
                    "param_alpha": params["alpha"],
                    "l2_u": l2_errs["u"],
                    "l2_v": l2_errs["v"],
                    "l2_p": l2_errs["p"],
                    "l2_tau_xx": l2_errs["tau_xx"],
                    "l2_tau_xy": l2_errs["tau_xy"],
                    "l2_tau_yy": l2_errs["tau_yy"],
                    "l2_tau_xx_masked": l2_errs["tau_xx_masked"],
                    "l2_tau_xy_masked": l2_errs["tau_xy_masked"],
                    "l2_tau_yy_masked": l2_errs["tau_yy_masked"],
                },
            )

        l_it[0] += 1
        pbar_lbfgs.update(1)
        pbar_lbfgs.set_postfix(
            {
                "Loss": f"{tot_loss:.2e}",
                "Data": f"{d_loss_accum:.2e}",
                "BC": f"{b_loss_val:.2e}",
                "PDE": f"{p_loss_accum:.2e}",
            }
        )

        return loss_tensor

    optimizer_lbfgs.step(closure)

    if physics.inverse_mode:
        physics.clamp_params()

    pbar_lbfgs.close()

    return history
