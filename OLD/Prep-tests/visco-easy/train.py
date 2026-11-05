"""
visco-easy/train.py
===================
Script MINIMALE e AUTOCONTENUTO per il training di una PINN viscoelastica.
Replica esattamente il caso Goal 1 Inverso (semi-inverse, dati u,v) del
framework principale, senza alcuna dipendenza esterna.

Precisione staged: FP32+TF32 per Adam, FP64 per L-BFGS.
CUDA forzato, pesi statici, non-staged.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from tqdm import tqdm
from pathlib import Path

# ============================================================================
# 1. SETUP & CONFIGURAZIONE
# ============================================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')  # Abilita TF32 per matmul (Ampere+)
torch.backends.cudnn.benchmark = False
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

DEVICE = torch.device("cuda")

# --- Percorsi ---
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / 'COMSOL' / 'Oldroyd_mau_res.csv'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Parametri fisici REALI del fluido (ground truth) ---
MU_S_TRUE = 0.1      # Viscosità solvente [Pa·s]
MU_P_TRUE = 0.9      # Viscosità polimerica [Pa·s]
LAM_TRUE = 1.0        # Tempo di rilassamento [s]
EPS_TRUE = 0.0        # Parametro PTT
ALPHA_TRUE = 0.0      # Parametro Giesekus
RHO = 1000.0          # Densità [kg/m³]

# --- Guess iniziali (80% dei valori veri) ---
GUESS_MULTIPLIER = 0.8
GUESS_MU_S = MU_S_TRUE * GUESS_MULTIPLIER
GUESS_MU_P = MU_P_TRUE * GUESS_MULTIPLIER
GUESS_LAM = LAM_TRUE * GUESS_MULTIPLIER
GUESS_EPS = 0.0
GUESS_ALPHA = 0.0

# --- Architettura NN ---
HIDDEN_LAYERS = [128] * 8  # 8 hidden layers da 128 neuroni
ACTIVATION = nn.SiLU

# --- Training ---
ADAM_EPOCHS = 80
LBFGS_MAX_ITERS = ADAM_EPOCHS*0.1
BASE_LR = 1e-3
ADAM_EPS = 1e-7
PARAM_LR_FACTOR = 0.1
GRAD_CLIP_NORM = 5.0
PARAM_CLIP_NORM = 1.0
MINIBATCH_INTERNAL = 4096
MINIBATCH_BOUNDARY = 512
SEED = 123

# Warmup: a quale epoca sbloccare i parametri fisici
WARMUP_UNLOCK_EPOCH = int(ADAM_EPOCHS * 0.25) 

# --- Pesi loss statici ---
W_BC = 10.0
W_PHYSICS = 10.0
W_DATA = 1.0

# --- Pesi PDE ---
W_MOMENTUM = 10.0
W_CONSTITUTIVE = 1.0

# --- Varianza epsilon ---
VARIANCE_EPS = 1e-4


# ============================================================================
# 2. DATA LOADING (inlinato)
# ============================================================================
def load_data():
    """Carica il CSV COMSOL, adimensionalizza, estrae boundary groups dalla mesh."""
    print("=" * 60)
    print("Caricamento dataset COMSOL...")

    # --- 2a. Parsing CSV ---
    rows = []
    with open(str(DATASET_PATH), 'r') as f:
        for line in f:
            s = line.strip()
            if s.startswith('%') or len(s) == 0:
                continue
            rows.append(s)

    data_np = np.loadtxt(rows, dtype=np.float64, delimiter=',')
    assert data_np.shape[1] >= 8, f"Attese almeno 8 colonne, trovate {data_np.shape[1]}"
    N = data_np.shape[0]

    x_raw, y_raw = data_np[:, 0], data_np[:, 1]
    u_raw, v_raw = data_np[:, 2], data_np[:, 3]
    p_raw = data_np[:, 4]
    txx_raw, txy_raw, tyy_raw = data_np[:, 5], data_np[:, 6], data_np[:, 7]

    # --- 2b. Scale di riferimento ---
    y_min, y_max = y_raw.min(), y_raw.max()
    x_min, x_max = x_raw.min(), x_raw.max()
    H = y_max - y_min if (y_max - y_min) > 1e-9 else 1.0
    U_ref = max(float(np.abs(u_raw).max()), 1e-9)
    mu_tot = MU_S_TRUE + MU_P_TRUE
    p_ref = mu_tot * U_ref / H
    tau_ref = mu_tot * U_ref / H

    # --- 2c. Adimensionalizzazione ---
    x_nd = (x_raw - x_min) / H
    y_nd = (y_raw - y_min) / H
    u_nd = u_raw / U_ref
    v_nd = v_raw / U_ref
    p_nd = p_raw / p_ref
    txx_nd = txx_raw / tau_ref
    txy_nd = txy_raw / tau_ref
    tyy_nd = tyy_raw / tau_ref

    # Tensori (FP32 esplicito — numpy usa float64 di default)
    coords = torch.tensor(np.stack([x_nd, y_nd], axis=1), dtype=torch.float32, device=DEVICE)
    u_t = torch.tensor(u_nd, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    v_t = torch.tensor(v_nd, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    p_t = torch.tensor(p_nd, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    txx_t = torch.tensor(txx_nd, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    txy_t = torch.tensor(txy_nd, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    tyy_t = torch.tensor(tyy_nd, dtype=torch.float32, device=DEVICE).reshape(-1, 1)

    # --- 2d. Varianze per normalizzazione ---
    var_weights = {
        'u': max(u_t.var().item(), VARIANCE_EPS),
        'v': max(v_t.var().item(), VARIANCE_EPS),
        'p': max(p_t.var().item(), VARIANCE_EPS),
        'tau_xx': max(txx_t.var().item(), VARIANCE_EPS),
        'tau_xy': max(txy_t.var().item(), VARIANCE_EPS),
        'tau_yy': max(tyy_t.var().item(), VARIANCE_EPS),
    }

    # --- 2e. Triangolazione per i plot ---
    x_np = coords[:, 0].cpu().numpy()
    y_np = coords[:, 1].cpu().numpy()
    triang = mtri.Triangulation(x_np, y_np)
    try:
        triangles = triang.triangles
        x_tri, y_tri = x_np[triangles], y_np[triangles]
        l1 = np.hypot(x_tri[:, 0] - x_tri[:, 1], y_tri[:, 0] - y_tri[:, 1])
        l2 = np.hypot(x_tri[:, 1] - x_tri[:, 2], y_tri[:, 1] - y_tri[:, 2])
        l3 = np.hypot(x_tri[:, 2] - x_tri[:, 0], y_tri[:, 2] - y_tri[:, 0])
        max_edge = np.maximum(np.maximum(l1, l2), l3)
        threshold = 2.0 * np.median(max_edge)
        mask = max_edge > threshold
        if mask.sum() > 0:
            triang.set_mask(mask)
    except Exception:
        pass

    print(f"  Punti totali: {N}")
    print(f"  H={H:.6e}, U_ref={U_ref:.6e}, p_ref={p_ref:.6e}")
    print(f"  Re={RHO * U_ref * H / mu_tot:.4f}, Wi={LAM_TRUE * U_ref / H:.4f}, beta={MU_S_TRUE / mu_tot:.4f}")

    # --- 2f. Boundary groups dalla mesh .mphtxt ---
    boundary_groups = _extract_boundary_groups(
        coords, x_raw, y_raw, x_min, y_min, H,
        {'u': u_t, 'v': v_t, 'p': p_t, 'tau_xx': txx_t, 'tau_xy': txy_t, 'tau_yy': tyy_t}
    )

    print("=" * 60)

    return {
        'coords': coords,
        'u': u_t, 'v': v_t, 'p': p_t,
        'tau_xx': txx_t, 'tau_xy': txy_t, 'tau_yy': tyy_t,
        'uv_data': torch.cat([u_t, v_t], dim=1),
        'var_weights': var_weights,
        'triang': triang,
        'boundary_groups': boundary_groups,
        'U_ref': U_ref, 'H': H,
    }


def _extract_boundary_groups(coords, x_raw, y_raw, x_min, y_min, H, fields):
    """Parsing mesh .mphtxt per estrarre boundary groups con normali."""
    from scipy.spatial import cKDTree

    mphtxt_path = str(DATASET_PATH).replace('.csv', '_geom.mphtxt')
    if not os.path.isfile(mphtxt_path):
        mphtxt_path = str(DATASET_PATH).replace('.csv', '.mphtxt')
    if not os.path.isfile(mphtxt_path):
        raise FileNotFoundError(f"File mesh .mphtxt non trovato per {DATASET_PATH}")

    with open(mphtxt_path, 'r') as f:
        lines = [line.strip() for line in f]

    # --- Parsing vertici ---
    num_vertices, vertices_start = 0, -1
    for idx, line in enumerate(lines):
        if '# number of mesh vertices' in line:
            num_vertices = int(line.split('#')[0].strip())
        elif '# Mesh vertex coordinates' in line:
            vertices_start = idx + 1

    vertices_raw = np.array([
        [float(x) for x in lines[vertices_start + i].split()]
        for i in range(num_vertices)
    ])

    # --- Parsing edg2 ---
    edg_elements, edg_entity_indices = [], []
    edg_start = -1
    for idx, line in enumerate(lines):
        if 'edg2 # type name' in line:
            edg_start = idx
            break

    if edg_start != -1:
        num_edg = 0
        edg_elem_idx = -1
        for i in range(edg_start, len(lines)):
            if '# number of elements' in lines[i]:
                num_edg = int(lines[i].split('#')[0].strip())
                break
        for i in range(edg_start, len(lines)):
            if '# Elements' in lines[i]:
                edg_elem_idx = i + 1
                break
        if edg_elem_idx != -1:
            for i in range(num_edg):
                parts = lines[edg_elem_idx + i].split()
                edg_elements.append([int(parts[0]), int(parts[1]), int(parts[2])])
        edg_entity_idx = -1
        for i in range(edg_elem_idx + num_edg, len(lines)):
            if '# Geometric entity indices' in lines[i]:
                edg_entity_idx = i + 1
                break
        if edg_entity_idx != -1:
            for i in range(num_edg):
                edg_entity_indices.append(int(lines[edg_entity_idx + i]))

    # --- Parsing tri2 ---
    tri_elements = []
    tri_start = -1
    for idx, line in enumerate(lines):
        if 'tri2 # type name' in line:
            tri_start = idx
            break
    if tri_start != -1:
        num_tri, tri_elem_idx = 0, -1
        for i in range(tri_start, len(lines)):
            if '# number of elements' in lines[i]:
                num_tri = int(lines[i].split('#')[0].strip())
                break
        for i in range(tri_start, len(lines)):
            if '# Elements' in lines[i]:
                tri_elem_idx = i + 1
                break
        if tri_elem_idx != -1:
            for i in range(num_tri):
                parts = lines[tri_elem_idx + i].split()
                tri_elements.append([int(parts[0]), int(parts[1]), int(parts[2])])

    # --- Parsing Selection ---
    selections = {}
    idx = 0
    while idx < len(lines):
        if 'Selection # class' in lines[idx]:
            label = ""
            for i in range(idx + 1, min(idx + 10, len(lines))):
                if '# Label' in lines[i]:
                    raw_label = lines[i].split('#')[0].strip()
                    parts = raw_label.split(maxsplit=1)
                    label = parts[1] if (len(parts) == 2 and parts[0].isdigit()) else raw_label
                    break
            num_entities, ent_start = 0, -1
            for i in range(idx + 1, min(idx + 20, len(lines))):
                if '# Number of entities' in lines[i]:
                    num_entities = int(lines[i].split('#')[0].strip())
                    ent_start = i + 2
                    break
            entities = []
            if ent_start != -1:
                for i in range(num_entities):
                    entities.append(int(lines[ent_start + i]))
            if label:
                selections[label] = entities
            idx = (ent_start + num_entities) if ent_start != -1 else idx + 1
        else:
            idx += 1

    # Fallback geometrico se nessuna selection trovata
    if not selections and edg_entity_indices:
        edge_nodes = {}
        for edg, eid in zip(edg_elements, edg_entity_indices):
            edge_nodes.setdefault(eid, set()).update(edg)
        x_min_m, x_max_m = vertices_raw[:, 0].min(), vertices_raw[:, 0].max()
        inlet_e, outlet_e, walls_e = [], [], []
        tol_g = 1e-6
        for eid, nids in edge_nodes.items():
            pts = vertices_raw[list(nids)]
            if abs(pts[:, 0].min() - x_min_m) < tol_g and abs(pts[:, 0].max() - x_min_m) < tol_g:
                inlet_e.append(eid)
            elif abs(pts[:, 0].min() - x_max_m) < tol_g and abs(pts[:, 0].max() - x_max_m) < tol_g:
                outlet_e.append(eid)
            else:
                walls_e.append(eid)
        if inlet_e: selections['Inlet'] = inlet_e
        if outlet_e: selections['Outlet'] = outlet_e
        if walls_e: selections['Walls'] = walls_e

    # --- Nodo → triangoli adiacenti ---
    node_to_tri = {}
    for t_idx, tri in enumerate(tri_elements):
        for nid in tri:
            node_to_tri.setdefault(nid, []).append(t_idx)

    # --- Accoppiamento geometrico ---
    coords_np = coords.cpu().numpy()
    x_min_mesh = vertices_raw[:, 0].min()
    y_min_mesh = vertices_raw[:, 1].min()
    vertices_nd = np.stack([
        (vertices_raw[:, 0] - x_min_mesh) / H,
        (vertices_raw[:, 1] - y_min_mesh) / H
    ], axis=1)

    tree_csv = cKDTree(coords_np)
    dists_nearest, _ = tree_csv.query(coords_np, k=2)
    tol_match = max(np.median(dists_nearest[:, 1]) * 0.5, 1e-6)

    edge_to_nodes = {}
    for edg, eid in zip(edg_elements, edg_entity_indices):
        edge_to_nodes.setdefault(eid, set()).update(edg)

    boundary_groups = {}
    for label, entities in selections.items():
        sel_nodes = set()
        for eid in entities:
            if eid in edge_to_nodes:
                sel_nodes.update(edge_to_nodes[eid])
        if not sel_nodes:
            continue

        # Calcolo normali locali
        group_normals = np.zeros((num_vertices, 2))
        eset = set(entities)
        for edg, eid in zip(edg_elements, edg_entity_indices):
            if eid not in eset:
                continue
            ga, gb, gmid = edg
            adj_tri = None
            if ga in node_to_tri and gb in node_to_tri:
                common = set(node_to_tri[ga]).intersection(node_to_tri[gb])
                if common:
                    adj_tri = list(common)[0]
            if adj_tri is None:
                continue
            tri = tri_elements[adj_tri]
            g_opp = None
            for nid in tri:
                if nid != ga and nid != gb:
                    g_opp = nid
                    break
            if g_opp is None:
                continue
            pa, pb, po = vertices_raw[ga], vertices_raw[gb], vertices_raw[g_opp]
            tangent = pb - pa
            length = np.linalg.norm(tangent)
            if length > 0:
                t_unit = tangent / length
                p_mid = 0.5 * (pa + pb)
                to_int = po - p_mid
                n_cand = np.array([t_unit[1], -t_unit[0]])
                if np.dot(n_cand, to_int) > 0:
                    n_cand = -n_cand
                for g in [ga, gb, gmid]:
                    group_normals[g] += n_cand

        # Matching con CSV
        global_idx, global_norm = [], []
        for nid in sel_nodes:
            dist, cidx = tree_csv.query(vertices_nd[nid])
            if dist < tol_match:
                global_idx.append(cidx)
                n_vec = group_normals[nid]
                n_mag = np.linalg.norm(n_vec)
                global_norm.append(n_vec / n_mag if n_mag > 1e-9 else n_vec)

        if not global_idx:
            continue

        boundary_groups[label] = {
            'indices': torch.tensor(global_idx, dtype=torch.long, device=DEVICE),
            'xy': coords[global_idx].to(DEVICE),
            'norm': torch.tensor(np.array(global_norm), dtype=torch.float32, device=DEVICE),
            'fields': {k: v[global_idx].to(DEVICE) for k, v in fields.items()},
        }
        print(f"  Boundary '{label}': {len(global_idx)} nodi")

    return boundary_groups


# ============================================================================
# 3. MODELLI NN
# ============================================================================
class FCN(nn.Module):
    """Fully Connected Network."""
    def __init__(self, n_input, n_output, hidden_layers):
        super().__init__()
        layers_sizes = [n_input] + hidden_layers + [n_output]
        self.fcs = nn.ModuleList()
        for i in range(len(layers_sizes) - 1):
            self.fcs.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
        self.act = ACTIVATION()

    def forward(self, x):
        for layer in self.fcs[:-1]:
            x = self.act(layer(x))
        return self.fcs[-1](x)


class CombinedModel(nn.Module):
    """Combina psi (1), p (1), tau (3) in un unico output [psi, p, txx, txy, tyy]."""
    def __init__(self):
        super().__init__()
        self.model_psi = FCN(2, 1, HIDDEN_LAYERS)
        self.model_p = FCN(2, 1, HIDDEN_LAYERS)
        self.model_tau = FCN(2, 3, HIDDEN_LAYERS)

    def forward(self, x):
        return torch.cat([self.model_psi(x), self.model_p(x), self.model_tau(x)], dim=1)


# ============================================================================
# 4. FISICA HARDCODATA
# ============================================================================
class Physics(nn.Module):
    """PDE adimensionali + boundary conditions hardcodate. Inverse mode sempre attivo."""
    def __init__(self, U_ref, H_ref):
        super().__init__()
        self.U_ref = U_ref
        self.H_ref = H_ref
        # Parametri trainabili (inverse problem)
        self.mu_s = nn.Parameter(torch.tensor([GUESS_MU_S], device=DEVICE))
        self.mu_p = nn.Parameter(torch.tensor([GUESS_MU_P], device=DEVICE))
        self.lam = nn.Parameter(torch.tensor([GUESS_LAM], device=DEVICE))
        self.eps = nn.Parameter(torch.tensor([GUESS_EPS], device=DEVICE))
        self.alpha = nn.Parameter(torch.tensor([GUESS_ALPHA], device=DEVICE))
        # Referenza fissa per adimensionalizzazione
        self.real_mu_tot = MU_S_TRUE + MU_P_TRUE

    def get_velocity(self, model, x):
        """Calcola u, v, p, tau dalla stream function."""
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        out = model(x)
        psi, p, tau = out[:, 0:1], out[:, 1:2], out[:, 2:5]
        grad_psi = torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        u = grad_psi[:, 1:2]
        v = -grad_psi[:, 0:1]
        return u, v, p, tau

    def _nondim(self):
        """Parametri adimensionali correnti."""
        Re = RHO * self.U_ref * self.H_ref / self.real_mu_tot
        Wi = self.lam * self.U_ref / self.H_ref
        beta = self.mu_s / self.real_mu_tot
        beta_poly = self.mu_p / self.real_mu_tot
        return Re, Wi, beta, beta_poly, self.eps, self.alpha

    def compute_residuals(self, model, x):
        """Calcola i residui PDE adimensionali."""
        Re, Wi, beta, beta_poly, eps, alpha = self._nondim()

        out = model(x)
        psi, p, tau = out[:, 0:1], out[:, 1:2], out[:, 2:5]
        tau_xx, tau_xy, tau_yy = tau[:, 0:1], tau[:, 1:2], tau[:, 2:3]

        # Cinematica
        grad_psi = torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        u, v = grad_psi[:, 1:2], -grad_psi[:, 0:1]
        grad_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
        grad_v = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_x, v_y = grad_v[:, 0:1], -u_x  # v_y = -u_x (incompressibilità)

        # Derivate seconde e pressione
        grad_p = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:, 0:1]
        u_yx = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0][:, 1:2]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0][:, 0:1]
        v_yy = -u_yx

        # Derivate stress
        g_txx = torch.autograd.grad(tau_xx.sum(), x, create_graph=True)[0]
        g_txy = torch.autograd.grad(tau_xy.sum(), x, create_graph=True)[0]
        g_tyy = torch.autograd.grad(tau_yy.sum(), x, create_graph=True)[0]
        tau_xx_x, tau_xx_y = g_txx[:, 0:1], g_txx[:, 1:2]
        tau_xy_x, tau_xy_y = g_txy[:, 0:1], g_txy[:, 1:2]
        tau_yy_x, tau_yy_y = g_tyy[:, 0:1], g_tyy[:, 1:2]

        # Momentum
        f_u = Re * (u * u_x + v * u_y) + p_x - beta * (u_xx + u_yy) - (tau_xx_x + tau_xy_y)
        f_v = Re * (u * v_x + v * v_y) + p_y - beta * (v_xx + v_yy) - (tau_xy_x + tau_yy_y)

        # Costitutive (Oldroyd-B/PTT/Giesekus)
        f_PTT = 1.0 + (eps * Wi / beta_poly) * (tau_xx + tau_yy)
        upper_xx = u * tau_xx_x + v * tau_xx_y - 2 * u_x * tau_xx - 2 * u_y * tau_xy
        upper_yy = u * tau_yy_x + v * tau_yy_y - 2 * v_x * tau_xy - 2 * v_y * tau_yy
        upper_xy = u * tau_xy_x + v * tau_xy_y - u_x * tau_xy - u_y * tau_yy - tau_xx * v_x - tau_xy * v_y

        f_txx = f_PTT * tau_xx + Wi * upper_xx + (alpha * Wi / beta_poly) * (tau_xx**2 + tau_xy**2) - 2.0 * beta_poly * u_x
        f_tyy = f_PTT * tau_yy + Wi * upper_yy + (alpha * Wi / beta_poly) * (tau_xy**2 + tau_yy**2) - 2.0 * beta_poly * v_y
        f_txy = f_PTT * tau_xy + Wi * upper_xy + (alpha * Wi / beta_poly) * tau_xy * (tau_xx + tau_yy) - beta_poly * (u_y + v_x)

        return f_u, f_v, f_txx, f_tyy, f_txy

    def pde_loss(self, model, x):
        """Loss PDE con pesi momentum/constitutive."""
        f_u, f_v, f_txx, f_tyy, f_txy = self.compute_residuals(model, x)
        loss_m = (f_u**2 + f_v**2).mean()
        loss_c = (f_txx**2).mean() + (f_tyy**2).mean() + (f_txy**2).mean()
        return W_MOMENTUM * loss_m + W_CONSTITUTIVE * loss_c

    def data_loss(self, model, x, uv_target, var_w):
        """Loss dati: solo u, v (Goal 1)."""
        u, v, _, _ = self.get_velocity(model, x)
        loss_u = nn.MSELoss()(u, uv_target[:, 0:1]) / var_w['u']
        loss_v = nn.MSELoss()(v, uv_target[:, 1:2]) / var_w['v']
        return 0.5 * (loss_u + loss_v)

    def boundary_loss(self, model, bc_data, var_w):
        """
        BC hardcodate:
          Inlet: Dirichlet u,v da CSV + tau=0
          Walls/Walls-dritte: Dirichlet u=v=0
          Outlet: Dirichlet v=0, p da CSV
        """
        total_loss = torch.tensor(0.0, device=DEVICE, dtype=next(model.parameters()).dtype)

        for group_name, gd in bc_data.items():
            x_bc = gd['xy'].clone().requires_grad_(True)
            u, v, p, tau = self.get_velocity(model, x_bc)
            g_loss = torch.tensor(0.0, device=DEVICE, dtype=next(model.parameters()).dtype)

            if group_name == 'Inlet':
                # u, v da CSV
                g_loss += nn.MSELoss()(u, gd['fields']['u']) / var_w['u']
                g_loss += nn.MSELoss()(v, gd['fields']['v']) / var_w['v']
                # tau = 0
                g_loss += nn.MSELoss()(tau[:, 0:1], torch.zeros_like(tau[:, 0:1])) / var_w['tau_xx']
                g_loss += nn.MSELoss()(tau[:, 1:2], torch.zeros_like(tau[:, 1:2])) / var_w['tau_xy']
                g_loss += nn.MSELoss()(tau[:, 2:3], torch.zeros_like(tau[:, 2:3])) / var_w['tau_yy']

            elif group_name in ('Walls', 'Walls-dritte'):
                # u = v = 0
                g_loss += nn.MSELoss()(u, torch.zeros_like(u)) / var_w['u']
                g_loss += nn.MSELoss()(v, torch.zeros_like(v)) / var_w['v']

            elif group_name == 'Outlet':
                # v = 0, p da CSV
                g_loss += nn.MSELoss()(v, torch.zeros_like(v)) / var_w['v']
                g_loss += nn.MSELoss()(p, gd['fields']['p']) / var_w['p']

            total_loss += g_loss

        return total_loss

    def clamp_params(self):
        """Vincoli fisici sui parametri."""
        with torch.no_grad():
            self.eps.clamp_(min=0.0)
            self.alpha.clamp_(min=0.0)
            self.mu_s.clamp_(min=1e-6)
            self.mu_p.clamp_(min=1e-6)
            self.lam.clamp_(min=1e-6)

    def log_params(self):
        """Restituisce i parametri correnti come dict di float."""
        return {
            'mu_s': self.mu_s.item(), 'mu_p': self.mu_p.item(),
            'lam': self.lam.item(), 'eps': self.eps.item(), 'alpha': self.alpha.item(),
        }


# ============================================================================
# 5. PLOTTING MINIMALE
# ============================================================================
class SimpleHistory:
    """Tracker minimale per loss e parametri."""
    def __init__(self):
        self.epochs = []
        self.losses = {}

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
        """Plot loss totale/data/bc/pde."""
        fig, ax = plt.subplots(figsize=(10, 5))
        keys_plot = ['total', 'data', 'bc', 'pde']
        colors = {'total': 'black', 'data': 'blue', 'bc': 'green', 'pde': 'red'}
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
        ax.set_title('Training Loss History')
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


def plot_fields(model, physics, data, save_path):
    """Confronto campi predetti vs COMSOL."""
    model.eval()
    _dtype = next(model.parameters()).dtype
    with torch.set_grad_enabled(True):
        x_in = data['coords'].to(_dtype).clone().requires_grad_(True)
        u_p, v_p, p_p, tau_p = physics.get_velocity(model, x_in)

    preds = {
        'u': u_p.detach().cpu().view(-1),
        'p': p_p.detach().cpu().view(-1),
        'tau_xx': tau_p[:, 0].detach().cpu().view(-1),
        'tau_xy': tau_p[:, 1].detach().cpu().view(-1),
        'tau_yy': tau_p[:, 2].detach().cpu().view(-1),
    }
    exacts = {
        'u': data['u'].cpu().view(-1),
        'p': data['p'].cpu().view(-1),
        'tau_xx': data['tau_xx'].cpu().view(-1),
        'tau_xy': data['tau_xy'].cpu().view(-1),
        'tau_yy': data['tau_yy'].cpu().view(-1),
    }
    triang = data['triang']
    field_names = ['u', 'p', 'tau_xx', 'tau_xy', 'tau_yy']

    fig, axs = plt.subplots(len(field_names), 3, figsize=(18, 4 * len(field_names)))
    for i, fn in enumerate(field_names):
        ex = exacts[fn].numpy().astype(np.float64)
        pr = preds[fn].numpy().astype(np.float64)
        err = np.abs(ex - pr)

        vmin, vmax = ex.min(), ex.max()
        axs[i, 0].tricontourf(triang, ex, levels=50, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axs[i, 0].set_title(f'{fn} (COMSOL)')
        axs[i, 0].set_aspect('equal')

        axs[i, 1].tricontourf(triang, pr, levels=50, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axs[i, 1].set_title(f'{fn} (PINN)')
        axs[i, 1].set_aspect('equal')

        im = axs[i, 2].tricontourf(triang, err, levels=50, cmap='hot_r')
        axs[i, 2].set_title(f'{fn} (|Error|)')
        axs[i, 2].set_aspect('equal')
        plt.colorbar(im, ax=axs[i, 2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [PLOT] Campi salvati in {save_path}")


# ============================================================================
# 6. TRAINING LOOP
# ============================================================================
def sample_minibatch(xy, targets, batch_size):
    """Campionamento random di un mini-batch."""
    if batch_size is None or batch_size >= xy.shape[0]:
        return xy, targets
    idx = torch.randperm(xy.shape[0], device=DEVICE)[:batch_size]
    return xy[idx], targets[idx] if targets is not None else None


def train(model, physics, data):
    """Training completo: Adam + L-BFGS."""
    history = SimpleHistory()

    xy_all = data['coords']
    uv_all = data['uv_data']
    var_w = data['var_weights']
    bc_data = data['boundary_groups']

    # ==================================================================
    # FASE ADAM
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"FASE ADAM: {ADAM_EPOCHS} epoche")
    print(f"{'='*60}")

    # Inizialmente: tutti i modelli attivi, parametri fisici congelati
    for p in model.parameters():
        p.requires_grad = True
    for pname in ['mu_s', 'mu_p', 'lam', 'eps', 'alpha']:
        getattr(physics, pname).requires_grad_(False)

    def build_optimizer(steps):
        net_params = [p for p in model.parameters() if p.requires_grad]
        phys_params = [p for p in physics.parameters() if p.requires_grad]
        groups = [
            {'params': net_params, 'lr': BASE_LR},
            {'params': phys_params, 'lr': BASE_LR * PARAM_LR_FACTOR},
        ]
        opt = torch.optim.Adam(groups, eps=ADAM_EPS)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=1e-6)
        all_trainable = net_params + phys_params
        return opt, sch, all_trainable

    optimizer, scheduler, trainable_params = build_optimizer(WARMUP_UNLOCK_EPOCH)

    pbar = tqdm(range(ADAM_EPOCHS), desc="Adam", mininterval=2.0)
    for epoch in pbar:
        # Warmup: sblocco parametri fisici
        if epoch == WARMUP_UNLOCK_EPOCH:
            print(f"\n  [Warmup] Sblocco mu_s, mu_p, lam (epoca {epoch})")
            physics.mu_s.requires_grad_(True)
            physics.mu_p.requires_grad_(True)
            physics.lam.requires_grad_(True)
            # eps e alpha restano congelati a 0
            optimizer, scheduler, trainable_params = build_optimizer(ADAM_EPOCHS - epoch)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Mini-batch dati interni
        xb, yb = sample_minibatch(xy_all, uv_all, MINIBATCH_INTERNAL)

        # Mini-batch boundary (campionamento proporzionale per gruppo)
        bc_mini = {}
        total_bc = sum(gd['xy'].shape[0] for gd in bc_data.values())
        for gname, gd in bc_data.items():
            n_g = gd['xy'].shape[0]
            n_sample = max(1, int(round(n_g * MINIBATCH_BOUNDARY / max(total_bc, 1))))
            n_sample = min(n_g, n_sample)
            if n_sample < n_g:
                idx = torch.randperm(n_g, device=DEVICE)[:n_sample]
                bc_mini[gname] = {
                    'xy': gd['xy'][idx],
                    'fields': {k: v[idx] for k, v in gd['fields'].items()},
                }
            else:
                bc_mini[gname] = gd

        # Compute losses
        d_loss = physics.data_loss(model, xb, yb, var_w)
        b_loss = physics.boundary_loss(model, bc_mini, var_w)

        xph = xb.clone().requires_grad_(True)
        p_loss = physics.pde_loss(model, xph)

        total_loss = W_DATA * d_loss + W_BC * b_loss + W_PHYSICS * p_loss

        total_loss.backward(inputs=trainable_params)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        phys_clip = [p for p in physics.parameters() if p.requires_grad]
        if phys_clip:
            torch.nn.utils.clip_grad_norm_(phys_clip, PARAM_CLIP_NORM)

        optimizer.step()
        physics.clamp_params()
        scheduler.step()

        # Logging
        if (epoch + 1) % 100 == 0:
            params = physics.log_params()
            history.update(epoch, {
                'total': total_loss.item(),
                'data': d_loss.item(),
                'bc': b_loss.item(),
                'pde': p_loss.item(),
                'param_mu_s': params['mu_s'],
                'param_mu_p': params['mu_p'],
                'param_lam': params['lam'],
                'param_eps': params['eps'],
                'param_alpha': params['alpha'],
            })
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.2e}",
                'LR': f"{optimizer.param_groups[0]['lr']:.2e}",
            })

    pbar.close()

    # ==================================================================
    # FASE L-BFGS (switch a FP64)
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"FASE L-BFGS: {LBFGS_MAX_ITERS} iterazioni (FP64)")
    print(f"{'='*60}")

    # Cast a FP64 per precisione scientifica
    torch.set_default_dtype(torch.float64)
    model.double()
    physics.double()
    xy_all = xy_all.double()
    uv_all = uv_all.double()
    # Cast anche i boundary groups
    for gname, gd in bc_data.items():
        gd['xy'] = gd['xy'].double()
        gd['norm'] = gd['norm'].double()
        for fname in gd['fields']:
            gd['fields'][fname] = gd['fields'][fname].double()

    # Tutti i modelli + mu_s, mu_p, lam trainabili
    for p in model.parameters():
        p.requires_grad = True
    physics.mu_s.requires_grad_(True)
    physics.mu_p.requires_grad_(True)
    physics.lam.requires_grad_(True)

    all_params = list(model.parameters()) + [physics.mu_s, physics.mu_p, physics.lam]
    optimizer_lbfgs = torch.optim.LBFGS(
        all_params, lr=1.0, max_iter=int(LBFGS_MAX_ITERS),
        tolerance_grad=1e-9, tolerance_change=1e-12,
        history_size=300, line_search_fn="strong_wolfe"
    )

    # Punti collocazione per la PDE (full batch)
    xph_full = xy_all.clone().requires_grad_(True)
    chunk_size = 2000
    l_it = [0]
    pbar = tqdm(total=int(LBFGS_MAX_ITERS), desc="L-BFGS", mininterval=2.0)

    def closure():
        optimizer_lbfgs.zero_grad()
        accum = {'data': 0.0, 'bc': 0.0, 'pde': 0.0}

        # Data loss (chunked)
        for i in range(0, xy_all.shape[0], chunk_size):
            xc = xy_all[i:i + chunk_size]
            yc = uv_all[i:i + chunk_size]
            dl = physics.data_loss(model, xc, yc, var_w)
            w = xc.shape[0] / xy_all.shape[0]
            accum['data'] += dl.item() * w
            (W_DATA * dl * w).backward()

        # BC loss (no chunking)
        bl = physics.boundary_loss(model, bc_data, var_w)
        accum['bc'] = bl.item()
        (W_BC * bl).backward()

        # PDE loss (chunked)
        for i in range(0, xph_full.shape[0], chunk_size):
            xc = xph_full[i:i + chunk_size]
            pl = physics.pde_loss(model, xc)
            w = xc.shape[0] / xph_full.shape[0]
            accum['pde'] += pl.item() * w
            (W_PHYSICS * pl * w).backward()

        total_val = W_DATA * accum['data'] + W_BC * accum['bc'] + W_PHYSICS * accum['pde']

        if l_it[0] % 50 == 0:
            params = physics.log_params()
            history.update(ADAM_EPOCHS + l_it[0], {
                'total': total_val,
                'data': accum['data'],
                'bc': accum['bc'],
                'pde': accum['pde'],
                'param_mu_s': params['mu_s'],
                'param_mu_p': params['mu_p'],
                'param_lam': params['lam'],
                'param_eps': params['eps'],
                'param_alpha': params['alpha'],
            })

        l_it[0] += 1
        pbar.update(1)
        pbar.set_postfix({'Loss': f'{total_val:.2e}'})

        return torch.tensor(total_val, device=DEVICE, requires_grad=True)

    optimizer_lbfgs.step(closure)
    physics.clamp_params()
    pbar.close()

    return history


# ============================================================================
# 7. METRICHE E MAIN
# ============================================================================
def compute_l2_errors(model, physics, data):
    """Calcola L2 relative errors per tutti i campi."""
    model.eval()
    _dtype = next(model.parameters()).dtype
    with torch.set_grad_enabled(True):
        xi = data['coords'].to(_dtype).clone().requires_grad_(True)
        u_p, v_p, p_p, tau_p = physics.get_velocity(model, xi)

    preds = {'u': u_p, 'p': p_p, 'tau_xx': tau_p[:, 0:1], 'tau_xy': tau_p[:, 1:2], 'tau_yy': tau_p[:, 2:3]}
    exacts = {'u': data['u'], 'p': data['p'], 'tau_xx': data['tau_xx'], 'tau_xy': data['tau_xy'], 'tau_yy': data['tau_yy']}

    errors = {}
    for fn in preds:
        pr = preds[fn].detach().view(-1)
        ex = exacts[fn].to(pr.dtype).view(-1)
        norm_ex = torch.norm(ex, 2)
        l2 = (torch.norm(pr - ex, 2) / norm_ex).item() if norm_ex > 1e-10 else 0.0
        errors[fn] = l2
    return errors


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print(f"Dtype: {torch.get_default_dtype()}")
    print(f"Dataset: {DATASET_PATH}")
    print()

    # 1. Carica dati
    data = load_data()

    # 2. Costruisci modelli
    model = CombinedModel().to(DEVICE)
    physics = Physics(U_ref=data['U_ref'], H_ref=data['H']).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModello: {total_params:,} parametri totali")
    print(f"Guess iniziali: mu_s={GUESS_MU_S}, mu_p={GUESS_MU_P}, lam={GUESS_LAM}")
    print(f"Valori veri:    mu_s={MU_S_TRUE}, mu_p={MU_P_TRUE}, lam={LAM_TRUE}")

    # 3. Training
    history = train(model, physics, data)

    # 4. Risultati finali
    params = physics.log_params()
    print(f"\n{'='*60}")
    print("RISULTATI FINALI")
    print(f"{'='*60}")
    print(f"  mu_s:  {params['mu_s']:.6f}  (true: {MU_S_TRUE})")
    print(f"  mu_p:  {params['mu_p']:.6f}  (true: {MU_P_TRUE})")
    print(f"  lam:   {params['lam']:.6f}  (true: {LAM_TRUE})")
    print(f"  eps:   {params['eps']:.6f}  (true: {EPS_TRUE})")
    print(f"  alpha: {params['alpha']:.6f}  (true: {ALPHA_TRUE})")

    errors = compute_l2_errors(model, physics, data)
    print(f"\nL2 Relative Errors:")
    for fn, err in errors.items():
        print(f"  {fn:>8s}: {err:.6f}")

    # 5. Plot
    history.plot_losses(str(OUTPUT_DIR / 'loss_history.png'))
    history.plot_params(str(OUTPUT_DIR / 'params_evolution.png'))
    plot_fields(model, physics, data, str(OUTPUT_DIR / 'fields_comparison.png'))

    print(f"\nPlot salvati in: {OUTPUT_DIR}")
    print("Done!")
