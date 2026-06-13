"""
visco-easy/train_4rollmill.py
=============================
Script MINIMALE e AUTOCONTENUTO per il training di una PINN viscoelastica su 4rollmill.
Adatta la struttura semplificata di visco-easy/train.py al caso 4rollmill, 
implementando lo staged training (Fase 1: psi + tau, Fase 2: psi + p, Fase 3: L-BFGS).

Precisione staged: FP32+TF32 per Adam, FP64 per L-BFGS.
CUDA forzato, pesi statici.
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
from datetime import datetime
from pathlib import Path

# --- Logging automatico di tutti i print ---
LOG_FILE_PATH = None

def print(*args, **kwargs):
    import builtins
    builtins.print(*args, **kwargs)
    if LOG_FILE_PATH is not None:
        sep = kwargs.get('sep', ' ')
        end = kwargs.get('end', '\n')
        text = sep.join(map(str, args)) + end
        with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            f.write(text)

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

# --- Opzioni di Controllo ---
STAGED_TRAINING = True  # True: staged (Fase 1: psi+tau, Fase 2: psi+p), False: non-staged (tutto attivo da subito)
INVERSE_PROBLEM = False  # True: semi-inverso (parametri fisici ottimizzati), False: diretto (parametri reali bloccati)
CHUNK_SIZE_ADAM = 7000  # Dimensione dei chunk per Adam. Aumenta per velocità, diminuisci se satura la VRAM (es. 20128 per full batch totale)

# --- Percorsi ---
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / 'COMSOL' / '4roll' / '4_roll_mill.csv'

# --- Costanti di Clamping per Parametri Strettamente Positivi ---
MIN_MU_S = 1e-6
MIN_MU_P = 1e-6
MIN_LAM = 1e-6

def weighted_mse(pred, target, var):
    """Formula esplicita per la weighted MSE normalizzata tramite la varianza."""
    return torch.mean(((pred - target) ** 2) / var)

def convert_to_fp64(model, physics, data):
    """Converte in modo centralizzato modello, fisica e dati a FP64 prima di L-BFGS."""
    model.double()
    physics.double()
    
    # Cast dei dati interni
    data['coords'] = data['coords'].double()
    data['u'] = data['u'].double()
    data['v'] = data['v'].double()
    data['p'] = data['p'].double()
    data['tau_xx'] = data['tau_xx'].double()
    data['tau_xy'] = data['tau_xy'].double()
    data['tau_yy'] = data['tau_yy'].double()
    data['uv_data'] = data['uv_data'].double()
    
    # Cast dei boundary groups
    for group_name, gd in data['boundary_groups'].items():
        gd['xy'] = gd['xy'].double()
        gd['norm'] = gd['norm'].double()
        for fname in gd['fields']:
            gd['fields'][fname] = gd['fields'][fname].double()

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
ACTIVATION = nn.Tanh

# --- Training ---
ADAM_EPOCHS = 1000*5
LBFGS_MAX_ITERS = int(0.1*ADAM_EPOCHS) # 10% di epoche Adam
BASE_LR = 1e-3
ADAM_EPS = 1e-7
PARAM_LR_FACTOR = 0.1
GRAD_CLIP_NORM = 5.0
PARAM_CLIP_NORM = 1.0
MINIBATCH_INTERNAL = 2048
MINIBATCH_BOUNDARY = 256
SEED = 123

# Warmup: a quale epoca sbloccare i parametri fisici in Fase 1
WARMUP_UNLOCK_EPOCH = int(0.2*ADAM_EPOCHS)

# --- Pesi loss statici ---
W_BC = 2.0
W_PHYSICS = 3.0
W_DATA = 1.0

# --- Pesi PDE ---
W_MOMENTUM = 1.0
W_CONSTITUTIVE = 1.0

# --- Varianza epsilon ---
VARIANCE_EPS = 1e-4

# --- Configurazione Cartella Output Dinamica ---
layers_str = f"{len(HIDDEN_LAYERS)}x{HIDDEN_LAYERS[0]}"
config_name = f"{DATASET_PATH.stem}_L{layers_str}_E{ADAM_EPOCHS}_{ACTIVATION.__name__}_staged{STAGED_TRAINING}_inv{INVERSE_PROBLEM}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = BASE_DIR / 'output_4rollmill' / config_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File per il salvataggio dei log di training
LOG_FILE_PATH = OUTPUT_DIR / 'train_log.txt'

# Frequenza monitoraggio print (circa 4-5 volte durante il training)
PRINT_EVERY = max(1, ADAM_EPOCHS // 4)


# ============================================================================
# 2. DATA LOADING (inlinato)
# ============================================================================
def load_data():
    """Carica il CSV COMSOL, adimensionalizza, estrae boundary groups dalla mesh."""
    print("=" * 60)
    print("Caricamento dataset COMSOL (4rollmill)...")

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
    U_ref = max(float(np.max(np.sqrt(u_raw**2 + v_raw**2))), 1e-9)
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

    # Tensori (FP32 esplicito)
    coords = torch.tensor(np.stack([x_nd, y_nd], axis=1), dtype=torch.float32, device=DEVICE)
    u_t = torch.tensor(u_nd, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    v_t = torch.tensor(v_nd, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    p_t = torch.tensor(p_nd, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    txx_t = torch.tensor(txx_nd, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    txy_t = torch.tensor(txy_nd, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    tyy_t = torch.tensor(tyy_nd, dtype=torch.float32, device=DEVICE).reshape(-1, 1)

    # Output scales
    p_scale = max(float(np.abs(p_nd).max()), 1.0)
    tau_scale = max(float(max(np.abs(txx_nd).max(), np.abs(txy_nd).max(), np.abs(tyy_nd).max())), 1.0)

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
        threshold = 5.0 * np.median(max_edge)
        mask = max_edge > threshold
        if mask.sum() > 0:
            triang.set_mask(mask)
    except Exception:
        pass

    print(f"  Punti totali: {N}")
    print(f"  H={H:.6e}, U_ref={U_ref:.6e}, p_ref={p_ref:.6e}")
    print(f"  Re={RHO * U_ref * H / mu_tot:.4f}, Wi={LAM_TRUE * U_ref / H:.4f}, beta={MU_S_TRUE / mu_tot:.4f}")
    print(f"  [Output Scaling] p_scale={p_scale:.4f}, tau_scale={tau_scale:.4f}")

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
        'p_scale': p_scale, 'tau_scale': tau_scale,
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
        if 'edg2 # type name' in line or 'edg # type name' in line:
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
                edg_elements.append([int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else int(parts[1])])
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
        if 'tri2 # type name' in line or 'tri # type name' in line:
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
        edge_to_nodes.setdefault(eid, set()).update(edg[:2])

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
            ga, gb = edg[0], edg[1]
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
                for g in [ga, gb]:
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

    # Iniettiamo il PressurePoint dinamicamente vicino al centro del bordo superiore (y massimo)
    if 'Walls' in boundary_groups and 'PressurePoint' not in boundary_groups:
        ref_group = boundary_groups['Walls']
        xy = ref_group['xy']
        x_coords = xy[:, 0]
        y_coords = xy[:, 1]
        x_mean = x_coords.mean()
        y_max_val = y_coords.max()
        
        # Filtra i punti con y >= 0.9 * y_max_val
        mask = y_coords >= 0.9 * y_max_val
        indices_filtered = torch.where(mask)[0]
        
        # Trova quello che minimizza (x - x_mean)**2
        x_filtered = x_coords[indices_filtered]
        dists = (x_filtered - x_mean)**2
        best_local_idx = torch.argmin(dists).item()
        best_idx = indices_filtered[best_local_idx].item()
        
        boundary_groups['PressurePoint'] = {
            'indices': ref_group['indices'][best_idx : best_idx + 1],
            'xy': ref_group['xy'][best_idx : best_idx + 1],
            'norm': ref_group['norm'][best_idx : best_idx + 1],
            'fields': {k: v[best_idx : best_idx + 1] for k, v in ref_group['fields'].items()}
        }
        print(f"  Boundary 'PressurePoint' iniettato dinamicamente (centro-alto): coordinate {xy[best_idx].tolist()}")

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
    """Combina psi (1), p (1), tau (3) in un unico output con scaling di pressione e stress."""
    def __init__(self, p_scale=1.0, tau_scale=1.0):
        super().__init__()
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
    """Azzera l'ultimo layer di una rete per stabilità iniziale."""
    last_layer = list(model.fcs)[-1]
    nn.init.zeros_(last_layer.weight)
    nn.init.zeros_(last_layer.bias)


def init_weights_xavier(m):
    """Inizializzazione dei pesi Xavier Normal e azzeramento dei bias con gain per Tanh."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ============================================================================
# 4. FISICA HARDCODATA
# ============================================================================
class Physics(nn.Module):
    """PDE adimensionali + boundary conditions. Supporta modalità diretta o inversa."""
    def __init__(self, U_ref, H_ref, var_weights=None, inverse_mode=True):
        super().__init__()
        self.U_ref = U_ref
        self.H_ref = H_ref
        self.var_weights = var_weights
        self.inverse_mode = inverse_mode
        
        if inverse_mode:
            # Parametri trainabili (inverse problem)
            self.mu_s = nn.Parameter(torch.tensor([GUESS_MU_S], device=DEVICE))
            self.mu_p = nn.Parameter(torch.tensor([GUESS_MU_P], device=DEVICE))
            self.lam = nn.Parameter(torch.tensor([GUESS_LAM], device=DEVICE))
            self.eps = nn.Parameter(torch.tensor([GUESS_EPS], device=DEVICE))
            self.alpha = nn.Parameter(torch.tensor([GUESS_ALPHA], device=DEVICE))
        else:
            # Parametri costanti bloccati (direct problem)
            self.register_buffer('mu_s', torch.tensor([MU_S_TRUE], device=DEVICE))
            self.register_buffer('mu_p', torch.tensor([MU_P_TRUE], device=DEVICE))
            self.register_buffer('lam', torch.tensor([LAM_TRUE], device=DEVICE))
            self.register_buffer('eps', torch.tensor([EPS_TRUE], device=DEVICE))
            self.register_buffer('alpha', torch.tensor([ALPHA_TRUE], device=DEVICE))
            
        # Referenza fissa per adimensionalizzazione
        self.real_mu_tot = MU_S_TRUE + MU_P_TRUE

    def get_velocity(self, model, x, create_graph=True):
        """Calcola u, v, p, tau dalla stream function."""
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        out = model(x)
        psi, p, tau = out[:, 0:1], out[:, 1:2], out[:, 2:5]
        grad_psi = torch.autograd.grad(psi.sum(), x, create_graph=create_graph)[0]
        u = grad_psi[:, 1:2]
        v = -grad_psi[:, 0:1]
        return u, v, p, tau

    def _nondim(self):
        """Parametri adimensionali correnti."""
        mu_tot = self.mu_s + self.mu_p
        Re = RHO * self.U_ref * self.H_ref / mu_tot
        Wi = self.lam * self.U_ref / self.H_ref
        beta = self.mu_s / mu_tot
        beta_poly = self.mu_p / mu_tot
        return Re, Wi, beta, beta_poly, self.eps, self.alpha

    def compute_residuals(self, model, x, w_momentum=1.0, w_constitutive=1.0):
        """Calcola i residui PDE adimensionali in modo ottimizzato con meno autograd.grad, saltando i calcoli superflui se i pesi associati sono nulli."""
        Re, Wi, beta, beta_poly, eps, alpha = self._nondim()

        # Chiamiamo le tre sotto-reti separatamente per avere grafi computazionali
        # completamente indipendenti. Usare model(x) che internamente fa torch.cat
        # collegherebbe i tre grafi in un unico nodo: il backward di tau passerebbe
        # attraverso cat e potrebbe liberare i saved tensors di model_p e model_psi.
        psi = model.model_psi(x)
        p   = model.model_p(x) * model.p_scale
        tau = model.model_tau(x) * model.tau_scale
        tau_xx, tau_xy, tau_yy = tau[:, 0:1], tau[:, 1:2], tau[:, 2:3]

        # Cinematica (gradiente primo w.r.t coordinates x)
        grad_psi = torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        u = grad_psi[:, 1:2]
        v = -grad_psi[:, 0:1]

        grad_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]

        grad_v = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_x, v_y = grad_v[:, 0:1], -u_x  # v_y = -u_x (incompressibilità)

        # Derivate stress (necessarie se momentum o constitutive sono attivi)
        if w_momentum > 0.0 or w_constitutive > 0.0:
            cg = (w_constitutive > 0.0)
            # tau_xx, tau_xy, tau_yy provengono dalla stessa sotto-rete model_tau:
            # retain_graph=True è necessario sulle prime due chiamate per non liberare
            # il grafo di model_tau prima di aver calcolato tutti e tre i gradienti.
            g_txx = torch.autograd.grad(tau_xx.sum(), x, create_graph=cg, retain_graph=True)[0]
            tau_xx_x, tau_xx_y = g_txx[:, 0:1], g_txx[:, 1:2]

            g_txy = torch.autograd.grad(tau_xy.sum(), x, create_graph=cg, retain_graph=True)[0]
            tau_xy_x, tau_xy_y = g_txy[:, 0:1], g_txy[:, 1:2]

            g_tyy = torch.autograd.grad(tau_yy.sum(), x, create_graph=cg)[0]  # ultima: libera il grafo
            tau_yy_x, tau_yy_y = g_tyy[:, 0:1], g_tyy[:, 1:2]
            
            if not cg:
                tau_xx_x = tau_xx_x.detach()
                tau_xx_y = tau_xx_y.detach()
                tau_xy_x = tau_xy_x.detach()
                tau_xy_y = tau_xy_y.detach()
                tau_yy_x = tau_yy_x.detach()
                tau_yy_y = tau_yy_y.detach()
        else:
            tau_xx_x = tau_xx_y = tau_xy_x = tau_xy_y = tau_yy_x = tau_yy_y = None

        # Momentum
        if w_momentum > 0.0:
            grad_p = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
            p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]

            grad_u_x = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
            u_xx = grad_u_x[:, 0:1]

            # Doppia chiamata ad autograd.grad su u_y risolta in un'unica chiamata (grad_u_y)
            grad_u_y = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0]
            u_yx, u_yy = grad_u_y[:, 0:1], grad_u_y[:, 1:2]

            grad_v_x = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
            v_xx = grad_v_x[:, 0:1]

            v_yy = -u_yx

            f_u = Re * (u * u_x + v * u_y) + p_x - beta * (u_xx + u_yy) - (tau_xx_x + tau_xy_y)
            f_v = Re * (u * v_x + v * v_y) + p_y - beta * (v_xx + v_yy) - (tau_xy_x + tau_yy_y)
        else:
            f_u = torch.zeros_like(u)
            f_v = torch.zeros_like(u)

        # Costitutive (Oldroyd-B/PTT/Giesekus)
        if w_constitutive > 0.0:
            f_PTT = 1.0 + (eps * Wi / beta_poly) * (tau_xx + tau_yy)
            upper_xx = u * tau_xx_x + v * tau_xx_y - 2 * u_x * tau_xx - 2 * u_y * tau_xy
            upper_yy = u * tau_yy_x + v * tau_yy_y - 2 * v_x * tau_xy - 2 * v_y * tau_yy
            upper_xy = u * tau_xy_x + v * tau_xy_y - u_x * tau_xy - u_y * tau_yy - tau_xx * v_x - tau_xy * v_y

            f_txx = f_PTT * tau_xx + Wi * upper_xx + (alpha * Wi / beta_poly) * (tau_xx**2 + tau_xy**2) - 2.0 * beta_poly * u_x
            f_tyy = f_PTT * tau_yy + Wi * upper_yy + (alpha * Wi / beta_poly) * (tau_xy**2 + tau_yy**2) - 2.0 * beta_poly * v_y
            f_txy = f_PTT * tau_xy + Wi * upper_xy + (alpha * Wi / beta_poly) * tau_xy * (tau_xx + tau_yy) - beta_poly * (u_y + v_x)
        else:
            f_txx = torch.zeros_like(u)
            f_tyy = torch.zeros_like(u)
            f_txy = torch.zeros_like(u)

        return f_u, f_v, f_txx, f_tyy, f_txy

    def compute_pde_losses(self, model, x, w_momentum=1.0, w_constitutive=1.0):
        """Calcola separatamente loss momentum e constitutive normalizzate per componente."""
        f_u, f_v, f_txx, f_tyy, f_txy = self.compute_residuals(model, x, w_momentum, w_constitutive)
        loss_m = 0.5 * (f_u**2 + f_v**2).mean()
        
        if self.var_weights is not None:
            loss_c = (
                (f_txx**2 / self.var_weights['tau_xx']) +
                (f_tyy**2 / self.var_weights['tau_yy']) +
                (f_txy**2 / self.var_weights['tau_xy'])
            ).mean() / 3.0
        else:
            loss_c = (f_txx**2 + f_tyy**2 + f_txy**2).mean() / 3.0
            
        return loss_m, loss_c

    def pde_loss_weighted(self, model, x, w_momentum, w_constitutive):
        """Loss PDE pesata per staged training."""
        loss_m, loss_c = self.compute_pde_losses(model, x, w_momentum, w_constitutive)
        return w_momentum * loss_m + w_constitutive * loss_c

    def pde_loss(self, model, x):
        """Loss PDE con pesi momentum/constitutive di default (tutto attivo)."""
        return self.pde_loss_weighted(model, x, W_MOMENTUM, W_CONSTITUTIVE)

    def data_loss(self, model, x, uv_target, var_w):
        """Loss dati: solo u, v (Goal 1, semi-inverso)."""
        u, v, _, _ = self.get_velocity(model, x)
        loss_u = weighted_mse(u, uv_target[:, 0:1], var_w['u'])
        loss_v = weighted_mse(v, uv_target[:, 1:2], var_w['v'])
        return 0.5 * (loss_u + loss_v)

    def boundary_loss(self, model, bc_data, var_w, active_bcs=None):
        """
        BC per 4rollmill:
          Walls: Dirichlet u=v=0
          Roll1-4: Dirichlet u, v da CSV
          PressurePoint: Dirichlet p da CSV
        """
        total_loss = torch.tensor(0.0, device=DEVICE, dtype=next(model.parameters()).dtype)

        for group_name, gd in bc_data.items():
            x_bc = gd['xy'].clone().requires_grad_(True)
            u, v, p, tau = self.get_velocity(model, x_bc)
            g_loss = torch.tensor(0.0, device=DEVICE, dtype=next(model.parameters()).dtype)

            if group_name == 'Walls':
                # u = v = 0
                if active_bcs is None or 'u' in active_bcs:
                    g_loss += weighted_mse(u, torch.zeros_like(u), var_w['u'])
                if active_bcs is None or 'v' in active_bcs:
                    g_loss += weighted_mse(v, torch.zeros_like(v), var_w['v'])

            elif group_name in ('Roll1', 'Roll2', 'Roll3', 'Roll4'):
                # u, v da CSV
                if active_bcs is None or 'u' in active_bcs:
                    g_loss += weighted_mse(u, gd['fields']['u'], var_w['u'])
                if active_bcs is None or 'v' in active_bcs:
                    g_loss += weighted_mse(v, gd['fields']['v'], var_w['v'])

            elif group_name == 'PressurePoint':
                # p da CSV
                if active_bcs is None or 'p' in active_bcs:
                    g_loss += weighted_mse(p, gd['fields']['p'], var_w['p'])

            total_loss += g_loss

        return total_loss

    def clamp_params(self):
        """Vincoli fisici sui parametri con logging delle variazioni."""
        if self.inverse_mode:
            with torch.no_grad():
                old_mu_s = self.mu_s.item()
                old_mu_p = self.mu_p.item()
                old_lam = self.lam.item()
                old_eps = self.eps.item()
                old_alpha = self.alpha.item()

                # Clamp normale per parametri che possono arrivare a zero
                self.eps.clamp_(min=0.0)
                self.alpha.clamp_(min=0.0)

                # Clamping con softplus per parametri strettamente positivi se scendono sotto le soglie minime
                if self.mu_s.item() < MIN_MU_S:
                    self.mu_s.copy_(torch.nn.functional.softplus(self.mu_s))
                if self.mu_p.item() < MIN_MU_P:
                    self.mu_p.copy_(torch.nn.functional.softplus(self.mu_p))
                if self.lam.item() < MIN_LAM:
                    self.lam.copy_(torch.nn.functional.softplus(self.lam))

                # Debug report se i parametri cambiano
                changes = []
                if self.mu_s.item() != old_mu_s:
                    changes.append(f"mu_s: {old_mu_s:.6e} -> {self.mu_s.item():.6e} (Softplus clamp)")
                if self.mu_p.item() != old_mu_p:
                    changes.append(f"mu_p: {old_mu_p:.6e} -> {self.mu_p.item():.6e} (Softplus clamp)")
                if self.lam.item() != old_lam:
                    changes.append(f"lam: {old_lam:.6e} -> {self.lam.item():.6e} (Softplus clamp)")
                if self.eps.item() != old_eps:
                    changes.append(f"eps: {old_eps:.6e} -> {self.eps.item():.6e} (Clamp)")
                if self.alpha.item() != old_alpha:
                    changes.append(f"alpha: {old_alpha:.6e} -> {self.alpha.item():.6e} (Clamp)")

                if changes:
                    print(f"  [DEBUG CLAMP] I parametri fisici sono stati aggiornati:\n    " + "\n    ".join(changes))

    def log_params(self):
        """Restituisce i parametri correnti come dict di float."""
        return {
            'mu_s': self.mu_s.item() if isinstance(self.mu_s, nn.Parameter) or torch.is_tensor(self.mu_s) else self.mu_s,
            'mu_p': self.mu_p.item() if isinstance(self.mu_p, nn.Parameter) or torch.is_tensor(self.mu_p) else self.mu_p,
            'lam': self.lam.item() if isinstance(self.lam, nn.Parameter) or torch.is_tensor(self.lam) else self.lam,
            'eps': self.eps.item() if isinstance(self.eps, nn.Parameter) or torch.is_tensor(self.eps) else self.eps,
            'alpha': self.alpha.item() if isinstance(self.alpha, nn.Parameter) or torch.is_tensor(self.alpha) else self.alpha,
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
        keys_plot = ['l2_u', 'l2_p', 'l2_tau_xx', 'l2_tau_xy', 'l2_tau_yy', 'l2_tau_xx_masked', 'l2_tau_xy_masked', 'l2_tau_yy_masked']
        colors = {
            'l2_u': 'blue',
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


def plot_fields(model, physics, data, save_path):
    """Confronto campi predetti vs COMSOL."""
    model.eval()
    _dtype = next(model.parameters()).dtype
    with torch.set_grad_enabled(True):
        x_in = data['coords'].to(_dtype).clone().requires_grad_(True)
        u_p, v_p, p_p, tau_p = physics.get_velocity(model, x_in, create_graph=False)

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
    cmaps_field = {
        'u': 'inferno',
        'p': 'viridis',
        'tau_xx': 'plasma',
        'tau_xy': 'plasma',
        'tau_yy': 'plasma'
    }

    fig, axs = plt.subplots(len(field_names), 3, figsize=(18, 4 * len(field_names)))
    for i, fn in enumerate(field_names):
        ex = exacts[fn].numpy().astype(np.float64)
        pr = preds[fn].numpy().astype(np.float64)

        # Errore relativo (in %) con cutoff a 10.0%
        abs_err = np.abs(ex - pr)
        rel_err = np.zeros_like(ex)
        max_val = np.max(np.abs(ex))
        thr = max(0.05 * max_val, 1e-8)
        m = np.abs(ex) > thr
        if m.sum() > 0:
            rel_err[m] = (abs_err[m] / np.abs(ex[m])) * 100.0

        # Usiamo i limiti condivisi per COMSOL e PINN
        vmin, vmax = min(ex.min(), pr.min()), max(ex.max(), pr.max())
        cmap = cmaps_field[fn]

        im0 = axs[i, 0].tricontourf(triang, ex, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[i, 0].set_title(f'{fn} (COMSOL)')
        axs[i, 0].set_aspect('equal')
        plt.colorbar(im0, ax=axs[i, 0])

        im1 = axs[i, 1].tricontourf(triang, pr, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[i, 1].set_title(f'{fn} (PINN)')
        axs[i, 1].set_aspect('equal')
        plt.colorbar(im1, ax=axs[i, 1])

        im2 = axs[i, 2].tricontourf(triang, rel_err, levels=50, cmap='jet', vmin=0.0, vmax=10.0)
        axs[i, 2].set_title(f'{fn} (Rel. Error %)')
        axs[i, 2].set_aspect('equal')
        plt.colorbar(im2, ax=axs[i, 2], label='%')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [PLOT] Campi salvati in {save_path}")


# ============================================================================
# 6. TRAINING LOOP (Staged Training)
# ============================================================================
def sample_minibatch(xy, targets, batch_size):
    """Campionamento random di un mini-batch."""
    if batch_size is None or batch_size >= xy.shape[0]:
        return xy, targets
    idx = torch.randperm(xy.shape[0], device=DEVICE)[:batch_size]
    return xy[idx], targets[idx] if targets is not None else None


def train(model, physics, data):
    """Training completo: Adam staged/non-staged + L-BFGS staged/non-staged."""
    history = SimpleHistory()

    xy_all = data['coords']
    uv_all = data['uv_data']
    var_w = data['var_weights']
    bc_data = data['boundary_groups']

    # Calcolo epoca di cambio fase
    half_epochs = int(ADAM_EPOCHS * 1.1)

    def build_optimizer(net_params, steps):
        if physics.inverse_mode:
            phys_params = [p for p in physics.parameters() if p.requires_grad]
            groups = [
                {'params': net_params, 'lr': BASE_LR},
                {'params': phys_params, 'lr': BASE_LR * PARAM_LR_FACTOR},
            ]
        else:
            groups = [
                {'params': net_params, 'lr': BASE_LR},
            ]
        opt = torch.optim.Adam(groups, eps=ADAM_EPS)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=1e-6)
        return opt, sch

    # --- INIZIALIZZAZIONE ADDESTRATORE ---
    if STAGED_TRAINING:
        print(f"\n{'='*60}")
        print(f"FASE 1 ADAM (Cinematica e Reologia): {half_epochs} epoche")
        print(f"{'='*60}")
        # Attivi solo model_psi e model_tau, model_p congelato
        for p in model.parameters():
            p.requires_grad = False
        for p in model.model_psi.parameters():
            p.requires_grad = True
        for p in model.model_tau.parameters():
            p.requires_grad = True
    else:
        print(f"\n{'='*60}")
        print(f"FASE ADAM UNICA (Tutto Attivo): {ADAM_EPOCHS} epoche")
        print(f"{'='*60}")
        # Tutti i modelli attivi
        for p in model.parameters():
            p.requires_grad = True

    # Parametri fisici congelati all'inizio se siamo in inverse_mode
    if physics.inverse_mode:
        for pname in ['mu_s', 'mu_p', 'lam', 'eps', 'alpha']:
            getattr(physics, pname).requires_grad_(False)

    net_params = [p for p in model.parameters() if p.requires_grad]
    optimizer, scheduler = build_optimizer(net_params, half_epochs if STAGED_TRAINING else ADAM_EPOCHS)

    pbar = tqdm(range(ADAM_EPOCHS), desc="Adam", mininterval=2.0)
    for epoch in pbar:
        # Sblocco parametri fisici dopo warmup (solo se inverse_mode attivo)
        if physics.inverse_mode and epoch == WARMUP_UNLOCK_EPOCH:
            print(f"\n  [Warmup Stage 1] Sblocco mu_s, mu_p, lam (epoca {epoch})")
            physics.mu_s.requires_grad_(True)
            physics.mu_p.requires_grad_(True)
            physics.lam.requires_grad_(True)
            net_params = [p for p in model.parameters() if p.requires_grad]
            steps_remaining = (half_epochs - WARMUP_UNLOCK_EPOCH) if STAGED_TRAINING else (ADAM_EPOCHS - WARMUP_UNLOCK_EPOCH)
            optimizer, scheduler = build_optimizer(net_params, steps_remaining)

        # Se STAGED_TRAINING attivo, cambio fase a metà delle epoche Adam
        if STAGED_TRAINING and epoch == half_epochs:
            print(f"\n{'='*60}")
            print(f"FASE 2 ADAM (Dinamica): {ADAM_EPOCHS - half_epochs} epoche")
            print(f"{'='*60}")
            # Attivi solo model_psi e model_p, model_tau congelato
            for p in model.parameters():
                p.requires_grad = False
            for p in model.model_psi.parameters():
                p.requires_grad = True
            for p in model.model_p.parameters():
                p.requires_grad = True

            # Parametri fisici congelati
            if physics.inverse_mode:
                for pname in ['mu_s', 'mu_p', 'lam', 'eps', 'alpha']:
                    getattr(physics, pname).requires_grad_(False)

            net_params = [p for p in model.parameters() if p.requires_grad]
            optimizer, scheduler = build_optimizer(net_params, ADAM_EPOCHS - half_epochs)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Accumulazione deterministica del gradiente in chunk per prevenire la saturazione della VRAM
        chunk_size = CHUNK_SIZE_ADAM
        total_points = xy_all.shape[0]

        # Pesi equazioni e BC in base allo Staged Training
        if STAGED_TRAINING:
            if epoch < half_epochs:
                active_bcs = ['u', 'v', 'tau_xx', 'tau_xy', 'tau_yy']
                pde_w_momentum = 0.0
                pde_w_constitutive = W_CONSTITUTIVE
            else:
                active_bcs = ['u', 'v', 'p']
                pde_w_momentum = W_MOMENTUM
                pde_w_constitutive = 0.0
        else:
            active_bcs = None  # Tutto attivo
            pde_w_momentum = W_MOMENTUM
            pde_w_constitutive = W_CONSTITUTIVE

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if physics.inverse_mode:
            trainable_params += [p for p in physics.parameters() if p.requires_grad]

        # 1. Calcolo di Data Loss e PDE Loss in chunk ed accumulazione dei gradienti per risparmiare memoria
        d_loss_accum = 0.0
        p_loss_accum = 0.0
        
        for i in range(0, total_points, chunk_size):
            xc = xy_all[i:i + chunk_size]
            yc = uv_all[i:i + chunk_size]
            w_chunk = xc.shape[0] / total_points
            
            # Data loss per questo chunk — backward immediato prima di costruire il grafo PDE
            # (evita RuntimeError: backward through the graph a second time)
            dl = physics.data_loss(model, xc, yc, var_w)
            d_loss_accum += dl.item() * w_chunk
            (W_DATA * dl * w_chunk).backward(inputs=trainable_params)
            
            # PDE loss per questo chunk — grafo costruito ex-novo dopo che dl è stato liberato
            xph = xc.clone().requires_grad_(True)
            pl = physics.pde_loss_weighted(model, xph, pde_w_momentum, pde_w_constitutive)
            p_loss_accum += pl.item() * w_chunk
            (W_PHYSICS * pl * w_chunk).backward(inputs=trainable_params)
            
        # 2. Calcolo di Boundary Loss (dataset piccolo, sicuro per la VRAM)
        b_loss = physics.boundary_loss(model, bc_data, var_w, active_bcs=active_bcs)
        b_loss_val = b_loss.item()
        (W_BC * b_loss).backward(inputs=trainable_params)

        # 3. Calcolo della loss totale per il logging
        total_loss_val = W_DATA * d_loss_accum + W_BC * b_loss_val + W_PHYSICS * p_loss_accum

        # 4. Clipping del gradiente e step
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        
        if physics.inverse_mode:
            phys_clip = [p for p in [physics.mu_s, physics.mu_p, physics.lam, physics.eps, physics.alpha] if p.requires_grad]
            if phys_clip:
                torch.nn.utils.clip_grad_norm_(phys_clip, PARAM_CLIP_NORM)

        optimizer.step()
        if physics.inverse_mode:
            physics.clamp_params()
        scheduler.step()

        # Logging veloce ad ogni epoca
        if (epoch + 1) % 10 == 0 or epoch == 0 or (STAGED_TRAINING and (epoch + 1) == half_epochs):
            params = physics.log_params()
            with torch.no_grad():
                l2_errs = compute_l2_errors(model, physics, data)

            history.update(epoch, {
                'total': total_loss_val,
                'data': d_loss_accum,
                'bc': b_loss_val,
                'pde': p_loss_accum,
                'param_mu_s': params['mu_s'],
                'param_mu_p': params['mu_p'],
                'param_lam': params['lam'],
                'param_eps': params['eps'],
                'param_alpha': params['alpha'],
                'l2_u': l2_errs['u'],
                'l2_p': l2_errs['p'],
                'l2_tau_xx': l2_errs['tau_xx'],
                'l2_tau_xy': l2_errs['tau_xy'],
                'l2_tau_yy': l2_errs['tau_yy'],
                'l2_tau_xx_masked': l2_errs['tau_xx_masked'],
                'l2_tau_xy_masked': l2_errs['tau_xy_masked'],
                'l2_tau_yy_masked': l2_errs['tau_yy_masked'],
            })
            
            # Monitoraggio periodico dettagliato di L2(u), Momentum e Constitutive
            if (epoch + 1) % PRINT_EVERY == 0 or epoch == 0:
                with torch.no_grad():
                    losses_eval = evaluate_final_losses(model, physics, data)
                model.train()
                print(
                    f"  [Epoch {epoch+1:>4d}/{ADAM_EPOCHS}] Detailed Status Report:\n"
                    f"    L2 Errors:         u={l2_errs['u']:.4f} | p={l2_errs['p']:.4f} | "
                    f"tau_xx={l2_errs['tau_xx']:.4f} | tau_xy={l2_errs['tau_xy']:.4f} | tau_yy={l2_errs['tau_yy']:.4f}\n"
                    f"    L2 Masked Stress:  tau_xx_masked={l2_errs['tau_xx_masked']:.4f} | tau_xy_masked={l2_errs['tau_xy_masked']:.4f} | tau_yy_masked={l2_errs['tau_yy_masked']:.4f}\n"
                    f"    Losses:            Data={losses_eval['Data Loss']:.2e} | BC={losses_eval['Boundary Loss']:.2e} (u={losses_eval['BC_u']:.2e}, v={losses_eval['BC_v']:.2e}, p={losses_eval['BC_p']:.2e})\n"
                    f"                       Momentum={losses_eval['Momentum Loss']:.2e} | Constitutive={losses_eval['Constitutive Loss']:.2e}\n"
                    f"    Mean Abs Res:      |f_u|={losses_eval['Mean Abs f_u']:.2e} | |f_v|={losses_eval['Mean Abs f_v']:.2e} | "
                    f"|f_txx|={losses_eval['Mean Abs f_txx']:.2e} | |f_txy|={losses_eval['Mean Abs f_txy']:.2e} | |f_tyy|={losses_eval['Mean Abs f_tyy']:.2e}",
                    flush=True
                )
        
        pbar.set_postfix({
            'Loss': f"{total_loss_val:.2e}",
            'Data': f"{d_loss_accum:.2e}",
            'BC': f"{b_loss_val:.2e}",
            'PDE': f"{p_loss_accum:.2e}",
            'LR': f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    pbar.close()

    # ==================================================================
    # FASE L-BFGS: Joint Fine-Tuning (FP64)
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"FASE L-BFGS: {int(LBFGS_MAX_ITERS)} iterazioni (FP64)")
    print(f"{'='*60}")

    # Cast a FP64 per precisione scientifica (centralizzato)
    convert_to_fp64(model, physics, data)
    
    # Aggiorna i riferimenti locali dopo il cast FP64
    xy_all = data['coords']
    uv_all = data['uv_data']
    bc_data = data['boundary_groups']

    # Controllo/Assert sui dtype prima di L-BFGS
    for p_name, param in model.named_parameters():
        assert param.dtype == torch.float64, f"[Assert FP64] Parametro {p_name} del modello non è float64: {param.dtype}"
    for p_name, param in physics.named_parameters():
        assert param.dtype == torch.float64, f"[Assert FP64] Parametro {p_name} della fisica non è float64: {param.dtype}"
    for b_name, buf in physics.named_buffers():
        assert buf.dtype == torch.float64, f"[Assert FP64] Buffer {b_name} della fisica non è float64: {buf.dtype}"
    assert xy_all.dtype == torch.float64, f"[Assert FP64] Dati xy_all non sono float64: {xy_all.dtype}"
    assert uv_all.dtype == torch.float64, f"[Assert FP64] Dati uv_all non sono float64: {uv_all.dtype}"
    for gname, gd in bc_data.items():
        assert gd['xy'].dtype == torch.float64, f"[Assert FP64] BC {gname} xy non è float64"
        assert gd['norm'].dtype == torch.float64, f"[Assert FP64] BC {gname} norm non è float64"
        for fname, fval in gd['fields'].items():
            assert fval.dtype == torch.float64, f"[Assert FP64] BC {gname} field {fname} non è float64"

    # Report di debug dtypes
    print(f"\n[DEBUG REPORT PRE-L-BFGS]")
    print(f"  - Modello: convertito a {next(model.parameters()).dtype}")
    print(f"  - Fisica:  convertito a {physics.mu_s.dtype}")
    print(f"  - coords:  {xy_all.dtype} (shape: {xy_all.shape})")
    print(f"  - uv_data: {uv_all.dtype} (shape: {uv_all.shape})")
    print(f"  - BC data dtypes:")
    for gname, gd in bc_data.items():
        fields_str = ", ".join([f"{k}:{v.dtype}" for k, v in gd['fields'].items()])
        print(f"    * {gname:<13s} -> xy: {gd['xy'].dtype} | norm: {gd['norm'].dtype} | fields: [{fields_str}]")

    # Tutti i modelli attivi
    for p in model.parameters():
        p.requires_grad = True
        
    if physics.inverse_mode:
        physics.mu_s.requires_grad_(True)
        physics.mu_p.requires_grad_(True)
        physics.lam.requires_grad_(True)
        all_params = list(model.parameters()) + [physics.mu_s, physics.mu_p, physics.lam]
    else:
        all_params = list(model.parameters())

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

        # Costruzione della loss dati come tensore scalare vivo (backward immediato per risparmiare memoria)
        loss_data = torch.tensor(0.0, device=DEVICE, dtype=torch.float64)
        for i in range(0, xy_all.shape[0], chunk_size):
            xc = xy_all[i:i + chunk_size]
            yc = uv_all[i:i + chunk_size]
            dl = physics.data_loss(model, xc, yc, var_w)
            w = xc.shape[0] / xy_all.shape[0]
            chunk_loss = W_DATA * dl * w
            loss_data = loss_data + chunk_loss
            chunk_loss.backward()

        # Loss BC
        loss_bc = physics.boundary_loss(model, bc_data, var_w, active_bcs=None)
        loss_bc_weighted = W_BC * loss_bc
        loss_bc_weighted.backward()

        # Costruzione della loss PDE come tensore scalare vivo (backward immediato per evitare OOM)
        loss_pde = torch.tensor(0.0, device=DEVICE, dtype=torch.float64)
        loss_m_val = 0.0
        loss_c_val = 0.0
        
        for i in range(0, xph_full.shape[0], chunk_size):
            xc = xph_full[i:i + chunk_size]
            w = xc.shape[0] / xph_full.shape[0]
            
            # Calcola le loss locali
            loss_m, loss_c = physics.compute_pde_losses(model, xc)
            loss_m_val += loss_m.item() * w
            loss_c_val += loss_c.item() * w
            
            chunk_loss = W_PHYSICS * (W_MOMENTUM * loss_m + W_CONSTITUTIVE * loss_c) * w
            loss_pde = loss_pde + chunk_loss
            chunk_loss.backward()

        # Loss totale come tensore vivo connesso al grafo
        total_loss = loss_data + loss_bc_weighted + loss_pde
        total_val = total_loss.item()

        if l_it[0] % 10 == 0 or l_it[0] == int(LBFGS_MAX_ITERS) - 1:
            params = physics.log_params()
            with torch.no_grad():
                l2_errs = compute_l2_errors(model, physics, data)

            history.update(ADAM_EPOCHS + l_it[0], {
                'total': total_val,
                'data': (loss_data / W_DATA).item(),
                'bc': loss_bc.item(),
                'pde': (loss_pde / W_PHYSICS).item(),
                'loss_momentum': loss_m_val,
                'loss_constitutive': loss_c_val,
                'param_mu_s': params['mu_s'],
                'param_mu_p': params['mu_p'],
                'param_lam': params['lam'],
                'param_eps': params['eps'],
                'param_alpha': params['alpha'],
                'l2_u': l2_errs['u'],
                'l2_p': l2_errs['p'],
                'l2_tau_xx': l2_errs['tau_xx'],
                'l2_tau_xy': l2_errs['tau_xy'],
                'l2_tau_yy': l2_errs['tau_yy'],
                'l2_tau_xx_masked': l2_errs['tau_xx_masked'],
                'l2_tau_xy_masked': l2_errs['tau_xy_masked'],
                'l2_tau_yy_masked': l2_errs['tau_yy_masked'],
            })

        l_it[0] += 1
        pbar.update(1)
        pbar.set_postfix({'Loss': f'{total_val:.2e}'})

        return total_loss

    optimizer_lbfgs.step(closure)
    if physics.inverse_mode:
        physics.clamp_params()
    pbar.close()

    return history


# ============================================================================
# 7. METRICHE E MAIN
# ============================================================================
def evaluate_final_losses(model, physics, data):
    """Calcola le loss finali valutate sul dataset completo in chunk per evitare OOM (usando la precisione corrente)."""
    model.eval()
    _dtype = next(model.parameters()).dtype
    
    xy_all = data['coords'].to(_dtype)
    uv_all = data['uv_data'].to(_dtype)
    var_w = data['var_weights']
    bc_data = data['boundary_groups']
    
    bc_data_typed = {}
    for gname, gd in bc_data.items():
        bc_data_typed[gname] = {
            'xy': gd['xy'].to(_dtype),
            'norm': gd['norm'].to(_dtype),
            'fields': {k: v.to(_dtype) for k, v in gd['fields'].items()}
        }

    chunk_size = 2000

    with torch.set_grad_enabled(True):
        # Data loss (chunked)
        d_loss_val = 0.0
        for i in range(0, xy_all.shape[0], chunk_size):
            xc = xy_all[i:i + chunk_size]
            yc = uv_all[i:i + chunk_size]
            dl = physics.data_loss(model, xc, yc, var_w)
            w = xc.shape[0] / xy_all.shape[0]
            d_loss_val += dl.item() * w
            
        # BC loss splits
        bc_u_val = 0.0
        bc_v_val = 0.0
        bc_p_val = 0.0
        
        for group_name, gd in bc_data_typed.items():
            x_bc = gd['xy'].clone().requires_grad_(True)
            u, v, p, tau = physics.get_velocity(model, x_bc)
            
            if group_name == 'Walls':
                bc_u_val += weighted_mse(u, torch.zeros_like(u), var_w['u']).item()
                bc_v_val += weighted_mse(v, torch.zeros_like(v), var_w['v']).item()
            elif group_name in ('Roll1', 'Roll2', 'Roll3', 'Roll4'):
                bc_u_val += weighted_mse(u, gd['fields']['u'], var_w['u']).item()
                bc_v_val += weighted_mse(v, gd['fields']['v'], var_w['v']).item()
            elif group_name == 'PressurePoint':
                bc_p_val += weighted_mse(p, gd['fields']['p'], var_w['p']).item()
                
        b_loss_val = bc_u_val + bc_v_val + bc_p_val
        
        # PDE losses (chunked)
        loss_m_val = 0.0
        loss_c_val = 0.0
        abs_fu_sum = 0.0
        abs_fv_sum = 0.0
        abs_ftxx_sum = 0.0
        abs_ftxy_sum = 0.0
        abs_ftyy_sum = 0.0
        
        for i in range(0, xy_all.shape[0], chunk_size):
            xc = xy_all[i:i + chunk_size].clone().requires_grad_(True)
            f_u, f_v, f_txx, f_tyy, f_txy = physics.compute_residuals(model, xc)
            
            loss_m = 0.5 * (f_u**2 + f_v**2).mean()
            loss_c = (f_txx**2 + f_tyy**2 + f_txy**2).mean() / 3.0
            
            w = xc.shape[0] / xy_all.shape[0]
            loss_m_val += loss_m.item() * w
            loss_c_val += loss_c.item() * w
            
            abs_fu_sum += f_u.abs().mean().item() * w
            abs_fv_sum += f_v.abs().mean().item() * w
            abs_ftxx_sum += f_txx.abs().mean().item() * w
            abs_ftxy_sum += f_txy.abs().mean().item() * w
            abs_ftyy_sum += f_tyy.abs().mean().item() * w
            
        pde_loss_val = W_MOMENTUM * loss_m_val + W_CONSTITUTIVE * loss_c_val
        total_loss_val = W_DATA * d_loss_val + W_BC * b_loss_val + W_PHYSICS * pde_loss_val
        
    return {
        'Data Loss': d_loss_val,
        'Boundary Loss': b_loss_val,
        'BC_u': bc_u_val,
        'BC_v': bc_v_val,
        'BC_p': bc_p_val,
        'Momentum Loss': loss_m_val,
        'Constitutive Loss': loss_c_val,
        'Total PDE Loss': pde_loss_val,
        'Total Loss': total_loss_val,
        'Mean Abs f_u': abs_fu_sum,
        'Mean Abs f_v': abs_fv_sum,
        'Mean Abs f_txx': abs_ftxx_sum,
        'Mean Abs f_txy': abs_ftxy_sum,
        'Mean Abs f_tyy': abs_ftyy_sum
    }


def compute_l2_errors(model, physics, data):
    """Calcola L2 relative errors per tutti i campi (globali e mascherati per lo stress)."""
    model.eval()
    _dtype = next(model.parameters()).dtype
    with torch.set_grad_enabled(True):
        xi = data['coords'].to(_dtype).clone().requires_grad_(True)
        u_p, v_p, p_p, tau_p = physics.get_velocity(model, xi, create_graph=False)

    preds = {'u': u_p, 'p': p_p, 'tau_xx': tau_p[:, 0:1], 'tau_xy': tau_p[:, 1:2], 'tau_yy': tau_p[:, 2:3]}
    exacts = {'u': data['u'], 'p': data['p'], 'tau_xx': data['tau_xx'], 'tau_xy': data['tau_xy'], 'tau_yy': data['tau_yy']}

    errors = {}
    for fn in preds:
        pr = preds[fn].detach().view(-1)
        ex = exacts[fn].to(pr.dtype).view(-1)
        norm_ex = torch.norm(ex, 2)
        l2 = (torch.norm(pr - ex, 2) / norm_ex).item() if norm_ex > 1e-10 else 0.0
        errors[fn] = l2

    # Calcolo errore L2 mascherato sulle zone dove lo stress esatto non è nullo (soglia al 5% del max)
    exact_txx = exacts['tau_xx'].to(_dtype)
    exact_txy = exacts['tau_xy'].to(_dtype)
    exact_tyy = exacts['tau_yy'].to(_dtype)
    tau_magnitude = torch.sqrt(exact_txx**2 + exact_txy**2 + exact_tyy**2)
    max_tau = torch.max(tau_magnitude).item()
    threshold = 0.05 * max_tau
    mask = (tau_magnitude >= threshold).view(-1)
    
    # Se per qualche motivo nessun punto supera la soglia, evitiamo errori usando tutti i punti
    if torch.sum(mask).item() == 0:
        mask = torch.ones_like(mask, dtype=torch.bool)

    for fn in ['tau_xx', 'tau_xy', 'tau_yy']:
        pr_m = preds[fn].detach().view(-1)[mask]
        ex_m = exacts[fn].to(pr_m.dtype).view(-1)[mask]
        norm_ex_m = torch.norm(ex_m, 2)
        l2_m = (torch.norm(pr_m - ex_m, 2) / norm_ex_m).item() if norm_ex_m > 1e-10 else 0.0
        errors[f"{fn}_masked"] = l2_m

    return errors


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print(f"Dtype iniziale: {torch.get_default_dtype()}")
    print(f"Dataset: {DATASET_PATH}")
    print()
    print("=" * 60)
    print("DEBUG REPORT CONFIGURAZIONE INIZIALE:")
    print("  - Formula Weighted MSE: Mean( ((pred - target) ** 2) / var )")
    print("  - Definizione U_ref:    max(sqrt(u_raw**2 + v_raw**2))")
    print("=" * 60)
    print()

    # 1. Carica dati
    data = load_data()

    # 2. Costruisci modelli con scaling
    model = CombinedModel(p_scale=data['p_scale'], tau_scale=data['tau_scale']).to(DEVICE)
    model.model_psi.apply(init_weights_xavier)
    model.model_p.apply(init_weights_xavier)
    model.model_tau.apply(init_weights_xavier)
    initialize_last_layer_zero(model.model_p)
    initialize_last_layer_zero(model.model_tau)

    physics = Physics(U_ref=data['U_ref'], H_ref=data['H'], var_weights=data['var_weights'], inverse_mode=INVERSE_PROBLEM).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModello: {total_params:,} parametri totali")
    if INVERSE_PROBLEM:
        print(f"Modalità: PROBLEMA INVERSO (Reologia da identificare)")
        print(f"  Guess iniziali: mu_s={GUESS_MU_S}, mu_p={GUESS_MU_P}, lam={GUESS_LAM}")
        print(f"  Valori veri:    mu_s={MU_S_TRUE}, mu_p={MU_P_TRUE}, lam={LAM_TRUE}")
    else:
        print(f"Modalità: PROBLEMA DIRETTO (Parametri fisici bloccati ai valori veri)")
        print(f"  Valori fissi:   mu_s={MU_S_TRUE}, mu_p={MU_P_TRUE}, lam={LAM_TRUE}")

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

    # Report finale dettagliato valutato sul dataset completo
    final_losses = evaluate_final_losses(model, physics, data)
    print(f"\n{'='*60}")
    print("REPORT FINALE DETTAGLIATO (Dataset Completo)")
    print(f"{'='*60}")
    print(f"  Data Loss:          {final_losses['Data Loss']:.6e}")
    print(f"  Boundary Loss:      {final_losses['Boundary Loss']:.6e}")
    print(f"    - BC_u:           {final_losses['BC_u']:.6e}")
    print(f"    - BC_v:           {final_losses['BC_v']:.6e}")
    print(f"    - BC_p:           {final_losses['BC_p']:.6e}")
    print(f"  Momentum Loss:      {final_losses['Momentum Loss']:.6e}")
    print(f"  Constitutive Loss:  {final_losses['Constitutive Loss']:.6e}")
    print(f"  Total PDE Loss:     {final_losses['Total PDE Loss']:.6e}")
    print(f"  Total Loss:         {final_losses['Total Loss']:.6e}")
    
    print(f"\nResidui medi assoluti (Dataset Completo):")
    print(f"  |f_u|:              {final_losses['Mean Abs f_u']:.6e}")
    print(f"  |f_v|:              {final_losses['Mean Abs f_v']:.6e}")
    print(f"  |f_txx|:            {final_losses['Mean Abs f_txx']:.6e}")
    print(f"  |f_txy|:            {final_losses['Mean Abs f_txy']:.6e}")
    print(f"  |f_tyy|:            {final_losses['Mean Abs f_tyy']:.6e}")

    errors = compute_l2_errors(model, physics, data)
    print(f"\nL2 Relative Errors:")
    for fn, err in errors.items():
        print(f"  {fn:>8s}: {err:.6f}")

    # 5. Plot
    history.plot_losses(str(OUTPUT_DIR / 'loss_history.png'))
    history.plot_params(str(OUTPUT_DIR / 'params_evolution.png'))
    history.plot_l2_errors(str(OUTPUT_DIR / 'l2_errors_history.png'))
    plot_fields(model, physics, data, str(OUTPUT_DIR / 'fields_comparison.png'))

    print(f"\nPlot salvati in: {OUTPUT_DIR}")
    print("Done!")
