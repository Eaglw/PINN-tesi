import os

import numpy as np
import torch
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch


def convert_to_fp64(model, physics, data):
    """
    Converte in modo centralizzato modello, fisica e dati a FP64 prima di L-BFGS.
    Include una logica ricorsiva interna per scorrere i dizionari dei dati.
    """

    def _cast_dict_to_double(
        d,
    ):  # funzione interna per ricorsione, per ciclare su tutti i dati e bc.
        for key, value in d.items():
            if isinstance(value, torch.Tensor):
                d[key] = value.double()
            elif isinstance(value, dict):
                # Se è un dizionario (es. boundary_groups), richiama se stessa
                _cast_dict_to_double(value)

    _cast_dict_to_double(data)
    model.double()
    physics.double()


def weighted_mse(pred, target, var):
    """Formula esplicita per la weighted MSE normalizzata tramite la varianza."""
    return torch.mean(((pred - target) ** 2) / var)


def load_data(use_fp64=False):
    """Carica il dataset COMSOL, adimensionalizza, estrae boundary groups e prepara i tensori."""
    print("=" * 60)
    print("Caricamento dataset COMSOL (4rollmill)...")
    pt_dtype = torch.float64 if use_fp64 else torch.float32

    data_np = np.loadtxt(
        str(DATASET_PATH), dtype=np.float64, delimiter=",", comments="%"
    )
    assert data_np.shape[1] >= 8, f"Attese almeno 8 colonne, trovate {data_np.shape[1]}"
    N = data_np.shape[0]

    # Unpacking delle colonne
    x_raw, y_raw = data_np[:, 0], data_np[:, 1]
    u_raw, v_raw = data_np[:, 2], data_np[:, 3]
    p_raw = data_np[:, 4]
    txx_raw, txy_raw, tyy_raw = data_np[:, 5], data_np[:, 6], data_np[:, 7]

    # --- 2. Scale di riferimento ---
    y_min, y_max = y_raw.min(), y_raw.max()
    x_min, x_max = x_raw.min(), x_raw.max()
    H = max(y_max - y_min, 1e-9)
    U_ref = max(float(np.max(np.sqrt(u_raw**2 + v_raw**2))), 1e-9)
    mu_tot = MU_S_TRUE + MU_P_TRUE
    p_ref = mu_tot * U_ref / H
    tau_ref = (
        mu_tot * U_ref / H
    )  # scaling viscoso, quindi riferimento uguale per eq adimensionali.

    # --- 3. Adimensionalizzazione (Vettorizzata) ---
    x_nd, y_nd = (
        (x_raw - x_min) / H,
        (y_raw - y_min) / H,
    )  # uso xmin per spostare il sistema di riferimento a 0,0

    # Raggruppiamo i campi scalari/vettoriali adimensionalizzati in un dizionario
    fields_nd = {
        "u": u_raw / U_ref,
        "v": v_raw / U_ref,
        "p": p_raw / p_ref,
        "tau_xx": txx_raw / tau_ref,
        "tau_xy": txy_raw / tau_ref,
        "tau_yy": tyy_raw / tau_ref,
    }

    # --- 4. Creazione Tensori (DRY Approach) ---
    # Gestiamo 'coords' separatamente perché ha dimensione [N, 2]
    coords = torch.tensor(np.column_stack([x_nd, y_nd]), dtype=pt_dtype, device=DEVICE)

    # Generiamo dinamicamente tutti i tensori colonna [N, 1] tramite dict comprehension
    tensors = {
        name: torch.tensor(arr, dtype=pt_dtype, device=DEVICE).reshape(-1, 1)
        for name, arr in fields_nd.items()
    }

    # Output scales
    p_scale = max(float(np.abs(fields_nd["p"]).max()), 1.0)
    tau_scale = max(
        float(
            max(
                np.abs(fields_nd["tau_xx"]).max(),
                np.abs(fields_nd["tau_xy"]).max(),
                np.abs(fields_nd["tau_yy"]).max(),
            )
        ),
        1.0,
    )

    # --- 5. Calcolo Varianze Automatizzato ---
    var_weights = {
        name: max(t.var().item(), VARIANCE_EPS) for name, t in tensors.items()
    }

    # --- 6. Stampa Statistiche ---
    print(f"  Punti totali: {N}")
    print(f"  H={H:.6e}, U_ref={U_ref:.6e}, p_ref={p_ref:.6e}")
    print(
        f"  Re={RHO * U_ref * H / mu_tot:.4f}, Wi={LAM_TRUE * U_ref / H:.4f}, beta={MU_S_TRUE / mu_tot:.4f}"
    )
    print(f"  [Output Scaling] p_scale={p_scale:.4f}, tau_scale={tau_scale:.4f}")

    # --- 7. Boundary Groups ---
    boundary_groups = _extract_boundary_groups(
        coords, x_raw, y_raw, x_min, y_min, H, tensors
    )

    print("=" * 60)

    # Ricostruiamo il dizionario finale scompattando 'tensors'
    return {
        "coords": coords,
        **tensors,  # Scompatta u, v, p, tau_xx, tau_xy, tau_yy direttamente nel dizionario
        "uv_data": torch.cat([tensors["u"], tensors["v"]], dim=1),
        "tau_data": torch.cat(
            [tensors["tau_xx"], tensors["tau_xy"], tensors["tau_yy"]], dim=1
        ),
        "var_weights": var_weights,
        "boundary_groups": boundary_groups,
        "U_ref": U_ref,
        "H": H,
        "p_scale": p_scale,
        "tau_scale": tau_scale,
    }


def _extract_boundary_groups(
    coords, x_raw, y_raw, x_min, y_min, H, fields, pt_dtype=torch.float32
):
    """
    Estrae i boundary groups e calcola le normali analizzando la topologia della mesh COMSOL.
    Non applica le BC, ma prepara i tensori per la Loss Function.
    """
    mphtxt_path = str(DATASET_PATH).replace(".csv", "_geom.mphtxt")
    if not os.path.isfile(mphtxt_path):
        mphtxt_path = str(DATASET_PATH).replace(".csv", ".mphtxt")
    if not os.path.isfile(mphtxt_path):
        raise FileNotFoundError(f"File mesh .mphtxt non trovato per {DATASET_PATH}")

    with open(mphtxt_path, "r") as f:
        lines = [line.strip() for line in f]

    # --- 1. Parsing Helper Functions ---
    def find_section(keyword):
        for i, line in enumerate(lines):
            if keyword in line:
                return i
        return -1

    def parse_elements(start_keyword):
        start_idx = find_section(start_keyword)
        if start_idx == -1:
            return [], []

        num_elems, elem_idx, entity_idx = 0, -1, -1

        for i in range(start_idx, len(lines)):
            if "# number of elements" in lines[i]:
                num_elems = int(lines[i].split("#")[0].strip())
            elif "# Elements" in lines[i]:
                elem_idx = i + 1
            elif "# Geometric entity indices" in lines[i]:
                entity_idx = i + 1
                break

        elements = []
        if elem_idx != -1:
            for i in range(num_elems):
                parts = lines[elem_idx + i].split()
                elements.append([int(p) for p in parts[:3]])

        entities = []
        if entity_idx != -1:
            for i in range(num_elems):
                entities.append(int(lines[entity_idx + i].strip()))

        return elements, entities

    # --- 2. Esecuzione Parsing ---
    v_start = find_section("# Mesh vertex coordinates") + 1
    num_vertices = int(
        lines[find_section("# number of mesh vertices")].split("#")[0].strip()
    )

    vertices_raw = np.array(
        [[float(x) for x in lines[v_start + i].split()] for i in range(num_vertices)]
    )

    edg_elements, edg_entity_indices = parse_elements("edg")
    tri_elements, _ = parse_elements("tri")

    # Parsing Selections (Nomi dei bordi)
    selections = {}
    for idx, line in enumerate(lines):
        if "Selection # class" in line:
            label = lines[idx + 2].split("#")[0].strip().split()[-1]
            num_ent = int(lines[idx + 5].split("#")[0].strip())
            ent_start = idx + 7
            selections[label] = [
                int(lines[i]) for i in range(ent_start, ent_start + num_ent)
            ]

    # --- 3. Topologia (Nodo -> Triangoli adiacenti) ---
    node_to_tri = {}
    for t_idx, tri in enumerate(tri_elements):
        for nid in tri:
            node_to_tri.setdefault(nid, []).append(t_idx)

    edge_to_nodes = {}
    for edg, eid in zip(edg_elements, edg_entity_indices):
        edge_to_nodes.setdefault(eid, set()).update(edg[:2])

    # --- 4. Matching spaziale KD-Tree e calcolo normali ---
    coords_np = coords.cpu().numpy()
    x_min_mesh, y_min_mesh = vertices_raw[:, 0].min(), vertices_raw[:, 1].min()

    vertices_nd = np.column_stack(
        [(vertices_raw[:, 0] - x_min_mesh) / H, (vertices_raw[:, 1] - y_min_mesh) / H]
    )

    tree_csv = cKDTree(coords_np)
    dists_nearest, _ = tree_csv.query(coords_np, k=2)
    tol_match = max(np.median(dists_nearest[:, 1]) * 0.5, 1e-6)

    boundary_groups = {}

    for label, entities in selections.items():
        sel_nodes = {
            nid
            for eid in entities
            if eid in edge_to_nodes
            for nid in edge_to_nodes[eid]
        }
        if not sel_nodes:
            continue

        group_normals = np.zeros((num_vertices, 2))
        eset = set(entities)

        for edg, eid in zip(edg_elements, edg_entity_indices):
            if eid not in eset:
                continue
            ga, gb = edg[0], edg[1]

            common_tris = set(node_to_tri.get(ga, [])).intersection(
                node_to_tri.get(gb, [])
            )
            if not common_tris:
                continue

            tri = tri_elements[list(common_tris)[0]]
            g_opp = list(set(tri) - {ga, gb})
            if not g_opp:
                continue
            g_opp = g_opp[0]

            pa, pb, po = vertices_raw[ga], vertices_raw[gb], vertices_raw[g_opp]
            tangent = pb - pa
            length = np.linalg.norm(tangent)

            if length > 0:
                t_unit = tangent / length
                to_int = po - (0.5 * (pa + pb))
                n_cand = np.array([t_unit[1], -t_unit[0]])

                if np.dot(n_cand, to_int) > 0:
                    n_cand = -n_cand

                group_normals[ga] += n_cand
                group_normals[gb] += n_cand

        global_idx, global_norm = [], []
        for nid in sel_nodes:
            dist, cidx = tree_csv.query(vertices_nd[nid])
            if dist < tol_match:
                global_idx.append(cidx)
                n_mag = np.linalg.norm(group_normals[nid])
                global_norm.append(
                    group_normals[nid] / n_mag if n_mag > 1e-9 else group_normals[nid]
                )

        if global_idx:
            boundary_groups[label] = {
                "indices": torch.tensor(global_idx, dtype=torch.long, device=DEVICE),
                "xy": coords[global_idx].to(DEVICE).to(pt_dtype),
                "norm": torch.tensor(
                    np.array(global_norm), dtype=pt_dtype, device=DEVICE
                ),
                "fields": {
                    k: v[global_idx].to(DEVICE).to(pt_dtype) for k, v in fields.items()
                },
            }
            print(f"  Boundary '{label}': {len(global_idx)} nodi identificati")

    # --- 5. Controllo di sicurezza sulla Pressione (Warning Non Bloccante) ---
    # Controlliamo in modo case-insensitive per intercettare varianti come 'pressurepoint', 'Pressure_Point', ecc.
    has_pressure_bc = any("pressure" in k.lower() for k in boundary_groups.keys())

    if not has_pressure_bc:
        print(
            "\n  [WARNING] Nessun gruppo riconducibile a 'PressurePoint' trovato nei boundary groups."
        )
        if boundary_groups:
            # Preleviamo il primo gruppo disponibile (es. 'Walls')
            first_group_name = list(boundary_groups.keys())[0]
            first_group = boundary_groups[first_group_name]
            
            # Creiamo il PressurePoint prendendo il primo nodo di questo gruppo
            boundary_groups["PressurePoint"] = {
                "indices": first_group["indices"][0:1],
                "xy": first_group["xy"][0:1],
                "norm": first_group["norm"][0:1],
                "fields": {k: v[0:1] for k, v in first_group["fields"].items()},
            }
            print(f"  [INFO] Creato 'PressurePoint' automatico usando 1 nodo da '{first_group_name}' per ancorare la pressione.")
        else:
            print(
                "  [WARNING] Ricorda di applicare un ancoraggio per la pressione (Dirichlet) o un profilo"
            )
            print(
                "  [WARNING] definito nel calcolo della loss per evitare che il problema Navier-Stokes"
            )
            print("  [WARNING] risulti mal posto (pressione fluttuante).")

    return boundary_groups




def generate_all_diagnostics(model, physics, data, save_dir, chunk_size=7000):
    """
    Esegue l'inferenza spaziale UNA SOLA VOLTA e smista i dati ai plotter specializzati.
    Pattern 'Data Provider' per abbattere l'overhead sulla GPU.
    Processato in chunk per evitare picchi di VRAM e OOM.
    """
    model.eval()
    _dtype = next(model.parameters()).dtype

    print("\n  [DIAGNOSTICA] Esecuzione inferenza e generazione grafici in corso...")

    x_in_all = data["coords"].to(_dtype)
    total_points = x_in_all.shape[0]

    u_list, p_list, tau_p_list = [], [], []

    # 1. Inferenza Unica (A chunk)
    with torch.set_grad_enabled(True):
        for i in range(0, total_points, chunk_size):
            x_in = x_in_all[i : i + chunk_size].clone().requires_grad_(True)
            u_p, v_p, p_p, tau_p = physics.get_velocity(model, x_in, create_graph=False)
            
            u_list.append(u_p.detach())
            p_list.append(p_p.detach())
            tau_p_list.append(tau_p.detach())

    tau_p_full = torch.cat(tau_p_list, dim=0)

    # Pacchetto predizioni disaccoppiato dal modello
    predictions = {
        "u": torch.cat(u_list, dim=0),
        "p": torch.cat(p_list, dim=0),
        "tau_xx": tau_p_full[:, 0],
        "tau_xy": tau_p_full[:, 1],
        "tau_yy": tau_p_full[:, 2],
    }

    # 2. Distribuzione dei dati ai plotter
    plot_fields(predictions, data, save_path=f"{save_dir}/global_fields.png")
    plot_high_stress_regions(predictions, data, save_path=f"{save_dir}/high_stress.png")
    print("  [DIAGNOSTICA] Pipeline visiva completata con successo.")


def plot_fields(predictions, data, save_path):
    """Confronto campi predetti vs COMSOL con contour plot (riceve dati pre-calcolati)."""

    # Helper per estrazione sicura su CPU come array numpy 1D
    def _to_np(tensor_or_array):
        if torch.is_tensor(tensor_or_array):
            return tensor_or_array.detach().cpu().view(-1).numpy().astype(np.float64)
        return np.asarray(tensor_or_array).reshape(-1).astype(np.float64)

    preds = {
        "u": _to_np(predictions["u"]),
        "p": _to_np(predictions["p"]),
        "tau_xx": _to_np(predictions["tau_xx"]),
        "tau_xy": _to_np(predictions["tau_xy"]),
        "tau_yy": _to_np(predictions["tau_yy"]),
    }
    exacts = {
        "u": _to_np(data["u"]),
        "p": _to_np(data["p"]),
        "tau_xx": _to_np(data["tau_xx"]),
        "tau_xy": _to_np(data["tau_xy"]),
        "tau_yy": _to_np(data["tau_yy"]),
    }

    x_np, y_np = _to_np(data["coords"][:, 0]), _to_np(data["coords"][:, 1])
    triang = mtri.Triangulation(x_np, y_np)

    # Gestione Mesh Masking
    try:
        rollers = []
        for rname in ["Roll1", "Roll2", "Roll3", "Roll4"]:
            if rname in data["boundary_groups"]:
                rxy = data["boundary_groups"][rname]["xy"].cpu().numpy()
                rcenter = np.mean(rxy, axis=0)
                rradius = np.mean(
                    np.hypot(rxy[:, 0] - rcenter[0], rxy[:, 1] - rcenter[1])
                )
                rollers.append((rcenter, rradius * 0.98))

        if rollers:
            cx = np.mean(x_np[triang.triangles], axis=1)
            cy = np.mean(y_np[triang.triangles], axis=1)
            mask = np.zeros(len(triang.triangles), dtype=bool)
            for rcenter, rradius in rollers:
                dists = np.hypot(cx - rcenter[0], cy - rcenter[1])
                mask = mask | (dists < rradius)
            if mask.any():
                triang.set_mask(mask)
    except Exception as e:
        print(f"  [WARNING] Errore nel masking geometrico per il plotting: {e}")

    field_names = ["u", "p", "tau_xx", "tau_xy", "tau_yy"]
    cmaps = {
        "u": "inferno",
        "p": "viridis",
        "tau_xx": "plasma",
        "tau_xy": "plasma",
        "tau_yy": "plasma",
    }

    fig, axs = plt.subplots(len(field_names), 3, figsize=(18, 4 * len(field_names)))

    for i, fn in enumerate(field_names):
        ex, pr = exacts[fn], preds[fn]

        # Errore relativo (in %) con cutoff intelligente per evitare divisioni per 0
        abs_err = np.abs(ex - pr)
        rel_err = np.zeros_like(ex)
        thr = max(0.05 * np.max(np.abs(ex)), 1e-8)
        m = np.abs(ex) > thr
        if m.any():
            rel_err[m] = (abs_err[m] / np.abs(ex[m])) * 100.0

        vmin, vmax = min(ex.min(), pr.min()), max(ex.max(), pr.max())
        cmap = cmaps[fn]

        # Configurazione matrice per i 3 subplot della riga
        plot_configs = [
            (ex, f"{fn} (COMSOL)", cmap, vmin, vmax, None),
            (pr, f"{fn} (PINN)", cmap, vmin, vmax, None),
            (rel_err, f"{fn} (Rel. Error %)", "jet", 0.0, 10.0, "%"),
        ]

        for col, (data_arr, title, c_map, c_min, c_max, cb_label) in enumerate(
            plot_configs
        ):
            im = axs[i, col].tricontourf(
                triang, data_arr, levels=50, cmap=c_map, vmin=c_min, vmax=c_max
            )
            axs[i, col].set_title(title)
            axs[i, col].set_aspect("equal")
            plt.colorbar(im, ax=axs[i, col], label=cb_label if cb_label else "")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"    -> Contour fields salvati in {save_path}")


def plot_high_stress_regions(predictions, data, save_path):
    """Scatter plot degli stress confinato solo alle regioni ad alto sforzo (>50% max)."""

    def _to_np(tensor_or_array):
        if torch.is_tensor(tensor_or_array):
            return tensor_or_array.detach().cpu().view(-1).numpy().astype(np.float64)
        return np.asarray(tensor_or_array).reshape(-1).astype(np.float64)

    # Estraiamo gli stress esatti dal dataset grezzo (tau_data ha shape N, 3)
    tau_true = (
        data["tau_data"].cpu().numpy()
        if torch.is_tensor(data["tau_data"])
        else data["tau_data"]
    )

    # Ricomponiamo lo stress predetto (shape N, 3) dai campi di predictions
    tau_pred_xx = _to_np(predictions["tau_xx"])[:, None]
    tau_pred_xy = _to_np(predictions["tau_xy"])[:, None]
    tau_pred_yy = _to_np(predictions["tau_yy"])[:, None]
    tau_pred = np.hstack([tau_pred_xx, tau_pred_xy, tau_pred_yy])

    mag_true = np.linalg.norm(tau_true, axis=1)
    mask = mag_true > 0.5 * mag_true.max()

    if not mask.any():
        print(
            "    -> [SKIPPED] Nessun punto supera il 50% dello stress massimo per il plot locale."
        )
        return

    x_np = _to_np(data["coords"][:, 0])[mask]
    y_np = _to_np(data["coords"][:, 1])[mask]
    t_true_masked = tau_true[mask]
    t_pred_masked = tau_pred[mask]

    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    field_names = ["tau_xx", "tau_xy", "tau_yy"]

    for i, fn in enumerate(field_names):
        ex = t_true_masked[:, i]
        pr = t_pred_masked[:, i]

        ratio = np.zeros_like(ex)
        valid = np.abs(ex) > 1e-5
        ratio[valid] = pr[valid] / ex[valid]

        plot_configs = [
            (ex, f"{fn} True (High Stress)", "plasma", None, None, None),
            (pr, f"{fn} Pred (High Stress)", "plasma", None, None, None),
            (ratio, f"{fn} Pred/True Ratio", "bwr", -1.0, 3.0, "Ratio"),
        ]

        for col, (data_arr, title, c_map, c_min, c_max, cb_label) in enumerate(
            plot_configs
        ):
            sc = axs[i, col].scatter(
                x_np, y_np, c=data_arr, cmap=c_map, vmin=c_min, vmax=c_max, s=5
            )
            axs[i, col].set_title(title)
            axs[i, col].set_aspect("equal")
            plt.colorbar(sc, ax=axs[i, col], label=cb_label if cb_label else "")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"    -> High-stress scatter salvato in {save_path}")

def get_optimal_chunk_size(
    phase=1,
    safety_factor=0.8, 
    default_cpu_chunk=5000,
    min_chunk=1000,
    max_chunk=50000
):
    """
    Calcola la dimensione ottimale del chunk in base alla VRAM totale disponibile
    e alla fase di addestramento.
    """
    if not torch.cuda.is_available():
        print(f"  [VRAM Check] Nessuna GPU trovata. Uso chunk size di default per CPU: {default_cpu_chunk}")
        return default_cpu_chunk

    try:
        # Usa la memoria totale del dispositivo per avere un calcolo invariante
        # rispetto alla VRAM temporaneamente occupata in quel momento
        total_vram = torch.cuda.get_device_properties(0).total_memory
        
        # Riserviamo solo una frazione (safety_factor) della VRAM totale
        usable_vram = total_vram * safety_factor
        
        # Stime basate su test empirici (24GB GPU, 50k chunk)
        if phase == 1:
            bytes_per_point_estimate = 245000  # ~11.5 GB per 50k punti
        elif phase == 2:
            bytes_per_point_estimate = 490000  # ~23.0 GB per 50k punti
        else:
            bytes_per_point_estimate = 1500000 # Fase 3 (L-BFGS FP64): stima molto prudente
            
        # Calcolo chunk size
        calculated_chunk = int(usable_vram / bytes_per_point_estimate)
        
        # Limitiamo i valori estremi per stabilità
        optimal_chunk = max(min_chunk, min(calculated_chunk, max_chunk))
        
        total_gb = total_vram / (1024**3)
        print(f"  [VRAM Check Fase {phase}] VRAM Totale GPU: {total_gb:.1f} GB")
        print(f"  [VRAM Check Fase {phase}] Chunk size stimato: {calculated_chunk} -> Limitato a: {optimal_chunk}")
        
        return optimal_chunk
        
    except Exception as e:
        print(f"  [WARNING] Impossibile calcolare la VRAM dinamicamente ({e}). Uso fallback: {default_cpu_chunk}")
        return default_cpu_chunkhunk

