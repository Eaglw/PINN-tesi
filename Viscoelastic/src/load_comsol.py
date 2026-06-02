import torch
import numpy as np
import os
import matplotlib.pyplot as plt


def load_comsol_csv(csv_path, params, device='cpu'):
    """
    Carica un file CSV esportato da COMSOL e restituisce un dataset adimensionalizzato
    compatibile con il framework Viscoelastic PINN.

    Il CSV deve avere colonne: x, y, u, v, p, tau_xx, tau_xy, tau_yy.
    Le righe che iniziano con '%' vengono ignorate (header COMSOL).

    Args:
        csv_path: Percorso al file CSV esportato da COMSOL.
        params: Dizionario con i parametri fisici del fluido:
            - mu_s: Viscosità del solvente [Pa·s]
            - mu_p: Viscosità polimerica [Pa·s]
            - lam: Tempo di rilassamento [s]
            - eps: Parametro di mobilità PTT (default 0)
            - alpha: Parametro di mobilità Giesekus (default 0)
            - rho: Densità del fluido [kg/m³] (default 1.0)
        device: Device torch su cui creare i tensori ('cpu' o 'cuda').

    Returns:
        dict: Dataset con campi adimensionalizzati, indici boundary,
              scale di riferimento e parametri adimensionali.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File COMSOL non trovato: {csv_path}")

    # --- 1. Parsing del CSV (skip righe con '%') ---
    rows = []
    with open(csv_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('%') or len(stripped) == 0:
                continue
            rows.append(stripped)

    # Parsing numerico con numpy
    data_np = np.loadtxt(rows, dtype=np.float64, delimiter=',')
    if data_np.ndim == 1:
        data_np = data_np.reshape(1, -1)

    assert data_np.shape[1] >= 8, (
        f"Attese almeno 8 colonne (x, y, u, v, p, tau_xx, tau_xy, tau_yy), "
        f"trovate {data_np.shape[1]}"
    )

    N = data_np.shape[0]

    # Estrazione colonne raw (dimensionali)
    x_raw = data_np[:, 0]
    y_raw = data_np[:, 1]
    u_raw = data_np[:, 2]
    v_raw = data_np[:, 3]
    p_raw = data_np[:, 4]
    tau_xx_raw = data_np[:, 5]
    tau_xy_raw = data_np[:, 6]
    tau_yy_raw = data_np[:, 7]

    # --- 2. Scale di riferimento ---
    y_min, y_max = float(y_raw.min()), float(y_raw.max())
    x_min, x_max = float(x_raw.min()), float(x_raw.max())
    H = y_max - y_min if (y_max - y_min) > 1e-9 else 1.0
    L = x_max - x_min if (x_max - x_min) > 1e-9 else 1.0
    U_ref = float(np.abs(u_raw).max())
    if U_ref < 1e-9:
        U_ref = 1.0

    mu_s = params['mu_s']
    mu_p = params['mu_p']
    lam = params.get('lam', 1.0)
    eps = params.get('eps', 0.0)
    alpha = params.get('alpha', 0.0)
    rho = params.get('rho', 1.0)
    mu_tot = mu_s + mu_p

    p_ref = mu_tot * U_ref / H      # Scala viscosa di pressione
    tau_ref = mu_tot * U_ref / H    # Stessa scala per gli sforzi

    # --- 3. Adimensionalizzazione (spostando l'origine a (0,0) per coerenza) ---
    x_nd = (x_raw - x_min) / H
    y_nd = (y_raw - y_min) / H
    u_nd = u_raw / U_ref
    v_nd = v_raw / U_ref
    p_nd = p_raw / p_ref
    tau_xx_nd = tau_xx_raw / tau_ref
    tau_xy_nd = tau_xy_raw / tau_ref
    tau_yy_nd = tau_yy_raw / tau_ref

    # Conversione a tensori torch
    coords = torch.tensor(np.stack([x_nd, y_nd], axis=1), dtype=torch.float32, device=device)
    u_t = torch.tensor(u_nd, dtype=torch.float32, device=device).reshape(-1, 1)
    v_t = torch.tensor(v_nd, dtype=torch.float32, device=device).reshape(-1, 1)
    p_t = torch.tensor(p_nd, dtype=torch.float32, device=device).reshape(-1, 1)
    tau_xx_t = torch.tensor(tau_xx_nd, dtype=torch.float32, device=device).reshape(-1, 1)
    tau_xy_t = torch.tensor(tau_xy_nd, dtype=torch.float32, device=device).reshape(-1, 1)
    tau_yy_t = torch.tensor(tau_yy_nd, dtype=torch.float32, device=device).reshape(-1, 1)

    # --- 4. Identificazione nodi boundary (basata su intervalli assoluti) ---
    tol = 1e-6
    L_nd = L / H  # Lunghezza adimensionale del canale
    H_nd = 1.0    # Altezza adimensionale (H/H = 1)

    # Indici candidati per inlet e outlet (prioritari sugli spigoli)
    inlet_mask = np.abs(x_raw - x_min) < tol
    outlet_mask = np.abs(x_raw - x_max) < tol

    # Bottom e top: escludono inlet e outlet per evitare duplicati
    bottom_mask = (np.abs(y_raw - y_min) < tol) & (~inlet_mask) & (~outlet_mask)
    top_mask = (np.abs(y_raw - y_max) < tol) & (~inlet_mask) & (~outlet_mask)

    # Internal: tutto il resto
    boundary_mask = inlet_mask | outlet_mask | bottom_mask | top_mask
    internal_mask = ~boundary_mask

    # Conversione a indici torch
    inlet_idx = torch.tensor(np.where(inlet_mask)[0], dtype=torch.long, device=device)
    outlet_idx = torch.tensor(np.where(outlet_mask)[0], dtype=torch.long, device=device)
    bottom_idx = torch.tensor(np.where(bottom_mask)[0], dtype=torch.long, device=device)
    top_idx = torch.tensor(np.where(top_mask)[0], dtype=torch.long, device=device)
    all_boundary_idx = torch.tensor(np.where(boundary_mask)[0], dtype=torch.long, device=device)
    internal_idx = torch.tensor(np.where(internal_mask)[0], dtype=torch.long, device=device)

    # --- 5. Parametri adimensionali ---
    Re = rho * U_ref * H / mu_tot
    Wi = lam * U_ref / H
    beta = mu_s / mu_tot

    # --- 6. Stampa riepilogo ---
    print("=" * 60)
    print(f"COMSOL Dataset caricato: {os.path.basename(csv_path)}")
    print(f"  Punti totali:    {N}")
    print(f"  Inlet:           {len(inlet_idx)}")
    print(f"  Outlet:          {len(outlet_idx)}")
    print(f"  Bottom wall:     {len(bottom_idx)}")
    print(f"  Top wall:        {len(top_idx)}")
    print(f"  Internal:        {len(internal_idx)}")
    print(f"  --- Scale di riferimento ---")
    print(f"  H  = {H:.6e} m")
    print(f"  L  = {L:.6e} m")
    print(f"  U_ref = {U_ref:.6e} m/s")
    print(f"  p_ref = {p_ref:.6e} Pa")
    print(f"  tau_ref = {tau_ref:.6e} N/m²")
    print(f"  --- Parametri adimensionali ---")
    print(f"  Re   = {Re:.4f}")
    print(f"  Wi   = {Wi:.4f}")
    print(f"  beta = {beta:.4f}")
    print(f"  --- Range campi (adimensionali) ---")
    print(f"  u*:      [{u_t.min().item():.4f}, {u_t.max().item():.4f}]")
    print(f"  v*:      [{v_t.min().item():.4f}, {v_t.max().item():.4f}]")
    print(f"  p*:      [{p_t.min().item():.4f}, {p_t.max().item():.4f}]")
    print(f"  tau_xx*: [{tau_xx_t.min().item():.4f}, {tau_xx_t.max().item():.4f}]")
    print(f"  tau_xy*: [{tau_xy_t.min().item():.4f}, {tau_xy_t.max().item():.4f}]")
    print(f"  tau_yy*: [{tau_yy_t.min().item():.4f}, {tau_yy_t.max().item():.4f}]")
    print("=" * 60)

    # --- 7. Assemblaggio dataset ---
    dataset = {
        'csv_path': csv_path,
        'coords': coords,
        'u': u_t,
        'v': v_t,
        'p': p_t,
        'tau_xx': tau_xx_t,
        'tau_xy': tau_xy_t,
        'tau_yy': tau_yy_t,
        'boundary_indices': {
            'inlet': inlet_idx,
            'outlet': outlet_idx,
            'bottom': bottom_idx,
            'top': top_idx,
            'all': all_boundary_idx,
        },
        'internal_indices': internal_idx,
        'scales': {
            'H': H,
            'L': L,
            'U_ref': U_ref,
            'p_ref': p_ref,
            'tau_ref': tau_ref,
        },
        'params': {
            'mu_s': mu_s,
            'mu_p': mu_p,
            'lam': lam,
            'eps': eps,
            'alpha': alpha,
            'rho': rho,
        },
        'nondim_params': {
            'Re': Re,
            'Wi': Wi,
            'beta': beta,
        },
    }

    return dataset


def extract_boundary_groups_from_comsol(dataset, device='cpu'):
    """
    Estrae i gruppi di nodi geometrici di contorno, identificando inlet, outlet e pareti (walls)
    tramite parsing nativo del file mesh COMSOL (.mphtxt).
    
    L'algoritmo opera in tre fasi principali:
    
    Fase 1: Lettura Geometria e Calcolo Topologico delle Normali
      - Legge le coordinate dei nodi e la topologia degli elementi bidimensionali (tri2)
        e di bordo (edg2) direttamente dal file .mphtxt.
      - Calcola la normale geometrica esterna unitaria n = (nx, ny) per ciascun nodo di bordo
        in modo topologico (direzione opposta al terzo vertice del triangolo adiacente).
        
    Fase 2: Parsing delle Selezioni (Boundary Selections)
      - Estrae gli oggetti Selection e le definizioni delle etichette (es. 'Inlet', 'Outlet', 'Walls')
        in fondo al file .mphtxt.
      - Se non sono presenti selezioni esportate nel file (es. canale rettilineo semplice), le
        ricostruisce geometricamente raggruppando gli Edge ID in base alla posizione spaziale dei nodi.
        
    Fase 3: Accoppiamento Geometrico con i Nodi del CSV
      - Associa i nodi di bordo identificati nel file mesh con i corrispettivi nodi del dataset CSV
        tramite ricerca del vicino più prossimo con tolleranza geometrica.
      - Associa i campi fisici del CSV e le normali calcolate ad ogni gruppo.
      - Genera un plot diagnostico 'plots/boundary_classification_[dataset].png' con colori distinti.
    
    Returns:
        dict: Dizionario contenente i gruppi di contorno con chiavi corrispondenti
              alle etichette delle selezioni (es. 'Inlet', 'Outlet', 'Walls').
              Ciascun gruppo è un dizionario con chiavi:
              - 'indices': Tensor degli indici globali dei nodi nel dataset
              - 'xy': Tensor delle coordinate adimensionali (M, 2)
              - 'norm': Tensor delle normali esterne (M, 2)
              - 'fields': Dizionario con i campi fisici associati {'u', 'v', 'p', 'tau_xx', 'tau_xy', 'tau_yy'}
    """
    csv_path = dataset.get('csv_path', None)
    mphtxt_path = None
    if csv_path:
        possible_paths = [
            csv_path.replace('.csv', '_geom.mphtxt'),
            csv_path.replace('.csv', '.mphtxt'),
        ]
        for p in possible_paths:
            if os.path.isfile(p):
                mphtxt_path = p
                break

    if mphtxt_path is None:
        raise FileNotFoundError(
            f"File mesh COMSOL (.mphtxt) non trovato per il dataset {csv_path}. "
            f"Assicurati che esista un file .mphtxt con lo stesso nome del CSV o con suffisso '_geom.mphtxt'."
        )

    print(f"\n[AUTO-BC] Caricamento geometria e selezioni da file COMSOL .mphtxt:")
    print(f"  File: {os.path.basename(mphtxt_path)}")

    # --- 1. LETTURA DEL FILE .MPHTXT ---
    with open(mphtxt_path, 'r') as f:
        lines = [line.strip() for line in f]

    # Parsing coordinate vertici
    num_vertices = 0
    vertices_start = -1
    for idx, line in enumerate(lines):
        if '# number of mesh vertices' in line:
            num_vertices = int(line.split('#')[0].strip())
        elif '# Mesh vertex coordinates' in line:
            vertices_start = idx + 1

    if vertices_start == -1 or num_vertices == 0:
        raise ValueError("Impossibile trovare la sezione delle coordinate dei vertici nel file .mphtxt.")

    vertices_raw = []
    for i in range(num_vertices):
        parts = lines[vertices_start + i].split()
        vertices_raw.append([float(parts[0]), float(parts[1])])
    vertices_raw = np.array(vertices_raw)

    # Parsing elementi edg2 (Type #1) e Geometric entity indices
    edg_elements = []
    edg_entity_indices = []
    edg_start = -1
    for idx, line in enumerate(lines):
        if 'edg2 # type name' in line:
            edg_start = idx
            break

    if edg_start != -1:
        num_edg_elements = 0
        edg_elem_idx = -1
        # Trova il numero di elementi edg2
        for i in range(edg_start, len(lines)):
            if '# number of elements' in lines[i]:
                num_edg_elements = int(lines[i].split('#')[0].strip())
                break
        # Trova la riga iniziale "# Elements"
        for i in range(edg_start, len(lines)):
            if '# Elements' in lines[i]:
                edg_elem_idx = i + 1
                break
        
        if edg_elem_idx != -1:
            for i in range(num_edg_elements):
                parts = lines[edg_elem_idx + i].split()
                # edg2 ha 3 nodi per elemento (nodi d'angolo ed intermedi)
                edg_elements.append([int(parts[0]), int(parts[1]), int(parts[2])])

        # Trova Geometric entity indices per edg2 (Boundary/Edge ID geometrici)
        edg_entity_idx = -1
        for i in range(edg_elem_idx + num_edg_elements, len(lines)):
            if '# Geometric entity indices' in lines[i]:
                edg_entity_idx = i + 1
                break
        
        if edg_entity_idx != -1:
            for i in range(num_edg_elements):
                edg_entity_indices.append(int(lines[edg_entity_idx + i]))

    # Parsing elementi tri2 (Type #2) per ricostruzione topologia normali
    tri_elements = []
    tri_start = -1
    for idx, line in enumerate(lines):
        if 'tri2 # type name' in line:
            tri_start = idx
            break

    if tri_start != -1:
        num_tri_elements = 0
        tri_elem_idx = -1
        for i in range(tri_start, len(lines)):
            if '# number of elements' in lines[i]:
                num_tri_elements = int(lines[i].split('#')[0].strip())
                break
        for i in range(tri_start, len(lines)):
            if '# Elements' in lines[i]:
                tri_elem_idx = i + 1
                break
        
        if tri_elem_idx != -1:
            for i in range(num_tri_elements):
                parts = lines[tri_elem_idx + i].split()
                # tri2 ha 6 nodi per elemento. A noi interessano i primi 3 per la topologia geometrica
                tri_elements.append([int(parts[0]), int(parts[1]), int(parts[2])])

    # Parsing delle Selezioni (Selection)
    selections = {}
    idx = 0
    while idx < len(lines):
        if 'Selection # class' in lines[idx]:
            label = ""
            for i in range(idx + 1, idx + 10):
                if '# Label' in lines[i]:
                    raw_label = lines[i].split('#')[0].strip()
                    # Rimuove il prefisso di lunghezza della stringa se presente (es. '5 Walls' -> 'Walls')
                    parts = raw_label.split(maxsplit=1)
                    if len(parts) == 2 and parts[0].isdigit():
                        label = parts[1]
                    else:
                        label = raw_label
                    break
            
            num_entities = 0
            ent_start = -1
            for i in range(idx + 1, idx + 20):
                if '# Number of entities' in lines[i]:
                    num_entities = int(lines[i].split('#')[0].strip())
                    ent_start = i + 2  # Salta '# Entities'
                    break
            
            entities = []
            if ent_start != -1:
                for i in range(num_entities):
                    entities.append(int(lines[ent_start + i]))
            
            if label:
                selections[label] = entities
            idx = ent_start + num_entities
        else:
            idx += 1

    # Se non sono state trovate selezioni esplicite (es. canale semplice Oldroyd_geom.mphtxt),
    # le costruiamo geometricamente analizzando le coordinate dei nodi degli Edge ID
    if not selections and len(edg_entity_indices) > 0:
        print("  [INFO] Nessuna selezione geometrica trovata nel file .mphtxt. Generazione geometrica automatica...")
        inlet_edges = []
        outlet_edges = []
        walls_edges = []
        
        # Mappiamo gli ID degli edge a tutti i relativi nodi di bordo
        edge_nodes = {}
        for edg, edge_id in zip(edg_elements, edg_entity_indices):
            if edge_id not in edge_nodes:
                edge_nodes[edge_id] = set()
            edge_nodes[edge_id].update(edg)
            
        x_min_mesh = vertices_raw[:, 0].min()
        x_max_mesh = vertices_raw[:, 0].max()
        
        # Classificazione degli edge_id in base alla posizione dei loro nodi
        for edge_id, node_ids in edge_nodes.items():
            pts = vertices_raw[list(node_ids)]
            x_min_edge = pts[:, 0].min()
            x_max_edge = pts[:, 0].max()
            
            # Tolleranza per il matching geometrico dei contorni
            tol_bc_geom = 1e-6
            if abs(x_min_edge - x_min_mesh) < tol_bc_geom and abs(x_max_edge - x_min_mesh) < tol_bc_geom:
                inlet_edges.append(edge_id)
            elif abs(x_min_edge - x_max_mesh) < tol_bc_geom and abs(x_max_edge - x_max_mesh) < tol_bc_geom:
                outlet_edges.append(edge_id)
            else:
                walls_edges.append(edge_id)
                
        if inlet_edges:
            selections['Inlet'] = inlet_edges
        if outlet_edges:
            selections['Outlet'] = outlet_edges
        if walls_edges:
            selections['Walls'] = walls_edges
            
        print(f"  [INFO] Selezioni generate geometricamente: Inlet={inlet_edges}, Outlet={outlet_edges}, Walls={walls_edges}")


    # --- 2. PREPARAZIONE PER IL CALCOLO DELLE NORMALI ESTERNE ---
    # Costruiamo una mappa nodo -> triangoli adiacenti per velocizzare la ricerca
    node_to_triangles = {}
    for t_idx, tri in enumerate(tri_elements):
        for n_id in tri:
            if n_id not in node_to_triangles:
                node_to_triangles[n_id] = []
            node_to_triangles[n_id].append(t_idx)

    # --- 3. ACCOPPIAMENTO GEOMETRICO CON I NODI DEL CSV ---
    coords_np = dataset['coords'].cpu().numpy()
    H = dataset['scales']['H']
    
    # Coordinate adimensionali del file mphtxt (con shift robusto basato sui minimi della mesh)
    x_min_mesh = vertices_raw[:, 0].min()
    y_min_mesh = vertices_raw[:, 1].min()
    vertices_nd = np.stack([
        (vertices_raw[:, 0] - x_min_mesh) / H,
        (vertices_raw[:, 1] - y_min_mesh) / H
    ], axis=1)
    
    # [PUNTO B] Calcolo tolleranza adattiva: metà della distanza minima tra i nodi del CSV
    from scipy.spatial import cKDTree
    tree_csv = cKDTree(coords_np)
    dists_nearest, _ = tree_csv.query(coords_np, k=2) # k=2 perché il primo è il punto stesso
    min_dist_mesh = np.median(dists_nearest[:, 1])
    tol_match = max(min_dist_mesh * 0.5, 1e-6)
    print(f"  [INFO] Tolleranza matching adattiva impostata a: {tol_match:.2e} (Distanza mesh tipica: {min_dist_mesh:.2e})")

    boundary_groups = {}
    edge_to_mphtxt_nodes = {}
    for edg, edge_id in zip(edg_elements, edg_entity_indices):
        if edge_id not in edge_to_mphtxt_nodes:
            edge_to_mphtxt_nodes[edge_id] = set()
        edge_to_mphtxt_nodes[edge_id].update(edg)

    # Elaborazione delle selezioni e calcolo delle normali LOCALE a ciascun gruppo di contorno (evita sovrapposizioni d'angolo)
    for label, entities in selections.items():
        selection_nodes_mphtxt = set()
        for edge_id in entities:
            if edge_id in edge_to_mphtxt_nodes:
                selection_nodes_mphtxt.update(edge_to_mphtxt_nodes[edge_id])
                
        if len(selection_nodes_mphtxt) == 0:
            continue

        # Vettore di accumulo delle normali locale per questo specifico gruppo
        group_normals_accum = np.zeros((num_vertices, 2))
        entities_set = set(entities)

        for edg, edge_id in zip(edg_elements, edg_entity_indices):
            if edge_id not in entities_set:
                continue
                
            ga, gb, gmid = edg[0], edg[1], edg[2]
            
            # Trova il triangolo adiacente
            adj_tri_idx = None
            if ga in node_to_triangles and gb in node_to_triangles:
                common = set(node_to_triangles[ga]).intersection(node_to_triangles[gb])
                if common:
                    adj_tri_idx = list(common)[0]
                    
            if adj_tri_idx is None:
                continue
                
            tri = tri_elements[adj_tri_idx]
            g_opp = None
            for n_id in tri:
                if n_id != ga and n_id != gb:
                    g_opp = n_id
                    break
                    
            if g_opp is None:
                continue
                
            p_a = vertices_raw[ga]
            p_b = vertices_raw[gb]
            p_opp = vertices_raw[g_opp]
            
            tangent = p_b - p_a
            length = np.linalg.norm(tangent)
            if length > 0:
                tangent_unit = tangent / length
                p_mid = 0.5 * (p_a + p_b)
                to_internal = p_opp - p_mid
                
                n_candidate = np.array([tangent_unit[1], -tangent_unit[0]])
                if np.dot(n_candidate, to_internal) > 0:
                    n_candidate = -n_candidate
                    
                for g in [ga, gb, gmid]:
                    group_normals_accum[g] += n_candidate

        global_indices = []
        global_normals = []
        
        for n_id in selection_nodes_mphtxt:
            coords_nd = vertices_nd[n_id]
            dist, idx = tree_csv.query(coords_nd)
            
            if dist < tol_match:
                global_indices.append(idx)
                # Calcola e normalizza la normale localmente al gruppo
                n_vec = group_normals_accum[n_id]
                n_mag = np.linalg.norm(n_vec)
                global_normals.append(n_vec / n_mag if n_mag > 1e-9 else n_vec)
                
        if len(global_indices) == 0:
            print(f"  [WARNING] La selezione '{label}' non contiene nodi accoppiati nel dataset CSV.")
            continue
            
        boundary_groups[label] = {
            'indices': torch.tensor(global_indices, dtype=torch.long, device=device),
            'xy': dataset['coords'][global_indices].to(device),
            'norm': torch.tensor(np.array(global_normals), dtype=torch.float32, device=device),
            'fields': {k: dataset[k][global_indices].to(device) for k in ['u', 'v', 'p', 'tau_xx', 'tau_xy', 'tau_yy'] if k in dataset}
        }
        print(f"  Boundary '{label}': accoppiati {len(global_indices)} nodi con normali locali.")

    # --- 4. GENERAZIONE DEL PLOT DI DIAGNOSTICA DINAMICA ---
    try:
        plt.figure(figsize=(10, 5))
        # Plot di sfondo di tutti i nodi fluidi
        plt.scatter(coords_np[:, 0], coords_np[:, 1], color='#E0E0E0', s=2, alpha=0.5)
        
        # Mappatura dei colori per ciascuna label per garantire che siano distinti
        override_rules = [
            ('inlet', 'green'),
            ('outlet', 'red'),
            ('wall', 'blue'),
            ('cylinder', 'orange')
        ]
        color_pool = ['purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'gray', 'olive']
        
        label_to_color = {}
        assigned_colors = set()
        
        # Prima passata: assegna i colori preferiti se matchano le keyword
        for label in boundary_groups.keys():
            lbl_low = label.lower()
            for pattern, col in override_rules:
                if pattern in lbl_low:
                    label_to_color[label] = col
                    assigned_colors.add(col)
                    break
        
        # Seconda passata: assegna colori liberi e distinti dal pool per i confini rimanenti
        pool_idx = 0
        for label in boundary_groups.keys():
            if label not in label_to_color:
                chosen_color = None
                while pool_idx < len(color_pool):
                    col = color_pool[pool_idx]
                    pool_idx += 1
                    if col not in assigned_colors:
                        chosen_color = col
                        break
                if chosen_color is None:
                    chosen_color = color_pool[pool_idx % len(color_pool)]
                    pool_idx += 1
                
                label_to_color[label] = chosen_color
                assigned_colors.add(chosen_color)
        
        for label, group in boundary_groups.items():
            color = label_to_color[label]
            pts = group['xy'].cpu().numpy()
            norms = group['norm'].cpu().numpy()
            
            # Disegna i punti
            plt.scatter(pts[:, 0], pts[:, 1], color=color, s=12, label=label)
            # Disegna i vettori delle normali
            step = max(1, len(pts) // 100)
            plt.quiver(pts[::step, 0], pts[::step, 1], norms[::step, 0], norms[::step, 1],
                       color=color, scale=30, width=0.002, alpha=0.8)
                       
        plt.title(f"Guida Condizioni al Contorno: {os.path.basename(csv_path) if csv_path else ''}")
        plt.xlabel("x*")
        plt.ylabel("y*")
        plt.axis('equal')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        project_plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'plots')
        os.makedirs(project_plots_dir, exist_ok=True)
        if csv_path:
            spec_name = f"boundary_classification_{os.path.splitext(os.path.basename(csv_path))[0]}.png"
            plt.savefig(os.path.join(project_plots_dir, spec_name), dpi=150)
        else:
            plt.savefig(os.path.join(project_plots_dir, 'boundary_classification.png'), dpi=150)
        plt.close()
        print(f"  [GUIDA SAVED] Grafico di guida salvato in: {project_plots_dir}")
    except Exception as e:
        print(f"  [WARNING] Errore nella generazione del plot di guida: {e}")

    return boundary_groups


def prepare_training_data(dataset_path, comsol_params, num_data_subset, initial_dtype, device, variance_eps=1e-5):
    """
    Esegue la preparazione completa dei dati COMSOL per il training:
    - Caricamento e adimensionalizzazione
    - Calcolo delle varianze per normalizzazione
    - Estrazione di subset e gruppi di contorno geometrici
    - Pre-cast al tipo numerico richiesto
    """
    dataset = load_comsol_csv(dataset_path, comsol_params, device=device)
    
    # Cast dei campi al tipo richiesto
    for key in ['coords', 'u', 'v', 'p', 'tau_xx', 'tau_xy', 'tau_yy']:
        if key in dataset:
            dataset[key] = dataset[key].to(initial_dtype)
            
    xy_grid_flat = dataset['coords']
    u_exact = dataset['u']
    v_exact = dataset['v']
    p_exact = dataset['p']
    tau_xx_exact = dataset['tau_xx']
    tau_xy_exact = dataset['tau_xy']
    tau_yy_exact = dataset['tau_yy']
    
    import matplotlib.tri as tri
    x_np = xy_grid_flat[:, 0].cpu().numpy()
    y_np = xy_grid_flat[:, 1].cpu().numpy()
    triang = tri.Triangulation(x_np, y_np)
    
    # Applichiamo il masking basato su edge-length per rimuovere triangoli spuri
    # in geometrie non rettangolari o concave (ad esempio restringimenti)
    try:
        triangles = triang.triangles
        x_tri = x_np[triangles]
        y_tri = y_np[triangles]
        
        # Lunghezza dei tre lati di ciascun triangolo
        l1 = np.hypot(x_tri[:, 0] - x_tri[:, 1], y_tri[:, 0] - y_tri[:, 1])
        l2 = np.hypot(x_tri[:, 1] - x_tri[:, 2], y_tri[:, 1] - y_tri[:, 2])
        l3 = np.hypot(x_tri[:, 2] - x_tri[:, 0], y_tri[:, 2] - y_tri[:, 0])
        max_edge = np.maximum(np.maximum(l1, l2), l3)
        
        # La spaziatura tipica del mesh è stimata tramite la mediana del lato più lungo dei triangoli
        typical_spacing = np.median(max_edge)
        threshold = 2.0 * typical_spacing
        
        # Applichiamo la maschera per escludere i triangoli spuri esterni
        mask = max_edge > threshold
        if mask.sum() > 0:
            triang.set_mask(mask)
            print(f"[TRIANGULATION] Applicato mask semplificato su {mask.sum()} triangoli spuri (soglia: {threshold:.6f})")
            
            # Warning per mesh fortemente non uniformi
            masked_ratio = mask.sum() / len(triangles)
            if masked_ratio > 0.15:
                print(f"  [WARNING] La percentuale di triangoli mascherati è elevata ({masked_ratio:.1%}). "
                      f"Se noti dei 'buchi bianchi' indesiderati nelle zone centrali o dove la mesh è più rada, "
                      f"considera di rilassare la soglia di masking incrementando il fattore di threshold.")
    except Exception as e:
        print(f"[WARNING] Errore nell'applicazione del mask alla triangolazione: {e}")
    
    validation_grid = (xy_grid_flat, u_exact, triang)
    stress_exact_grids = {
        'p': p_exact,
        'tau_xx': tau_xx_exact,
        'tau_xy': tau_xy_exact,
        'tau_yy': tau_yy_exact
    }
    
    # Varianze per normalizzazione
    sigma2_u   = max(u_exact.var().item(), variance_eps)
    sigma2_v   = max(v_exact.var().item(), variance_eps)
    sigma2_p   = max(p_exact.var().item(), variance_eps)
    
    # Calcolo varianze reali per lo stress
    sigma2_txx = max(tau_xx_exact.var().item(), variance_eps)
    sigma2_txy = max(tau_xy_exact.var().item(), variance_eps)
    sigma2_tyy = max(tau_yy_exact.var().item(), variance_eps)
    
    var_weights = {
        'u': 1.0, 'v': 1.0, 'p': 1.0,
        'tau_xx': 1.0,
        'tau_xy': 1.0,
        'tau_yy': 1.0,
    }
    
    # Boundary Conditions: estraiamo solo i gruppi geometrici
    boundary_groups = extract_boundary_groups_from_comsol(dataset, device=device)
    
    # Cast dei tensori dei gruppi al tipo richiesto
    for name, group in boundary_groups.items():
        group['xy'] = group['xy'].to(initial_dtype)
        group['norm'] = group['norm'].to(initial_dtype)
        for f_name, f_t in group['fields'].items():
            group['fields'][f_name] = f_t.to(initial_dtype)
    
    # Data Subset
    torch.manual_seed(42)
    idx = torch.randperm(xy_grid_flat.shape[0])[:num_data_subset]
    xy_pinn_data = xy_grid_flat[idx]
    psip_pinn_data = torch.cat([u_exact[idx], v_exact[idx], p_exact[idx], tau_xx_exact[idx], tau_xy_exact[idx], tau_yy_exact[idx]], dim=1)
    uv_pinn_data = torch.cat([u_exact[idx], v_exact[idx]], dim=1)
    
    # GPU Pre-cast
    xy_pinn_data = xy_pinn_data.to(initial_dtype)
    psip_pinn_data = psip_pinn_data.to(initial_dtype)
    uv_pinn_data = uv_pinn_data.to(initial_dtype)
    
    return {
        'dataset': dataset,
        'xy_grid_flat': xy_grid_flat,
        'triang': triang,
        'validation_grid': validation_grid,
        'stress_exact_grids': stress_exact_grids,
        'var_weights': var_weights,
        'data_subsets': {
            'xy': xy_pinn_data,
            'psip': psip_pinn_data,
            'uv': uv_pinn_data
        },
        'boundary_groups': boundary_groups
    }
