import torch
import numpy as np
import os


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
    H = float(y_raw.max())
    L = float(x_raw.max())
    U_ref = float(u_raw.max())

    mu_s = params['mu_s']
    mu_p = params['mu_p']
    lam = params.get('lam', 1.0)
    eps = params.get('eps', 0.0)
    alpha = params.get('alpha', 0.0)
    rho = params.get('rho', 1.0)
    mu_tot = mu_s + mu_p

    p_ref = mu_tot * U_ref / H      # Scala viscosa di pressione
    tau_ref = mu_tot * U_ref / H    # Stessa scala per gli sforzi

    # --- 3. Adimensionalizzazione ---
    x_nd = x_raw / H
    y_nd = y_raw / H
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

    # --- 4. Identificazione nodi boundary ---
    tol = 1e-8
    L_nd = L / H  # Lunghezza adimensionale del canale
    H_nd = 1.0    # Altezza adimensionale (H/H = 1)

    # Indici candidati per inlet e outlet (prioritari sugli spigoli)
    inlet_mask = x_raw < tol
    outlet_mask = np.abs(x_raw - L) < tol

    # Bottom e top: escludono inlet e outlet per evitare duplicati
    bottom_mask = (y_raw < tol) & (~inlet_mask) & (~outlet_mask)
    top_mask = (np.abs(y_raw - H) < tol) & (~inlet_mask) & (~outlet_mask)

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


def generate_boundaries_from_comsol(dataset, device='cpu'):
    """
    Genera le condizioni al contorno (Dirichlet + Neumann) dal dataset COMSOL
    nel formato compatibile con ViscoelasticPhysics.boundary_loss().

    Restituisce la stessa tupla di generate_boundaries():
        (xy_boundary, dirichlet_boundary, neumann_boundary, normals_boundary)

    Logica BC per ogni bordo:
        - Inlet  (x*≈0):   Dirichlet su tutti i campi (da CSV). Neumann = NaN. n=[-1,0].
        - Bottom (y*≈0):   Dirichlet: u*=0, v*=0, p=NaN, tau=NaN.
                           Neumann: dp/dn=0, resto NaN. n=[0,-1].
        - Top    (y*≈H*):  Come bottom, ma n=[0,+1].
        - Outlet (x*≈L*):  Dirichlet: p* da CSV, resto NaN.
                           Neumann: tau_xx=0, tau_xy=0, tau_yy=0 (stress-free), resto NaN. n=[1,0].

    Args:
        dataset: Dizionario restituito da load_comsol_csv().
        device: Device torch su cui creare i tensori.

    Returns:
        tuple: (xy_boundary, dirichlet_boundary, neumann_boundary, normals_boundary)
            - xy_boundary: (N_bc, 2) coordinate adimensionali dei nodi boundary
            - dirichlet_boundary: (N_bc, 6) valori target [u, v, p, tau_xx, tau_xy, tau_yy]
            - neumann_boundary: (N_bc, 6) gradienti normali target
            - normals_boundary: (N_bc, 2) vettori normali uscenti
    """
    coords = dataset['coords']
    bidx = dataset['boundary_indices']

    # --- Helper interni ---
    def pack_state(u, v, p, txx, txy, tyy):
        """Assembla un tensore (N, 6) con colonne [u, v, p, tau_xx, tau_xy, tau_yy]."""
        return torch.cat([u, v, p, txx, txy, tyy], dim=1)

    def get_nan(n_pts, device):
        """Crea un tensore (n_pts, 1) di NaN."""
        return torch.full((n_pts, 1), float('nan'), device=device)

    def get_zero(n_pts, device):
        """Crea un tensore (n_pts, 1) di zeri."""
        return torch.zeros((n_pts, 1), device=device)

    # ===================================================================
    # 1. INLET (x* ≈ 0) → Dirichlet completo da CSV, Neumann = NaN
    # ===================================================================
    idx_in = bidx['inlet']
    n_in = len(idx_in)
    xy_in = coords[idx_in].to(device)
    n_inlet = torch.tensor([[-1.0, 0.0]], device=device).expand(n_in, 2)

    # Dirichlet: tutti i campi dal CSV
    inlet_dir = pack_state(
        dataset['u'][idx_in].to(device),
        dataset['v'][idx_in].to(device),
        dataset['p'][idx_in].to(device),
        dataset['tau_xx'][idx_in].to(device),
        dataset['tau_xy'][idx_in].to(device),
        dataset['tau_yy'][idx_in].to(device),
    )
    # Neumann: nessun vincolo
    inlet_neu = pack_state(
        get_nan(n_in, device), get_nan(n_in, device), get_nan(n_in, device),
        get_nan(n_in, device), get_nan(n_in, device), get_nan(n_in, device),
    )

    # ===================================================================
    # 2. BOTTOM WALL (y* ≈ 0) → No-slip + dp/dn = 0
    # ===================================================================
    idx_bot = bidx['bottom']
    n_bot = len(idx_bot)
    xy_bot = coords[idx_bot].to(device)
    n_bottom = torch.tensor([[0.0, -1.0]], device=device).expand(n_bot, 2)

    # Dirichlet: u=0, v=0, pressione e stress liberi
    bot_dir = pack_state(
        get_zero(n_bot, device), get_zero(n_bot, device),
        get_nan(n_bot, device),
        get_nan(n_bot, device), get_nan(n_bot, device), get_nan(n_bot, device),
    )
    # Neumann: dp/dn = 0, resto libero
    bot_neu = pack_state(
        get_nan(n_bot, device), get_nan(n_bot, device),
        get_zero(n_bot, device),
        get_nan(n_bot, device), get_nan(n_bot, device), get_nan(n_bot, device),
    )

    # ===================================================================
    # 3. TOP WALL (y* ≈ H*) → No-slip + dp/dn = 0
    # ===================================================================
    idx_top = bidx['top']
    n_top_pts = len(idx_top)
    xy_top = coords[idx_top].to(device)
    n_top = torch.tensor([[0.0, 1.0]], device=device).expand(n_top_pts, 2)

    # Dirichlet: u=0, v=0, pressione e stress liberi
    top_dir = pack_state(
        get_zero(n_top_pts, device), get_zero(n_top_pts, device),
        get_nan(n_top_pts, device),
        get_nan(n_top_pts, device), get_nan(n_top_pts, device), get_nan(n_top_pts, device),
    )
    # Neumann: dp/dn = 0, resto libero
    top_neu = pack_state(
        get_nan(n_top_pts, device), get_nan(n_top_pts, device),
        get_zero(n_top_pts, device),
        get_nan(n_top_pts, device), get_nan(n_top_pts, device), get_nan(n_top_pts, device),
    )

    # ===================================================================
    # 4. OUTLET (x* ≈ L*) → p da CSV, stress-free Neumann
    # ===================================================================
    idx_out = bidx['outlet']
    n_out = len(idx_out)
    xy_out = coords[idx_out].to(device)
    n_outlet = torch.tensor([[1.0, 0.0]], device=device).expand(n_out, 2)

    # Dirichlet: solo pressione dal CSV
    out_dir = pack_state(
        get_nan(n_out, device), get_nan(n_out, device),
        dataset['p'][idx_out].to(device),
        get_nan(n_out, device), get_nan(n_out, device), get_nan(n_out, device),
    )
    # Neumann: stress-free (tau_xx=0, tau_xy=0, tau_yy=0)
    out_neu = pack_state(
        get_nan(n_out, device), get_nan(n_out, device),
        get_nan(n_out, device),
        get_zero(n_out, device), get_zero(n_out, device), get_zero(n_out, device),
    )

    # ===================================================================
    # 5. CONCATENAZIONE GLOBALE (stesso ordine di generate_boundaries)
    # ===================================================================
    xy_boundary = torch.cat([xy_in, xy_bot, xy_top, xy_out], dim=0)

    dirichlet_boundary = torch.cat([inlet_dir, bot_dir, top_dir, out_dir], dim=0)

    neumann_boundary = torch.cat([inlet_neu, bot_neu, top_neu, out_neu], dim=0)

    normals_boundary = torch.cat([n_inlet, n_bottom, n_top, n_outlet], dim=0)

    print(f"\nBoundary assemblate dal dataset COMSOL:")
    print(f"  Totale nodi BC:  {xy_boundary.shape[0]}")
    print(f"  Inlet:           {n_in}")
    print(f"  Bottom wall:     {n_bot}")
    print(f"  Top wall:        {n_top_pts}")
    print(f"  Outlet:          {n_out}")

    return xy_boundary, dirichlet_boundary, neumann_boundary, normals_boundary
