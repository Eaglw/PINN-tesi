#!/usr/bin/env python3
"""
===============================================================================
Kaggle Standalone Script: Inverse Problem from COMSOL MLS Derivatives
===============================================================================
Solves the inverse viscoelastic problem directly from COMSOL spatial derivatives
obtained via Moving Least Squares (MLS) with local [-1, 1] scaling.

Key Architectural & Physical Features:
- No kinematic or stress neural networks: only PressureModel(2 -> [128]*8 -> 1, SiLU).
- [Proposta AB] Algebraic hard pressure anchoring:
    p(x) = p_scale * (hat_p(x) - hat_p(x0)) + p_ref
  guaranteeing p(x0) == p_ref exactly at every evaluation without soft boundary penalties.
- [Proposta AC] Trainable physical parameter mu_s protected by softplus (mu_s > 0).
- [Proposta AA] Dimensionless momentum scaling by scale_mom = 400.0 Pa/m.
- [Proposta B] Differentiated Adam epsilon: 1e-8 for network, 1e-15 for physics.
- [Proposta H] Rigid gradient clipping (GRAD_CLIP_NORM = 5.0).
- [Proposta M] Strict FP64 conversion before L-BFGS.
- Run 23 L-BFGS configuration: history_size = 300, line_search_fn = "strong_wolfe".
- Full-budget (20k Adam + 2k L-BFGS) and fast --smoke-test (2 Adam + 2 L-BFGS).
===============================================================================
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import spatial

# Set standard math precision and disable TF32
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# 1. NEURAL NETWORK ARCHITECTURE WITH HARD PRESSURE ANCHORING
# ============================================================================
class FCN(nn.Module):
    """Fully Connected MLP with SiLU activations."""
    def __init__(self, n_input=2, n_output=1, hidden_layers=None, activation=nn.SiLU):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128] * 8
        layers_sizes = [n_input] + hidden_layers + [n_output]
        layers = []
        for i in range(len(layers_sizes) - 1):
            layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
            if i < len(layers_sizes) - 2:
                layers.append(activation())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PressureModel(nn.Module):
    """
    Pressure Neural Network with Algebraic Hard Dirichlet Anchoring (Proposal AB).
    Enforces p(x_0) = p_ref algebraically:
      p(x) = p_scale * (hat_p(x) - hat_p(x_0)) + p_ref
    """
    def __init__(self, p_scale=1.0, x_anchor=None, p_ref=0.0, hidden_layers=None):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128] * 8
        self.model_p = FCN(2, 1, hidden_layers, nn.SiLU)
        self.p_scale = p_scale

        if x_anchor is not None:
            if not isinstance(x_anchor, torch.Tensor):
                x_anc = torch.tensor(x_anchor, dtype=torch.get_default_dtype())
            else:
                x_anc = x_anchor.clone().detach().to(dtype=torch.get_default_dtype())
            self.register_buffer("x_anchor", x_anc.view(1, 2))

            if not isinstance(p_ref, torch.Tensor):
                p_r = torch.tensor([[float(p_ref)]], dtype=torch.get_default_dtype())
            else:
                p_r = p_ref.clone().detach().to(dtype=torch.get_default_dtype()).view(1, 1)
            self.register_buffer("p_ref", p_r)
            self.hard_anchor = True
        else:
            self.register_buffer("x_anchor", None)
            self.register_buffer("p_ref", None)
            self.hard_anchor = False

    def forward(self, x):
        p_raw = self.model_p(x)
        if getattr(self, "hard_anchor", False) and self.x_anchor is not None:
            p_anchor = self.model_p(self.x_anchor)
            return self.p_scale * (p_raw - p_anchor) + self.p_ref
        return self.p_scale * p_raw


def init_weights_xavier(m):
    """Xavier Normal Initialization for Linear Layers."""
    if isinstance(m, nn.Linear):
        gain = nn.init.calculate_gain("relu")  # Gain proxy for SiLU
        nn.init.xavier_normal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ============================================================================
# 2. PHYSICAL PARAMETER MODULE: INVERSE SOLVENT VISCOSITY
# ============================================================================
class InversePhysicsMLS(nn.Module):
    """
    Handles physical parameter mu_s identification with softplus protection (Proposal AC).
    Guarantees mu_s > 0 strictly throughout training.
    """
    def __init__(self, guess_mu_s=0.08, scale_mom=1.0, rho=1000.0, U_ref=8.33319e-3, H_ref=0.005, H_coord=0.05, eta_0=1.0):
        super().__init__()
        self.register_buffer("guess_mu_s", torch.tensor(float(guess_mu_s), dtype=torch.get_default_dtype()))
        self.register_buffer("scale_mom", torch.tensor(float(scale_mom), dtype=torch.get_default_dtype()))
        
        # Characteristic scales
        self.U_ref = U_ref
        self.H_ref = H_ref
        self.H_coord = H_coord
        self.eta_0 = eta_0
        self.rho_phys = rho
        # Geometric ratio s = H_ref / H_coord for coordinate scaling on [0, 1] domain
        s_val = H_ref / H_coord
        self.register_buffer("s_geom", torch.tensor(float(s_val), dtype=torch.get_default_dtype()))
        # Dimensionless convective factor Re_scale = rho * U_ref * H_ref / eta_0
        re_val = (rho * U_ref * H_ref) / eta_0
        self.register_buffer("Re_scale", torch.tensor(float(re_val), dtype=torch.get_default_dtype()))

        # Invert softplus so that when _raw_mu_s == 0, mu_s == guess_mu_s exactly
        # y = softplus(x, beta) => x = log(expm1(beta * y)) / beta
        beta = 20.0
        val_clamped = max(float(guess_mu_s), 1e-6)
        inv_sp = math.log(math.expm1(beta * val_clamped)) / beta
        self.register_buffer("init_mu_s_pre_sp", torch.tensor(float(inv_sp), dtype=torch.get_default_dtype()))

        # Raw log-space parameter initialized at 0.0 -> mu_s = guess_mu_s
        self.register_parameter("_raw_mu_s", nn.Parameter(torch.zeros(1, dtype=torch.get_default_dtype())))

    @property
    def mu_s(self):
        """Solvent viscosity with strict softplus safeguard."""
        raw_val = self.init_mu_s_pre_sp * torch.exp(self._raw_mu_s).squeeze()
        return F.softplus(raw_val, beta=20.0)

    @property
    def mu_s_nd(self):
        """Dimensionless solvent viscosity mu_s* = mu_s / eta_0."""
        return self.mu_s / self.eta_0


# ============================================================================
# 3. MOVING LEAST SQUARES (MLS) DERIVATIVE COMPUTATION & CACHING
# ============================================================================
def compute_mls_derivatives_scaled(coords_np, u_np, v_np, txx_np, txy_np, tyy_np, K=25):
    """
    Computes spatial derivatives from discrete 2D mesh using Moving Least Squares (MLS)
    with local coordinate scaling dx/h in [-1, 1] and 2nd-degree polynomial basis (6 terms).
    """
    print(f"\n[MLS] Computing 2nd-degree MLS derivatives on {coords_np.shape[0]} nodes (K={K}, local scaling [-1, 1])...")
    t0 = time.time()
    N = coords_np.shape[0]
    tree = spatial.cKDTree(coords_np)
    distances, indices = tree.query(coords_np, k=K, workers=-1)

    u_x = np.zeros((N, 1), dtype=np.float32)
    u_y = np.zeros((N, 1), dtype=np.float32)
    u_xx = np.zeros((N, 1), dtype=np.float32)
    u_yy = np.zeros((N, 1), dtype=np.float32)

    v_x = np.zeros((N, 1), dtype=np.float32)
    v_y = np.zeros((N, 1), dtype=np.float32)
    v_xx = np.zeros((N, 1), dtype=np.float32)
    v_yy = np.zeros((N, 1), dtype=np.float32)

    txx_x = np.zeros((N, 1), dtype=np.float32)
    txy_y = np.zeros((N, 1), dtype=np.float32)
    txy_x = np.zeros((N, 1), dtype=np.float32)
    tyy_y = np.zeros((N, 1), dtype=np.float32)

    for i in range(N):
        x0 = coords_np[i]
        idx = indices[i]
        dist = distances[i]
        h = max(dist[-1], 1e-4)

        dxy = coords_np[idx] - x0
        dx_scaled = dxy[:, 0] / h
        dy_scaled = dxy[:, 1] / h

        # Quadratic polynomial basis with dimensionless local coordinates in [-1, 1]
        X = np.column_stack([
            np.ones(K),
            dx_scaled,
            dy_scaled,
            0.5 * dx_scaled**2,
            0.5 * dy_scaled**2,
            dx_scaled * dy_scaled
        ])

        w = np.exp(-(dist**2) / (h**2))
        W = np.diag(w)

        XTW = X.T @ W
        XTWX = XTW @ X + np.eye(6) * 1e-12

        try:
            inv_XTWX = np.linalg.inv(XTWX)
            c_u = inv_XTWX @ XTW @ u_np[idx]
            c_v = inv_XTWX @ XTW @ v_np[idx]
            c_txx = inv_XTWX @ XTW @ txx_np[idx]
            c_txy = inv_XTWX @ XTW @ txy_np[idx]
            c_tyy = inv_XTWX @ XTW @ tyy_np[idx]

            u_x[i] = c_u[1] / h
            u_y[i] = c_u[2] / h
            u_xx[i] = c_u[3] / (h**2)
            u_yy[i] = c_u[4] / (h**2)

            v_x[i] = c_v[1] / h
            v_y[i] = c_v[2] / h
            v_xx[i] = c_v[3] / (h**2)
            v_yy[i] = c_v[4] / (h**2)

            txx_x[i] = c_txx[1] / h
            txy_y[i] = c_txy[2] / h
            txy_x[i] = c_txy[1] / h
            tyy_y[i] = c_tyy[2] / h
        except np.linalg.LinAlgError:
            pass

    conv_u = u_np * u_x + v_np * u_y
    conv_v = u_np * v_x + v_np * v_y
    lap_u = u_xx + u_yy
    lap_v = v_xx + v_yy
    div_tau_x = txx_x + txy_y
    div_tau_y = txy_x + tyy_y

    print(f"[MLS] Derivative computation completed in {time.time() - t0:.2f} s")
    return {
        "conv_u": torch.from_numpy(conv_u).float(),
        "conv_v": torch.from_numpy(conv_v).float(),
        "lap_u": torch.from_numpy(lap_u).float(),
        "lap_v": torch.from_numpy(lap_v).float(),
        "div_tau_x": torch.from_numpy(div_tau_x).float(),
        "div_tau_y": torch.from_numpy(div_tau_y).float(),
    }


def load_dataset_and_derivatives(dataset_path, cache_path, device):
    """
    Loads COMSOL data and either loads or computes MLS derivatives.
    """
    print("=" * 70)
    print(f"[Data] Loading dataset from: {dataset_path}")
    raw_data = np.loadtxt(str(dataset_path), dtype=np.float64, delimiter=",", comments="%")
    assert raw_data.shape[1] >= 8, f"Expected at least 8 columns, found {raw_data.shape[1]}"

    x_raw, y_raw = raw_data[:, 0], raw_data[:, 1]
    u_raw, v_raw = raw_data[:, 2], raw_data[:, 3]
    p_raw = raw_data[:, 4]
    txx_raw, txy_raw, tyy_raw = raw_data[:, 5], raw_data[:, 6], raw_data[:, 7]

    # Coordinate scaling to [0, 1]
    x_min, x_max = x_raw.min(), x_raw.max()
    y_min, y_max = y_raw.min(), y_raw.max()
    H_coord = max(y_max - y_min, 1e-9)
    H_ref = 0.005  # Roll radius [m]
    U_ref = max(float(np.max(np.sqrt(u_raw**2 + v_raw**2))), 1e-9)
    eta_0 = 1.0
    p_ref_scale = eta_0 * U_ref / H_ref
    tau_ref_scale = eta_0 * U_ref / H_ref

    x_nd = (x_raw - x_min) / H_coord
    y_nd = (y_raw - y_min) / H_coord
    coords_np = np.column_stack([x_nd, y_nd]).astype(np.float32)

    u_nd = (u_raw / U_ref).reshape(-1, 1).astype(np.float32)
    v_nd = (v_raw / U_ref).reshape(-1, 1).astype(np.float32)
    p_nd = (p_raw / p_ref_scale).reshape(-1, 1).astype(np.float32)
    txx_nd = (txx_raw / tau_ref_scale).reshape(-1, 1).astype(np.float32)
    txy_nd = (txy_raw / tau_ref_scale).reshape(-1, 1).astype(np.float32)
    tyy_nd = (tyy_raw / tau_ref_scale).reshape(-1, 1).astype(np.float32)

    p_scale = max(float(np.abs(p_nd).max()), 1e-6)

    # Extract Dirichlet Anchor point (wall boundary node at x_nd ~ 1.0)
    wall_mask = (x_nd >= 0.999)
    if np.any(wall_mask):
        anchor_idx = np.where(wall_mask)[0][0]
    else:
        anchor_idx = 0
    x_anchor = torch.tensor(coords_np[anchor_idx : anchor_idx + 1], dtype=torch.float32, device=device)
    p_ref_val = float(p_nd[anchor_idx, 0])
    print(f"  [Anchor] Selected node {anchor_idx}: x_anchor = {x_anchor.cpu().numpy().tolist()}, p_ref = {p_ref_val:.4f}")

    # Load or compute MLS cache
    derivatives = None
    if cache_path is not None and Path(cache_path).is_file():
        print(f"[Cache] Found precomputed MLS derivatives at: {cache_path}")
        try:
            cache = torch.load(str(cache_path), map_location="cpu")
            required_keys = ["conv_u", "conv_v", "lap_u", "lap_v", "div_tau_x", "div_tau_y"]
            if all(k in cache for k in required_keys):
                print("  MLS derivatives cache verified and loaded successfully!")
                derivatives = {k: cache[k].to(device=device, dtype=torch.float32) for k in required_keys}
        except Exception as e:
            print(f"  [Warning] Failed loading cache ({e}), computing from scratch...")

    if derivatives is None:
        derivatives_cpu = compute_mls_derivatives_scaled(coords_np, u_nd, v_nd, txx_nd, txy_nd, tyy_nd, K=25)
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(derivatives_cpu, str(cache_path))
            print(f"  [Cache] Saved newly computed MLS derivatives to: {cache_path}")
        derivatives = {k: v.to(device=device, dtype=torch.float32) for k, v in derivatives_cpu.items()}

    coords_tensor = torch.from_numpy(coords_np).to(device=device, dtype=torch.float32)
    p_exact_tensor = torch.from_numpy(p_nd).to(device=device, dtype=torch.float32)

    return {
        "coords": coords_tensor,
        "p_exact": p_exact_tensor,
        "p_scale": p_scale,
        "x_anchor": x_anchor,
        "p_ref": p_ref_val,
        "derivatives": derivatives,
        "U_ref": U_ref,
        "H_ref": H_ref,
        "H_coord": H_coord,
        "eta_0": eta_0,
        "p_ref_scale": p_ref_scale,
    }


def compute_pressure_l2_error(model, coords, p_exact, chunk_size=8192):
    """Computes relative L2 error for pressure against ground truth."""
    model.eval()
    p_pred_list = []
    with torch.no_grad():
        for i in range(0, coords.shape[0], chunk_size):
            xc = coords[i : i + chunk_size]
            p_pred_list.append(model(xc))
    p_pred = torch.cat(p_pred_list, dim=0)

    diff = p_pred - p_exact
    norm_diff = torch.norm(diff, 2)
    norm_exact = torch.norm(p_exact, 2)
    if norm_exact > 1e-12:
        return (norm_diff / norm_exact).item()
    return 0.0


# ============================================================================
# 4. TRAINING PIPELINE (ADAM FP32 -> L-BFGS FP64)
# ============================================================================
def train_inverse_mls(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data and MLS Derivatives
    data = load_dataset_and_derivatives(args.dataset, args.cache_path, DEVICE)
    coords = data["coords"]
    p_exact = data["p_exact"]
    p_scale = data["p_scale"]
    x_anchor = data["x_anchor"]
    p_ref = data["p_ref"]
    derivs = data["derivatives"]

    conv_u = derivs["conv_u"]
    conv_v = derivs["conv_v"]
    lap_u = derivs["lap_u"]
    lap_v = derivs["lap_v"]
    div_tau_x = derivs["div_tau_x"]
    div_tau_y = derivs["div_tau_y"]

    N_points = coords.shape[0]

    # 2. Instantiate Model and Physical Parameter
    model = PressureModel(p_scale=p_scale, x_anchor=x_anchor, p_ref=p_ref).to(DEVICE)
    model.apply(init_weights_xavier)

    physics = InversePhysicsMLS(
        guess_mu_s=args.guess_mu_s,
        scale_mom=args.scale_mom,
        rho=args.rho,
        U_ref=data["U_ref"],
        H_ref=data["H_ref"],
        H_coord=data["H_coord"],
        eta_0=data["eta_0"]
    ).to(DEVICE)

    # Verify hard anchor algebraically
    with torch.no_grad():
        p_at_anchor = model(x_anchor).item()
        assert abs(p_at_anchor - p_ref) < 1e-5, f"Hard anchor verification failed: {p_at_anchor} != {p_ref}"
        print(f"[Algebraic Hard Anchor Verified] p(x_anchor) = {p_at_anchor:.6f} == p_ref = {p_ref:.6f}")

    # 3. Setup Differentiated Adam Optimizer (Proposal B)
    param_groups = [
        {"params": model.parameters(), "lr": args.lr_adam, "eps": 1e-8},
        {"params": [physics._raw_mu_s], "lr": args.lr_adam, "eps": 1e-15},
    ]
    optimizer_adam = torch.optim.Adam(param_groups)
    scheduler_adam = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_adam, T_max=args.epochs_adam, eta_min=1e-5
    )

    history = {
        "epoch": [],
        "loss": [],
        "mu_s": [],
        "l2_p": [],
    }

    print("\n" + "=" * 70)
    print(f"PHASE 1: ADAM OPTIMIZATION (FP32) — {args.epochs_adam} Epochs")
    print(f"  scale_mom = {args.scale_mom:.1f} Pa/m | Guess mu_s = {physics.mu_s.item():.4f} Pa·s (True = 0.1000)")
    print("=" * 70)

    start_time = time.time()
    chunk_size = args.chunk_size

    for epoch in range(args.epochs_adam):
        model.train()
        optimizer_adam.zero_grad(set_to_none=True)

        loss_accum = 0.0
        # Mini-batch / chunk evaluation over collocation points
        for i in range(0, N_points, chunk_size):
            xc = coords[i : i + chunk_size]
            w_chunk = xc.shape[0] / N_points

            xc_ph = xc.clone().requires_grad_(True)
            p_pred = model(xc_ph)

            grad_p = torch.autograd.grad(p_pred.sum(), xc_ph, create_graph=True, retain_graph=True)[0]
            px = grad_p[:, 0:1]
            py = grad_p[:, 1:2]

            cu = conv_u[i : i + chunk_size]
            cv = conv_v[i : i + chunk_size]
            lu = lap_u[i : i + chunk_size]
            lv = lap_v[i : i + chunk_size]
            dtx = div_tau_x[i : i + chunk_size]
            dty = div_tau_y[i : i + chunk_size]

            # Dimensionless Navier-Stokes Momentum Residuals
            re_eff = physics.Re_scale
            mu_s_nd = physics.mu_s_nd
            s_geom = physics.s_geom
            scale_m = physics.scale_mom

            fu = (re_eff * cu + px - mu_s_nd * s_geom * lu - dtx) / scale_m
            fv = (re_eff * cv + py - mu_s_nd * s_geom * lv - dty) / scale_m

            loss_chunk = 0.5 * ((fu**2 + fv**2).mean()) * w_chunk
            loss_chunk.backward()
            loss_accum += loss_chunk.item()

        # Rigid Gradient Clipping for Adam (Proposal H)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        torch.nn.utils.clip_grad_norm_([physics._raw_mu_s], args.grad_clip)

        optimizer_adam.step()
        scheduler_adam.step()

        # Logging & Diagnostics
        log_step = (
            (epoch + 1) % max(1, args.epochs_adam // 20) == 0
            or epoch == 0
            or (epoch + 1) == args.epochs_adam
        )
        if log_step:
            l2_p_val = compute_pressure_l2_error(model, coords, p_exact, chunk_size=8192)
            mu_s_val = physics.mu_s.item()
            history["epoch"].append(epoch + 1)
            history["loss"].append(loss_accum)
            history["mu_s"].append(mu_s_val)
            history["l2_p"].append(l2_p_val)

            elapsed = time.time() - start_time
            print(f"Adam Epoch {epoch+1:5d}/{args.epochs_adam} | Loss: {loss_accum:.4e} | mu_s: {mu_s_val:.4f} Pa·s | L2(p): {l2_p_val*100:.2f}% | Elapsed: {elapsed:.1f}s")

    # 4. Phase 2: L-BFGS FP64 Optimization
    if args.iters_lbfgs > 0:
        print("\n" + "=" * 70)
        print(f"PHASE 2: L-BFGS REFINEMENT (FP64) — {args.iters_lbfgs} Iterations (History 300, Strong Wolfe)")
        print("=" * 70)

        # Convert model, physics, and tensors strictly to FP64 (Proposal M)
        torch.set_default_dtype(torch.float64)
        model.double()
        physics.double()
        coords_64 = coords.double()
        p_exact_64 = p_exact.double()
        conv_u_64 = conv_u.double()
        conv_v_64 = conv_v.double()
        lap_u_64 = lap_u.double()
        lap_v_64 = lap_v.double()
        div_tau_x_64 = div_tau_x.double()
        div_tau_y_64 = div_tau_y.double()

        optimizer_lbfgs = torch.optim.LBFGS(
            list(model.parameters()) + [physics._raw_mu_s],
            lr=1.0,
            max_iter=args.iters_lbfgs,
            tolerance_grad=1e-12,
            tolerance_change=1e-16,
            history_size=300,
            line_search_fn="strong_wolfe",
        )

        iter_count = [0]
        lbfgs_start_time = time.time()
        chunk_size_lbfgs = 16384

        def closure():
            optimizer_lbfgs.zero_grad()
            re_eff = physics.Re_scale
            s_geom = physics.s_geom
            scale_m = physics.scale_mom

            total_loss = 0.0
            # Chunked evaluation over all collocation points to bound FP64 VRAM
            for i in range(0, N_points, chunk_size_lbfgs):
                xc = coords_64[i : i + chunk_size_lbfgs]
                w_chunk = xc.shape[0] / N_points

                xc_ph = xc.clone().requires_grad_(True)
                p_pred = model(xc_ph)

                grad_p = torch.autograd.grad(p_pred.sum(), xc_ph, create_graph=True, retain_graph=True)[0]
                px = grad_p[:, 0:1]
                py = grad_p[:, 1:2]

                cu = conv_u_64[i : i + chunk_size_lbfgs]
                cv = conv_v_64[i : i + chunk_size_lbfgs]
                lu = lap_u_64[i : i + chunk_size_lbfgs]
                lv = lap_v_64[i : i + chunk_size_lbfgs]
                dtx = div_tau_x_64[i : i + chunk_size_lbfgs]
                dty = div_tau_y_64[i : i + chunk_size_lbfgs]

                # Compute mu_s_nd fresh inside chunk loop to avoid graph reuse across multiple backward() calls
                mu_s_nd = physics.mu_s_nd

                fu = (re_eff * cu + px - mu_s_nd * s_geom * lu - dtx) / scale_m
                fv = (re_eff * cv + py - mu_s_nd * s_geom * lv - dty) / scale_m

                chunk_loss = 0.5 * ((fu**2 + fv**2).mean()) * w_chunk
                chunk_loss.backward()
                total_loss += chunk_loss.item()

            # Note: Do NOT apply clip_grad_norm_ inside closure!
            # Modifying gradients inside Strong Wolfe line search breaks curvature condition and aborts L-BFGS.

            iter_count[0] += 1
            curr_it = iter_count[0]
            log_step = (
                curr_it % 50 == 0
                or curr_it == 1
                or curr_it == args.iters_lbfgs
            )
            if log_step:
                l2_p_val = compute_pressure_l2_error(model, coords_64, p_exact_64, chunk_size=8192)
                mu_s_val = physics.mu_s.item()
                history["epoch"].append(args.epochs_adam + curr_it)
                history["loss"].append(total_loss)
                history["mu_s"].append(mu_s_val)
                history["l2_p"].append(l2_p_val)
                print(f"L-BFGS Iter {curr_it:4d}/{args.iters_lbfgs} | Loss: {total_loss:.4e} | mu_s: {mu_s_val:.5f} Pa·s | L2(p): {l2_p_val*100:.2f}% | Elapsed: {time.time() - lbfgs_start_time:.1f}s")

            return torch.tensor(total_loss, dtype=torch.float64, device=coords_64.device)

        optimizer_lbfgs.step(closure)

    total_time = time.time() - start_time
    final_l2_p = compute_pressure_l2_error(model, coords_64 if args.iters_lbfgs > 0 else coords, p_exact_64 if args.iters_lbfgs > 0 else p_exact)
    final_mu_s = physics.mu_s.item()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"  Total Wall Time: {total_time:.2f} s ({total_time / 60.0:.2f} min)")
    print(f"  Final Identified mu_s: {final_mu_s:.5f} Pa·s (True: 0.10000 Pa·s, Rel Error: {abs(final_mu_s - 0.1) / 0.1 * 100:.2f}%)")
    print(f"  Final Relative L2(p):  {final_l2_p * 100:.2f}%")
    print("=" * 70)

    # Save Checkpoint
    ckpt_path = output_dir / "checkpoint_inverse_mls.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "physics_state_dict": physics.state_dict(),
        "history": history,
        "final_mu_s": final_mu_s,
        "final_l2_p": final_l2_p,
        "total_time": total_time,
    }, ckpt_path)
    print(f"[Save] Model checkpoint saved to: {ckpt_path}")

    # Plot History
    if len(history["epoch"]) > 0:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(history["epoch"], history["loss"], "b-")
        plt.yscale("log")
        plt.title("Momentum Loss")
        plt.xlabel("Step")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.plot(history["epoch"], history["mu_s"], "g-", label="Learned mu_s")
        plt.axhline(0.1, color="r", linestyle="--", label="True mu_s (0.1)")
        plt.title(r"Solvent Viscosity $\mu_s$ [Pa·s]")
        plt.xlabel("Step")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        plt.plot(history["epoch"], [v * 100 for v in history["l2_p"]], "m-")
        plt.title("Relative L2(p) [%]")
        plt.xlabel("Step")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "training_history.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[Plot] Training history saved to: {plot_path}")

    return {
        "final_mu_s": final_mu_s,
        "final_l2_p": final_l2_p,
        "total_time": total_time,
    }


# ============================================================================
# 5. CLI ARGUMENT PARSER
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Kaggle Standalone Inverse Viscoelastic Problem from COMSOL MLS")
    
    # Paths with intelligent repository defaults
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    default_dataset = repo_root / "COMSOL" / "4roll" / "4_roll_mill.csv"
    default_cache = repo_root / "COMSOL" / "4roll" / "comsol_derivatives_mls.pt"
    default_output = script_dir / "output_kaggle_inverse_mls"

    parser.add_argument("--dataset", type=str, default=str(default_dataset), help="Path to COMSOL CSV dataset")
    parser.add_argument("--cache-path", type=str, default=str(default_cache), help="Path to MLS derivatives .pt cache")
    parser.add_argument("--output-dir", type=str, default=str(default_output), help="Output directory")

    # Training settings
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test (2 Adam + 2 L-BFGS)")
    parser.add_argument("--epochs-adam", type=int, default=20000, help="Number of Adam epochs")
    parser.add_argument("--iters-lbfgs", type=int, default=2000, help="Number of L-BFGS iterations")
    parser.add_argument("--lr-adam", type=float, default=1e-3, help="Adam base learning rate")
    parser.add_argument("--chunk-size", type=int, default=8192, help="Chunk size for Adam mini-batches")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Rigid gradient clipping norm")

    # Physical parameters
    parser.add_argument("--scale-mom", type=float, default=1.0, help="Momentum loss scaling (1.0 for dimensionless residual)")
    parser.add_argument("--guess-mu-s", type=float, default=0.08, help="Initial guess for mu_s [Pa·s]")
    parser.add_argument("--rho", type=float, default=1000.0, help="Fluid density [kg/m^3]")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")

    args = parser.parse_args()
    if args.smoke_test:
        args.epochs_adam = 2
        args.iters_lbfgs = 2
    return args


if __name__ == "__main__":
    args = parse_args()
    train_inverse_mls(args)
