import torch

def test_random_points(model, physics, data, num_points=10):
    """Confronta le predizioni di tau_xx con la ground truth su punti casuali."""
    print(f"\n{'=' * 60}\nTEST {num_points} PUNTI CASUALI (Stress tau_xx)\n{'=' * 60}")
    model.eval()
    _dtype = next(model.parameters()).dtype

    with torch.set_grad_enabled(True):
        idx_rand = torch.randperm(data["coords"].shape[0])[:num_points]
        xi = data["coords"][idx_rand].to(_dtype).clone()

        # Riapplichiamo requires_grad solo se get_velocity lo esige internamente
        xi.requires_grad_(True)
        _, _, _, tau_p = physics.get_velocity(model, xi, create_graph=False)

        tau_xx_pred = tau_p[:, 0].detach().cpu().numpy()
        tau_xx_true = data["tau_xx"][idx_rand].view(-1).cpu().numpy()

        for i in range(num_points):
            print(
                f"Point {i + 1:2d}: COMSOL = {tau_xx_true[i]:.4f} | PINN = {tau_xx_pred[i]:.4f}"
            )


def debug_physics_magnitudes(model, physics, data, num_points=2000):
    """Calcola e stampa le magnitudo dei termini delle equazioni Costitutiva e Momentum."""
    print(
        f"\n{'=' * 60}\nTEST MAGNITUDO TERMINI PDE (Su {num_points} punti)\n{'=' * 60}"
    )
    model.eval()
    _dtype = next(model.parameters()).dtype

    with torch.set_grad_enabled(True):
        idx_diag = torch.randperm(data["coords"].shape[0])[:num_points]
        x = data["coords"][idx_diag].to(_dtype).clone().requires_grad_(True)

        psi = model.model_psi(x)
        p = model.model_p(x) * model.p_scale
        tau = model.model_tau(x) * model.tau_scale
        tau_xx, tau_xy, tau_yy = tau[:, 0:1], tau[:, 1:2], tau[:, 2:3]

        # --- Derivate Prime (Velocità e Pressione) ---
        grad_psi = physics._grad(psi, x, create_graph=True)
        u, v = grad_psi[:, 1:2], -grad_psi[:, 0:1]

        grad_u = physics._grad(u, x, create_graph=True)
        u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]

        grad_v = physics._grad(v, x, create_graph=True)
        v_x, v_y = (
            grad_v[:, 0:1],
            -u_x,
        )  # Incomprimibilità garantita dalla streamfunction

        grad_p = physics._grad(p, x, create_graph=True)
        p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]

        # --- Derivate Seconde (Viscosità e Stress) ---
        grad_u_x = physics._grad(u_x, x, create_graph=True)
        grad_u_y = physics._grad(u_y, x, create_graph=True)
        u_xx, u_yy = grad_u_x[:, 0:1], grad_u_y[:, 1:2]

        grad_v_x = physics._grad(v_x, x, create_graph=True)
        v_xx, v_yy = grad_v_x[:, 0:1], -grad_u_y[:, 0:1]

        g_txx = physics._grad(tau_xx, x, create_graph=True)
        g_txy = physics._grad(tau_xy, x, create_graph=True)
        g_tyy = physics._grad(tau_yy, x, create_graph=True)

        tau_xx_x, tau_xx_y = g_txx[:, 0:1], g_txx[:, 1:2]
        tau_xy_x, tau_xy_y = g_txy[:, 0:1], g_txy[:, 1:2]
        tau_yy_x, tau_yy_y = g_tyy[:, 0:1], g_tyy[:, 1:2]

        # --- Termini Costitutivi ---
        upper_xx = u * tau_xx_x + v * tau_xx_y - 2 * u_x * tau_xx - 2 * u_y * tau_xy
        upper_yy = u * tau_yy_x + v * tau_yy_y - 2 * v_x * tau_xy - 2 * v_y * tau_yy
        upper_xy = (
            u * tau_xy_x
            + v * tau_xy_y
            - u_x * tau_xy
            - u_y * tau_yy
            - tau_xx * v_x
            - tau_xy * v_y
        )

        Re, Wi, beta, beta_poly, eps, alpha = physics._nondim()

        source_xx = -2.0 * beta_poly * u_x
        source_yy = -2.0 * beta_poly * v_y
        source_xy = -beta_poly * (u_y + v_x)

        # --- Termini Momentum ---
        div_tau_x, div_tau_y = (tau_xx_x + tau_xy_y), (tau_xy_x + tau_yy_y)
        viscous_x, viscous_y = beta * (u_xx + u_yy), beta * (v_xx + v_yy)
        advection_x = Re * (u * u_x + v * u_y)
        advection_y = Re * (u * v_x + v * v_y)

        # --- Stampa Report ---
        print("--- EQUAZIONE COSTITUTIVA ---")
        for name, terms in zip(
            ["xx", "yy", "xy"],
            [
                (tau_xx, upper_xx, source_xx),
                (tau_yy, upper_yy, source_yy),
                (tau_xy, upper_xy, source_xy),
            ],
        ):
            print(f" Componente {name}:")
            print(f"  tau term mean:    {terms[0].abs().mean().item():.6f}")
            print(f"  Wi*upper mean:    {(Wi * terms[1]).abs().mean().item():.6f}")
            print(
                f"  source term mean: {terms[2].abs().mean().item():.6f}\n" + "-" * 30
            )

        print("--- EQUAZIONE MOMENTUM ---")
        print(
            f" Asse X -> div_tau: {div_tau_x.abs().mean().item():.6f} | grad_p: {p_x.abs().mean().item():.6f} | visc: {viscous_x.abs().mean().item():.6f} | adv: {advection_x.abs().mean().item():.6f}"
        )
        print(
            f" Asse Y -> div_tau: {div_tau_y.abs().mean().item():.6f} | grad_p: {p_y.abs().mean().item():.6f} | visc: {viscous_y.abs().mean().item():.6f} | adv: {advection_y.abs().mean().item():.6f}"
        )
