"""
Motore Gaussian Process per Level Set Estimation (Straddle)
e selezione di batch diversificati (Kriging Believer).
Implementato con architettura robusta e fallback autonomo in SciPy/NumPy
per garantire esecuzione immediata senza dipendenze esterne bloccanti.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from .config import (
    FEATURE_NAMES,
    CONVERGENCE_THRESHOLD_LOG,
    CONVERGENCE_THRESHOLD_PCT,
    BETA_EXPLORATION,
    PARAM_BOUNDS,
    MESH_NODES,
    MESH_COST
)


class StandardFeatureScaler:
    """Standard scaler semplice e robusto basato su NumPy."""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Evita divisioni per zero se una colonna è costante
        self.std[self.std < 1e-6] = 1.0
        return (X - self.mean) / self.std

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


class SciPyGPRegressor:
    """
    Gaussian Process Regressor con Kernel Matérn 5/2 ad ARD (Automatic Relevance Determination)
    e fattorizzazione di Cholesky, ottimizzato tramite massima verosimiglianza marginale (L-BFGS-B).
    """
    def __init__(self, n_restarts: int = 5, random_state: int = 42):
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.X_train = None
        self.y_train = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.L_ = None
        self.alpha_ = None
        self.length_scales = None
        self.sigma_f2 = 1.0
        self.sigma_n2 = 1e-3

    def _matern52_kernel(self, X1: np.ndarray, X2: np.ndarray, length_scales: np.ndarray, sigma_f2: float) -> np.ndarray:
        # Distanza pesata per ARD: d = sqrt(sum ((x_i - x'_i)/l_i)^2)
        X1_scaled = X1 / length_scales
        X2_scaled = X2 / length_scales
        d = cdist(X1_scaled, X2_scaled, metric="euclidean")
        sqrt5_d = np.sqrt(5.0) * d
        K = sigma_f2 * (1.0 + sqrt5_d + (5.0 / 3.0) * (d ** 2)) * np.exp(-sqrt5_d)
        return K

    def _nll(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        # theta = [log(sigma_f2), log(sigma_n2), log(l_1), ..., log(l_D)]
        sigma_f2 = np.exp(theta[0])
        sigma_n2 = np.exp(theta[1])
        length_scales = np.exp(theta[2:])

        K = self._matern52_kernel(X, X, length_scales, sigma_f2)
        K[np.diag_indices_from(K)] += sigma_n2 + 1e-6

        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10

        # Risolve L * alpha = y
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        n = len(y)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        nll = 0.5 * np.dot(y, alpha) + 0.5 * log_det + 0.5 * n * np.log(2.0 * np.pi)
        return float(nll)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = np.copy(X)
        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y)) if np.std(y) > 1e-4 else 1.0
        y_norm = (y - self.y_mean) / self.y_std
        self.y_train = y_norm

        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        best_nll = float("inf")
        best_theta = None

        # Bounds: [log(sigma_f2), log(sigma_n2), log(length_scales...)]
        # Regolarizzazione fisica:
        # min length_scale = 0.65 (elimina isole e bolle artificiali isolate attorno ai singoli punti)
        # max length_scale = 2.50 (previene cross-talk spurio tra modelli costitutivi distinti)
        bounds = [(-2.0, 3.0), (-6.0, -1.0)] + [(float(np.log(0.65)), float(np.log(2.5)))] * n_features

        for i in range(self.n_restarts):
            if i == 0:
                init_theta = np.array([0.0, -4.0] + [0.0] * n_features)
            else:
                init_theta = rng.uniform(
                    low=[b[0] for b in bounds],
                    high=[b[1] for b in bounds]
                )

            res = minimize(
                fun=self._nll,
                x0=init_theta,
                args=(X, y_norm),
                method="L-BFGS-B",
                bounds=bounds
            )
            if res.success and res.fun < best_nll:
                best_nll = res.fun
                best_theta = res.x

        if best_theta is None:
            best_theta = np.array([0.0, -4.0] + [0.0] * n_features)

        self.sigma_f2 = float(np.exp(best_theta[0]))
        self.sigma_n2 = float(np.exp(best_theta[1]))
        self.length_scales = np.exp(best_theta[2:])

        # Calcola la decomposizione di Cholesky finale
        K = self._matern52_kernel(X, X, self.length_scales, self.sigma_f2)
        K[np.diag_indices_from(K)] += self.sigma_n2 + 1e-6
        self.L_ = np.linalg.cholesky(K)
        self.alpha_ = np.linalg.solve(self.L_.T, np.linalg.solve(self.L_, y_norm))

    def predict(self, X_star: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        K_star = self._matern52_kernel(self.X_train, X_star, self.length_scales, self.sigma_f2)
        
        # mu = K_*^T * alpha
        mu_norm = np.dot(K_star.T, self.alpha_)
        mu = mu_norm * self.y_std + self.y_mean

        if not return_std:
            return mu, None

        # v = L \ K_*
        v = np.linalg.solve(self.L_, K_star)
        
        # K(X_*, X_*) diagonale
        K_star_star_diag = self.sigma_f2 + self.sigma_n2
        var_norm = K_star_star_diag - np.sum(v ** 2, axis=0)
        var_norm = np.maximum(var_norm, 1e-8)
        std = np.sqrt(var_norm) * self.y_std
        return mu, std


class BoundaryGaussianProcess:
    """
    Gaussian Process specializzato nella stima della frontiera di convergenza
    (Level Set Estimation) e nella selezione di batch attivi tramite Kriging Believer.
    """

    def __init__(
        self,
        threshold_log: float = CONVERGENCE_THRESHOLD_LOG,
        beta: float = BETA_EXPLORATION,
        random_state: int = 42
    ):
        self.threshold_log = threshold_log
        self.beta = beta
        self.random_state = random_state
        self.scaler = StandardFeatureScaler()
        self.gp: Optional[SciPyGPRegressor] = None
        self.X_train_orig: Optional[np.ndarray] = None
        self.y_train_orig: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Allena il Gaussian Process con kernel Matérn 5/2 ad ARD (length scale per feature).
        """
        self.X_train_orig = np.copy(X)
        self.y_train_orig = np.copy(y)

        # Standardizzazione feature
        X_scaled = self.scaler.fit_transform(X)

        self.gp = SciPyGPRegressor(n_restarts=8, random_state=self.random_state)
        self.gp.fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Restituisce la media mu(x) e la deviazione standard sigma(x) del GP
        in scala logaritmica.
        """
        if self.gp is None:
            raise RuntimeError("Il modello GP non è stato ancora addestrato.")
        X_scaled = self.scaler.transform(X)
        mu, sigma = self.gp.predict(X_scaled, return_std=True)
        return mu, sigma

    def loo_cv_report(self) -> Dict[str, float]:
        """
        Esegue Leave-One-Out Cross-Validation (LOO-CV) sul dataset di addestramento.
        Calcola le predizioni incrociate mu_{-i} e deviazioni standard sigma_{-i}
        escludendo un campione alla volta.
        Restituisce:
        - rmse: Root Mean Square Error sui residui logaritmici
        - coverage_95_pct: frazione empirica di residui standardizzati in [-1.96, 1.96] (%)
        - mean_std_residual: valore atteso del residuo standardizzato (indice di calibrazione/bias)
        - n_samples: numero di campioni valutati
        """
        if self.gp is None or self.X_train_orig is None or self.y_train_orig is None:
            raise RuntimeError("Il modello deve essere addestrato prima di eseguire il report LOO-CV.")

        X_scaled = self.scaler.transform(self.X_train_orig)
        y = self.y_train_orig
        n_samples = len(y)

        # Formula analitica esatta LOO per Gaussian Process (Rasmussen & Williams 2006, eq. 5.12-5.13):
        # mu_{-i} = y_i - alpha_i / [K^{-1}]_{ii}
        # sigma_{-i}^2 = 1 / [K^{-1}]_{ii}
        K = self.gp._matern52_kernel(X_scaled, X_scaled, self.gp.length_scales, self.gp.sigma_f2)
        K[np.diag_indices_from(K)] += self.gp.sigma_n2 + 1e-6
        K_inv = np.linalg.inv(K)

        mu_norm_loo = self.gp.y_train - self.gp.alpha_ / np.diag(K_inv)
        var_norm_loo = 1.0 / np.diag(K_inv)
        sigma_norm_loo = np.sqrt(np.maximum(var_norm_loo, 1e-8))

        mu_loo = mu_norm_loo * self.gp.y_std + self.gp.y_mean
        sigma_loo = sigma_norm_loo * self.gp.y_std

        residuals = y - mu_loo
        std_residuals = residuals / np.maximum(sigma_loo, 1e-6)

        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        coverage_95 = float(np.mean(np.abs(std_residuals) <= 1.96) * 100.0)
        mean_std_res = float(np.mean(std_residuals))

        return {
            "rmse": rmse,
            "coverage_95_pct": coverage_95,
            "mean_std_residual": mean_std_res,
            "n_samples": int(n_samples)
        }

    def acquisition_straddle(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Calcola la funzione di acquisizione Straddle:
        a(x) = beta * sigma(x) - |mu(x) - gamma|
        
        Massimizza l'incertezza epistemica (esplorabilità) pesando la vicinanza
        alla soglia critica di errore del 10% (gamma_log = 1.0).
        """
        dist_to_boundary = np.abs(mu - self.threshold_log)
        return self.beta * sigma - dist_to_boundary

    def generate_candidate_pool(self, fluid_model_filter: Optional[str] = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Genera una griglia densa di nuove configurazioni fisiche candidate nello spazio
        dei parametri a 7 dimensioni (escludendo combinazioni già testate in passato).
        """
        candidates = []
        features_list = []

        # Griglia di discretizzazione derivata coerentemente da PARAM_BOUNDS
        lam_min, lam_max = PARAM_BOUNDS["lambda"]
        etap_min, etap_max = PARAM_BOUNDS["eta_p"]
        lambda_base = [0.008, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.85, 1.00, 1.20, 1.50, 1.80, 2.00]
        lambda_vals = [l for l in lambda_base if lam_min <= l <= lam_max]
        eta_p_base = [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.90, 0.95, 0.98]
        eta_p_vals = [p for p in eta_p_base if etap_min <= p <= etap_max]
        mesh_keys = ["5k", "12k", "29k", "52k", "88k", "125k"]

        # Famiglie costitutive
        model_configs = []
        # Oldroyd-B (is_giesekus=0, is_ptt=0)
        if not fluid_model_filter or fluid_model_filter.lower() in ["oldroyd-b", "oldroyd"]:
            model_configs.append(("Oldroyd-B", 0.0, 0.0))
        # Giesekus (alpha in [0.05, 0.50], is_giesekus=1, is_ptt=0)
        if not fluid_model_filter or fluid_model_filter.lower() == "giesekus":
            for a in [0.05, 0.10, 0.20, 0.35, 0.50]:
                model_configs.append(("Giesekus", a, 0.0))
        # PTT (eps in [0.05, 0.50], is_giesekus=0, is_ptt=1)
        if not fluid_model_filter or fluid_model_filter.lower() == "ptt":
            for e in [0.05, 0.10, 0.20, 0.35, 0.50]:
                model_configs.append(("PTT", 0.0, e))

        for model_name, alpha, eps in model_configs:
            is_giesekus = 1.0 if model_name.lower() == "giesekus" else 0.0
            is_ptt = 1.0 if model_name.lower() == "ptt" else 0.0

            for lam in lambda_vals:
                for eta_p in eta_p_vals:
                    eta_s = float(np.round(1.0 - eta_p, 4))
                    for m_name in mesh_keys:
                        n_pts = MESH_NODES[m_name]
                        log_n = float(np.log10(n_pts))

                        # Feature vector aggiornato (7 dimensioni)
                        feat = [lam, eta_p, alpha, eps, is_giesekus, is_ptt, log_n]
                        
                        # Esclusione di candidati identici a punti già testati
                        if self.X_train_orig is not None:
                            dists = np.linalg.norm(self.X_train_orig - np.array(feat), axis=1)
                            if np.min(dists) < 1e-3:
                                continue

                        features_list.append(feat)
                        candidates.append({
                            "fluid_model": model_name,
                            "lambda": lam,
                            "eta_p": eta_p,
                            "eta_s": eta_s,
                            "alpha": alpha,
                            "eps": eps,
                            "is_giesekus": is_giesekus,
                            "is_ptt": is_ptt,
                            "mesh": m_name,
                            "n_points": n_pts,
                            "features": feat
                        })

        X_cand = np.array(features_list, dtype=np.float64)
        return X_cand, candidates

    def suggest_batch(
        self,
        batch_size: int = 3,
        fluid_model_filter: Optional[str] = None,
        diverse_models: bool = False,
        model_sequence: Optional[List[str]] = None,
        cost_aware: bool = False
    ) -> List[Dict]:
        """
        Algoritmo Kriging Believer:
        Seleziona iterativamente un batch diversificato di candidati ottimali
        aggiornando la covarianza del GP dopo ciascuna selezione.
        Supporta una model_sequence specifica (es. ['Giesekus', 'Giesekus', 'Oldroyd-B'])
        e acquisizione cost-aware pesata sul costo relativo delle mesh.
        """
        if self.gp is None or self.X_train_orig is None or self.y_train_orig is None:
            raise RuntimeError("Il modello deve essere addestrato prima di richiedere un batch.")

        X_cand, candidate_pool = self.generate_candidate_pool(fluid_model_filter=fluid_model_filter)

        # Copia di lavoro del dataset per Kriging Believer
        X_curr = np.copy(self.X_train_orig)
        y_curr = np.copy(self.y_train_orig)

        gp_temp = BoundaryGaussianProcess(
            threshold_log=self.threshold_log,
            beta=self.beta,
            random_state=self.random_state
        )
        gp_temp.fit(X_curr, y_curr)

        selected_batch = []
        available_indices = list(range(len(candidate_pool)))
        chosen_models = set()

        effective_batch_size = len(model_sequence) if model_sequence else batch_size

        for b in range(effective_batch_size):
            # Se è specificata una sequenza per modello per questo slot di batch
            target_model = model_sequence[b] if model_sequence and b < len(model_sequence) else None

            # Filtra gli indici disponibili per il target model se specificato
            if target_model:
                slot_indices = [idx for idx in available_indices if candidate_pool[idx]["fluid_model"].lower() == target_model.lower()]
                if not slot_indices:
                    slot_indices = available_indices
            else:
                slot_indices = available_indices

            # Valutazione di mu e sigma sui candidati eleggibili per lo slot
            X_sub = X_cand[slot_indices]
            mu, sigma = gp_temp.predict(X_sub)
            acq = gp_temp.acquisition_straddle(mu, sigma)

            # Acquisizione Cost-Aware: divide per il costo relativo della mesh
            if cost_aware:
                costs = np.array([MESH_COST.get(candidate_pool[idx]["mesh"], 1.0) for idx in slot_indices], dtype=np.float64)
                acq = np.where(acq > 0, acq / costs, acq * costs)

            # Se è richiesta diversità generica tra modelli
            if diverse_models and not model_sequence:
                acq_span = float(np.ptp(acq)) if np.ptp(acq) > 1e-4 else 1.0
                for idx_sub, orig_idx in enumerate(slot_indices):
                    cand_model = candidate_pool[orig_idx]["fluid_model"]
                    if cand_model in chosen_models:
                        acq[idx_sub] -= 0.5 * acq_span

            best_sub_idx = int(np.argmax(acq))
            best_cand_idx = slot_indices[best_sub_idx]
            chosen = candidate_pool[best_cand_idx]
            chosen_models.add(chosen["fluid_model"])

            pred_mu_log = float(mu[best_sub_idx])
            pred_sigma_log = float(sigma[best_sub_idx])
            chosen_acq = float(acq[best_sub_idx])
            pred_err_pct = float(10.0 ** pred_mu_log)

            chosen_info = dict(chosen)
            chosen_info["pred_err_pct"] = pred_err_pct
            chosen_info["pred_mu_log"] = pred_mu_log
            chosen_info["pred_sigma_log"] = pred_sigma_log
            chosen_info["acquisition_score"] = chosen_acq
            chosen_info["batch_rank"] = b + 1
            chosen_info["cost_aware"] = cost_aware
            chosen_info["mesh_cost"] = MESH_COST.get(chosen["mesh"], 1.0)
            
            # Param tag standard L{lambda}-P{etap}-S{etas}-A{alpha}-E{eps}_M{mesh}
            chosen_info["param_tag"] = (
                f"L{chosen['lambda']}-P{chosen['eta_p']}-S{chosen['eta_s']}-"
                f"A{chosen['alpha']}-E{chosen['eps']}_M{chosen['mesh']}"
            )
            chosen_info["filename"] = f"4_roll_mill_{chosen_info['param_tag']}.csv"

            rationale = self._explain_choice(chosen, pred_err_pct, pred_sigma_log)
            chosen_info["rationale"] = rationale

            selected_batch.append(chosen_info)

            # Kriging Believer: allucina il risultato e aggiorna la covarianza
            hallucinated_y = pred_mu_log
            X_curr = np.vstack([X_curr, chosen["features"]])
            y_curr = np.append(y_curr, hallucinated_y)

            if best_cand_idx in available_indices:
                available_indices.remove(best_cand_idx)
            gp_temp.fit(X_curr, y_curr)

        return selected_batch

    def _explain_choice(self, cand: Dict, pred_err: float, sigma: float) -> str:
        """Fornisce una spiegazione fisica concisa del perché il punto è critico per l'esplorazione."""
        lam = cand["lambda"]
        etap = cand["eta_p"]
        model = cand["fluid_model"]

        reasons = []
        if lam >= 0.5:
            reasons.append(f"Regime ad alta elasticita (lambda={lam}) critico per HWNP")
        elif lam >= 0.2:
            reasons.append(f"Elasticita moderata (lambda={lam}) con gradienti elongazionali intensi")
        else:
            reasons.append(f"Elasticita controllata (lambda={lam})")

        if etap >= 0.85:
            reasons.append(f"frazione polimerica elevata (eta_p={etap}) con solvente minimo (eta_s={cand['eta_s']})")
        
        if model == "Giesekus":
            reasons.append(f"modello Giesekus (alpha={cand['alpha']}) per valutare l'effetto stabilizzante dello shear-thinning")
        elif model == "PTT":
            reasons.append(f"modello PTT (eps={cand['eps']}) per testare l'attenuazione reticolare dello stress")
        elif model == "Oldroyd-B":
            reasons.append("modello Oldroyd-B puro senza attenuazione di stress")

        if sigma > 0.4:
            reasons.append("massima incertezza epistemica nello spazio inesplorato")
        else:
            reasons.append("predizione situata precisamente a ridosso della soglia critica del 10% di errore")

        return "; ".join(reasons) + "."
