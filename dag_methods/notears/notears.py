import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid


class Notears:
    def __init__(self, lambda_1_ev):
        self.lambda_1_ev = lambda_1_ev

    def fit(self, X=None, cov_emp=None):
        """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

        Args:
            X (np.ndarray): [n, d] sample matrix
            lambda1 (float): l1 penalty parameter
            loss_type (str): l2, logistic, poisson
            max_iter (int): max num of dual ascent steps
            h_tol (float): exit if |h(w_est)| <= htol
            rho_max (float): exit if rho >= rho_max
            w_threshold (float): drop edge if |weight| < threshold

        Returns:
            W_est (np.ndarray): [d, d] estimated DAG
        """
        def _loss(W):
            """Evaluate value and gradient of loss."""
            I = np.eye(d)
            loss = 0.5 * np.trace((I - W).T @ cov_emp @ (I - W))
            G_loss = - cov_emp @ (I - W)
            return loss, G_loss

        def _h(W):
            """Evaluate value and gradient of acyclicity constraint."""
            E = slin.expm(W * W)  # (Zheng et al. 2018)
            h = np.trace(E) - d
            #     # A different formulation, slightly faster at the cost of numerical stability
            #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
            #     E = np.linalg.matrix_power(M, d - 1)
            #     h = (E.T * M).sum() - d
            G_h = E.T * W * 2
            return h, G_h

        def _adj(w):
            """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
            return (w[:d * d] - w[d * d:]).reshape([d, d])

        def _func(w):
            """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
            W = _adj(w)
            loss, G_loss = _loss(W)
            h, G_h = _h(W)
            obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
            G_smooth = G_loss + (rho * h + alpha) * G_h
            g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
            return obj, g_obj

        assert (X is not None) ^ (cov_emp is not None), "Input only one of X and cov_emp"
        if X is not None:
            cov_emp = np.cov(X.T, bias=True)

        # Default hyperparameters used by NOTEARS
        lambda1 = self.lambda_1_ev
        max_iter, h_tol, rho_max = 100, 1e-8, 1e+16

        d = len(cov_emp)
        w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
        bnds = [(0, 0) if i == j else (0, None) for _ in range(2)
                for i in range(d) for j in range(d)]
        for _ in range(max_iter):
            w_new, h_new = None, None
            while rho < rho_max:
                sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
                w_new = sol.x
                h_new, _ = _h(_adj(w_new))
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
            w_est, h = w_new, h_new
            alpha += rho * h
            if h <= h_tol or rho >= rho_max:
                break
        W_est = _adj(w_est)
        return W_est
