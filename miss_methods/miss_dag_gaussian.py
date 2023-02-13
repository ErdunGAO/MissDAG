import logging

import numpy as np

from miss_methods.em_utils import get_miss_group_info, E_step


_logger = logging.getLogger(__name__)


def miss_dag_gaussian(X, mask, dag_method, em_iter=20, equal_variances=True):
    """
    - X corresponds to the observed samples with missing entries (i.e., NaN)
    - mask has the same shape as X; they both have a shape of (n, d)
    - If an entry in mask is False, the corresponding entry in X is a missing value (i.e., NaN)
    """
    n, d = X.shape
    miss_group_df = get_miss_group_info(mask)

    # Initial values
    mu_init = np.zeros((d))
    B_init = np.zeros((d, d))
    # TODO: Find a better way to initialize Omega_init
    # We do not initialize the diagonals to 1.0 to avoid leaking prior information,
    # because that is the variance used in the synthetic data (for equal noise variances)
    Omega_init = np.diag(np.random.uniform(low=0.1, high=0.3, size=(d,)))

    # Variables for EM algorithms
    mu_m = mu_init
    B_m = B_init
    Omega_m = Omega_init
    I = np.eye(d)    # Constant

    # To store histories of MissDAG
    histories = []

    for m in range(1, em_iter + 1):
        _logger.info("Started the {}th iteration for EM algorithm with DAG learning.".format(m))
        K_m = (I - B_m) @ np.linalg.inv(Omega_m) @ (I - B_m).T

        ###################### E-step ######################
        T1_m, T2_m, _ = E_step(X, miss_group_df, mu_m, K_m)
        ####################################################

        ###################### M-step ######################
        mu_m = T1_m / n
        S_m = T2_m / n - np.outer(mu_m, mu_m)
        B_m = dag_method.fit(X=None, cov_emp=S_m)
        if equal_variances:
            sigma_m = np.trace((I - B_m).T @ S_m @ (I - B_m)) / d
            Omega_m = np.diag([sigma_m] * d)
        else:
            Omega_m = np.diag(np.diagonal((I - B_m).T @ S_m @ (I - B_m)))
        ####################################################

        # Save B_m and S_m to histories
        histories.append({'B_m': B_m, 'cov_m': S_m})

    B_est = B_m
    cov_est = S_m
    return B_est, cov_est, histories