import logging
import numpy as np
import torch
from scipy.stats import multivariate_normal
import pandas as pd

_logger = logging.getLogger(__name__)

def mean_imputation(X):
    # Use mean of each variable to impute the missing entries
    X = np.copy(X)
    X_df = pd.DataFrame(X)
    for col in X_df:
        X_df[col] = X_df[col].fillna(X_df[col].mean(skipna=True))
    return X_df.values

def P_dis_prob(X_imputed, model, sigma, equal_variances=True):
    sample_num, d = X_imputed.shape
    X_torch = torch.from_numpy(X_imputed)
    Residul = (X_torch - model(X_torch)).detach().numpy()
    if equal_variances:
        P_model = multivariate_normal(mean=np.zeros(d), cov=np.diag([np.average(sigma * sigma)] * d))
    else:
        P_model = multivariate_normal(mean=np.zeros(d), cov=np.diag(sigma * sigma))
    p_x = P_model.pdf(Residul)
    return p_x

def num2weight(num_index):
    weight_index = np.ones(np.sum(num_index))
    start = 0
    for i in range(len(num_index)):
        weight_index[start:start + num_index[i]] = 1 / num_index[i]
        start += num_index[i]
    return weight_index.reshape(-1,1)

def Sampling(X_imputed, model, q_cond_dis, sigma, equal_variances):
    # This is for rejection sampling
    sample_num, d = X_imputed.shape
    # Calculate p(x)/q(x_mis|x_obs)
    p_joint_dis = P_dis_prob(X_imputed, model, sigma, equal_variances)
    # Calculate k
    k = np.max(np.true_divide(p_joint_dis, q_cond_dis))
    # Samples selection
    dele_index = []
    for i in range(sample_num):
        random_u = np.random.uniform(0, k * q_cond_dis[i])
        if random_u > p_joint_dis[i]:
            dele_index.append(i)

    X_left = np.delete(X_imputed, dele_index, axis=0)
    num_left = X_left.shape[0]

    return X_left, num_left

def Adaptive_Sampling(X, Sampling_model, model, M_i, sigma, equal_variances):
    # one_run 10000 samples
    minial_num, maximal_num = 30, 50
    one_run_num = 10000
    X_extend = np.tile(X, (one_run_num, 1))
    X_left_list = []
    num_left_list = []

    for i in range(100):
        samples_mispart = Sampling_model.rvs(size=one_run_num, random_state=i)
        q_cond_dis = Sampling_model.pdf(samples_mispart)
        X_extend[:, M_i] = samples_mispart.reshape(one_run_num, -1)
        X_left, num_left = Sampling(X_extend, model, q_cond_dis, sigma, equal_variances)
        X_left_list.append(X_left)
        num_left_list.append(num_left)
        if np.sum(num_left_list) >= maximal_num:
            X_final = np.vstack(X_left_list)[0:maximal_num]
            num_final = maximal_num
            return X_final, num_final
        elif np.sum(num_left_list) >= minial_num:
            X_final = np.vstack(X_left_list)
            num_final = np.sum(num_left_list)
            return X_final, num_final
        else:
            pass

    X_final = np.vstack(X_left_list)
    num_final = np.sum(num_left_list)

    return X_final, num_final

def E_step(X, model, sigma, equal_variances):
    """Sampling some samples"""
    nr, nc = X.shape
    Mu = np.nanmean(X, axis=0)
    S_nan = np.nanvar(X, axis=0) # size of nc

    C = np.isnan(X) == False
    one_to_nc = np.arange(1, nc + 1, step=1)
    M = one_to_nc * (C == False) - 1
    O = one_to_nc * C - 1

    X_sampling_list = []
    num_index = []
    for i in range(nr):
        if set(O[i, ]) != set(one_to_nc - 1):  # missing component exists
            M_i, O_i = M[i, ][M[i, ] != -1], O[i, ][O[i, ] != -1]
            Sampling_model = multivariate_normal(mean=Mu[np.ix_(M_i)], cov=np.diag(S_nan[np.ix_(M_i)]))
            X_left, num_left = Adaptive_Sampling(X[i], Sampling_model, model, M_i, sigma, equal_variances)
            num_index.append(num_left)
            X_sampling_list.append(X_left)
        else:
            num_index.append(1)
            X_sampling_list.append(X[i])

    X_sampling = np.vstack(X_sampling_list)
    num_index = np.hstack(num_index) # length of nr
    weight_index = num2weight(num_index)

    return X_sampling, num_index, weight_index.astype(np.float32)

def sigma_estimate(X_last, model, init=True, weight_index=None, num=None):
    X_torch = torch.from_numpy(X_last)
    if init:
        residul = (X_torch - model(X_torch)).detach().numpy()
        sigma = np.sqrt(np.var(residul, axis=0))
    else:
        sigma = np.sqrt(
            np.sum(
                np.multiply(np.square((X_torch - model(X_torch)).detach().numpy()), weight_index), axis=0) / num
        )
    return sigma

def miss_dag_nonlinear(X, dag_init_method, dag_method, em_iter=10, equal_variances=True):
    """
    - X corresponds to the observed samples with missing entries (i.e., NaN)
    - mask has the same shape as X; they both have a shape of (n, d)
    - If an entry in mask is False, the corresponding entry in X is a missing value (i.e., NaN)
    """
    # Initial imputation
    X_init_imputed = mean_imputation(X)
    B_m, model = dag_init_method.fit(X_init_imputed)
    sigma = sigma_estimate(X_init_imputed, model, init=True)

    # To store histories of MissDAG
    histories = []

    for m in range(1, em_iter + 1):
        _logger.info("Started the {}th iteration for EM algorithm with DAG learning.".format(m))

        ####### E-step #######
        X_sampling, _, weight_index = E_step(X, model, sigma, equal_variances)
        ####### E-step #######

        ####### M-step #######
        # 1. Likelihood of observations
        B_m, model = dag_method.fit(X_sampling, X.shape[0], weight_index)
        # 2. Modelling the noise by GMM
        sigma = sigma_estimate(X_sampling, model, init=False, weight_index=weight_index, num=X.shape[0])
        ####### M-step #######

        # Compute empirical covariance (although it may not make sense for non-Gaussian case)
        cov_m = np.cov(X_sampling.T, bias=True)
        # Save B_m and cov_m to histories
        histories.append({'B_m': B_m, 'cov_m': cov_m})

    return B_m, cov_m, histories