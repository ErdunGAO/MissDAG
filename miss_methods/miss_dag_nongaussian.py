import logging
import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
from utils.utils import MetricsDAG, postprocess

_logger = logging.getLogger(__name__)

def logcosh(x):
    # s always has real part >= 0
    s = np.sign(x) * x
    p = np.exp(-2 * s)
    return s + np.log1p(p) - np.log(2)

def mean_imputation(X):
    # Use mean of each variable to impute the missing entries
    X = np.copy(X)
    X_df = pd.DataFrame(X)
    for col in X_df:
        X_df[col] = X_df[col].fillna(X_df[col].mean(skipna=True))
    return X_df.values

def P_dis_prob(X_imputed, B_m, sigma, MLEScore):
    sample_num, d = X_imputed.shape
    Residul = X_imputed - X_imputed @ B_m
    P_prob_sep = np.zeros((sample_num, d))
    if MLEScore == 'Sup-G':
        # for j in range(d):
        #     P_prob_sep[:, j] = np.exp(-2 * np.log(np.cosh(Residul[:, j] / sigma[j])))
        P_prob_sep = np.exp(-2 * logcosh(Residul / sigma))

    elif MLEScore == 'Sub-G':
        # for j in range(d):
        #     P_prob_sep[:, j] = np.exp(np.log(np.cosh(Residul[:, j] / sigma[j])))
        P_prob_sep = np.exp(logcosh(Residul / sigma) - np.square(Residul / sigma) / 2)

    p_x = np.prod(P_prob_sep, axis=1)
    return p_x

def num2weight(num_index):
    weight_index = np.ones(np.sum(num_index))
    start = 0
    for i in range(len(num_index)):
        weight_index[start:start + num_index[i]] = 1 / num_index[i]
        start += num_index[i]
    return weight_index.reshape(-1,1)

def Sampling(X_imputed, B_m, q_cond_dis, sigma, MLEScore):
    # This is for rejection sampling
    sample_num, d = X_imputed.shape
    # Calculate p(x)/q(x_mis|x_obs)
    p_joint_dis = P_dis_prob(X_imputed, B_m, sigma, MLEScore)
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

def Adaptive_Sampling(X, Sampling_model, B_m, M_i, sigma, MLEScore, num_sampling):

    one_run_num = 10000
    X_extend = np.tile(X, (one_run_num, 1))
    X_left_list = []
    num_left_list = []

    for i in range(500):
        samples_mispart = Sampling_model.rvs(size=one_run_num, random_state=i)
        q_cond_dis = Sampling_model.pdf(samples_mispart)
        X_extend[:, M_i] = samples_mispart.reshape(one_run_num, -1)
        X_left, num_left = Sampling(X_extend, B_m, q_cond_dis, sigma, MLEScore)
        X_left_list.append(X_left)
        num_left_list.append(num_left)

        if np.sum(num_left_list) >= num_sampling:
            X_final = np.vstack(X_left_list)[0:num_sampling]
            num_final = num_sampling
            return X_final, num_final
        else:
            pass

    X_final = np.vstack(X_left_list)
    num_final = np.sum(num_left_list)

    return X_final, num_final

def E_step(X, B_m, sigma, MLEScore, num_sampling):
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
            X_left, num_left = Adaptive_Sampling(X[i], Sampling_model, B_m, M_i, sigma, MLEScore, num_sampling)
            num_index.append(num_left)
            X_sampling_list.append(X_left)
        else:
            num_index.append(1)
            X_sampling_list.append(X[i])

    X_sampling = np.vstack(X_sampling_list)
    num_index = np.hstack(num_index) # length of nr
    weight_index = num2weight(num_index)

    return X_sampling, num_index, weight_index

def sigma_estimate(X_last, B_m, init=True, weight_index=None, num=None):
    _, d = X_last.shape
    if init:
        residul = X_last - X_last @ B_m
        sigma = np.sum(np.sqrt(np.var(residul, axis=0))) / d
    else:
        sigma = np.sum(np.sqrt(
            np.sum(np.multiply(np.square(X_last - X_last @ B_m), weight_index), axis=0) / num
        )) / d

    return sigma

def miss_dag_nongaussian(X, dag_init_method, dag_method, em_iter=10, MLEScore='Sup-G', num_sampling=20, B_true=None):
    """
    - X corresponds to the observed samples with missing entries (i.e., NaN)
    - mask has the same shape as X; they both have a shape of (n, d)
    - If an entry in mask is False, the corresponding entry in X is a missing value (i.e., NaN)
    """
    # Initial imputation
    X_init_imputed = mean_imputation(X)
    B_m = dag_init_method.fit(X_init_imputed)
    sigma = sigma_estimate(X_init_imputed, B_m, init=True)

    # To store histories of MissDAG
    histories = []

    for m in range(1, em_iter + 1):

        ############ Visulize the results #############
        # Post-process estimated solution
        B_vis = B_m.copy()
        _, B_processed_bin = postprocess(B_vis, 0.3)
        _logger.info("Finished post-processing the estimated graph.")

        raw_result = MetricsDAG(B_processed_bin, B_true).metrics
        _logger.info("run result:{0}".format(raw_result))
        ################################################

        _logger.info("Started the {}th iteration for EM algorithm with DAG learning.".format(m))

        ####### E-step #######
        X_sampling, _, weight_index = E_step(X, B_m, sigma, MLEScore, num_sampling)
        ####### E-step #######

        ####### M-step #######
        # 1. Likelihood of observations
        B_m = dag_method.fit(X_sampling, X.shape[0], weight_index)
        # 2. Modelling the noise by GMM
        sigma = sigma_estimate(X_sampling, B_m, init=False, weight_index=weight_index, num=X.shape[0])
        ####### M-step #######

        # Compute empirical covariance (although it may not make sense for non-Gaussian case)
        cov_m = np.cov(X_sampling.T, bias=True)
        # Save B_m and cov_m to histories
        histories.append({'B_m': B_m, 'cov_m': cov_m})

    return B_m, cov_m, histories