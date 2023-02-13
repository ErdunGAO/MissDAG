import numpy as np
import pandas as pd


def get_mask_info(mask):
    d = mask.shape[1]
    mask_df = pd.DataFrame(mask)
    mask_df['observed_indices'] = mask_df.apply(lambda row: tuple(np.where(row)[0]), axis=1)
    mask_df['missed_indices'] = mask_df.apply(
        lambda row: tuple(set(range(d)) - set(row['observed_indices'])),
        axis=1
    )
    return mask_df[['observed_indices', 'missed_indices']]


def get_miss_group_info(mask):
    """
    Group the data by the missing variables to save time during the EM algorithm,
    so that we could do the E-step group by group, instead of one by one
    """
    mask_info_df = get_mask_info(mask)
    miss_group_df = pd.DataFrame(
        mask_info_df.groupby(['observed_indices', 'missed_indices']).apply(lambda x: x.index.tolist()),
        columns=['sample_indices']
    )
    miss_group_df = miss_group_df.reset_index()
    return miss_group_df


def compute_outer_product(X, left_indices, right_indices):
    X_left = np.copy(X[:, left_indices]).T
    X_right = np.copy(X[:, right_indices]).T
    return np.einsum('ik,jk->ijk', X_left, X_right)


def E_step(X, miss_group_df, mu_m, K_m):
    n, d = X.shape
    X_m = X.copy()
    cross_X_m = np.zeros((d, d, n))

    # Group the data by the missing variables to save time, so that we could do
    # the E-step group by group, instead of one by one
    for _, observed_indices, missed_indices, sample_indices in miss_group_df.itertuples():
        if len(missed_indices) == 0:
            # No missing data
            cross_X_m[np.ix_(observed_indices, observed_indices, sample_indices)] \
                = compute_outer_product(X_m[sample_indices], observed_indices, observed_indices)
            continue

        # Need to convert to lists for numpy indexing
        observed_indices = list(observed_indices)
        missed_indices = list(missed_indices)

        # i indicates missing variables, j indicates observed variables
        if len(observed_indices) == 0:
            mu_i = mu_m[missed_indices]
            K_ii = K_m[np.ix_(missed_indices, missed_indices)]
            K_ii_inv = np.linalg.inv(K_ii)
            c = mu_i[:, np.newaxis]
            # For E[x_ij | x_{obs,i}, mu^m, K^m]
            X_m[np.ix_(sample_indices, missed_indices)] = np.tile(c.T, (len(sample_indices), 1))
            cross_X_m[np.ix_(missed_indices, missed_indices, sample_indices)] \
                = compute_outer_product(X_m[sample_indices], missed_indices, missed_indices) \
                    + K_ii_inv[:, :, np.newaxis]
        else:
            mu_i = mu_m[missed_indices]
            mu_j = mu_m[observed_indices]
            K_ii = K_m[np.ix_(missed_indices, missed_indices)]
            K_ij = K_m[np.ix_(missed_indices, observed_indices)]
            K_ii_inv = np.linalg.inv(K_ii)
            X_j = X_m[np.ix_(sample_indices, observed_indices)]
            c = (mu_i[:, np.newaxis] - K_ii_inv @ K_ij @ (X_j - mu_j).T).T
            # For E[x_ij | x_{obs,i}, mu^m, K^m]
            X_m[np.ix_(sample_indices, missed_indices)] = c

            # For E[x_ij, x_ij' | x_{obs,i}, mu^m, K^m]
            cross_X_m[np.ix_(observed_indices, observed_indices, sample_indices)] \
                = compute_outer_product(X_m[sample_indices], observed_indices, observed_indices)
            cross_X_m[np.ix_(observed_indices, missed_indices, sample_indices)] \
                = compute_outer_product(X_m[sample_indices], observed_indices, missed_indices)
            cross_X_m[np.ix_(missed_indices, observed_indices, sample_indices)] \
                = compute_outer_product(X_m[sample_indices], missed_indices, observed_indices)
            cross_X_m[np.ix_(missed_indices, missed_indices, sample_indices)] \
                = compute_outer_product(X_m[sample_indices], missed_indices, missed_indices) \
                    + K_ii_inv[:, :, np.newaxis]

    # Compute T1 and T2
    T1_m = X_m.sum(axis=0)
    T2_m = cross_X_m.sum(axis=2)
    return T1_m, T2_m, X_m