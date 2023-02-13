import argparse
import sys
import yaml


def load_yaml_config(path):
    """Load the config file in yaml format.

    Args:
        path (str): Path to load the config file.

    Returns:
        dict: config.
    """
    with open(path, 'r') as infile:
        return yaml.safe_load(infile)


def save_yaml_config(config, path):
    """Load the config file in yaml format.

    Args:
        config (dict object): Config.
        path (str): Path to save the config.
    """
    with open(path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def get_args():
    """Add arguments for parser.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    add_dataset_args(parser)
    add_dag_args(parser)
    add_miss_args(parser)
    add_other_args(parser)

    return parser.parse_args(args=sys.argv[1:])


def add_dataset_args(parser):
    """Add dataset arguments for parser.

    Args:
        parser (argparse.ArgumentParser): Parser.
    """
    parser.add_argument('--n',
                        type=int,
                        default=500,
                        help="Number of samples.")

    parser.add_argument('--d',
                        type=int,
                        default=10,
                        help="Number of nodes.")

    parser.add_argument('--graph_type',
                        type=str,
                        default='ER',
                        help="Type of graph ('ER' or 'SF').")

    parser.add_argument('--degree',
                        type=int,
                        default=4,
                        help="Degree of graph.")

    parser.add_argument('--sem_type',
                        type=str,
                        default='linear',
                        help="Type of the non-linear function ['linear', 'mlp', 'gp', 'gp-add', 'mim']."
                        )

    parser.add_argument('--noise_type',
                        type=str,
                        default='gaussian',
                        help="Type of noise ['gaussian', 'exponential', 'gumbel', 'laplace', 'uniform'].")

    parser.add_argument('--miss_type',
                        type=str,
                        default='mcar',
                        help="Type of missing data ['mcar', 'mar', 'mnar'].")

    parser.add_argument('--miss_percent',
                        type=float,
                        default=0.2,
                        help="Percentage of missing data.")

    parser.add_argument('--mnar_type',
                        type=str,
                        default='logistic',
                        help="Type of the MNAR mechanism ['logistic', 'quantile', 'selfmasked'].")

    parser.add_argument('--p_obs',
                        type=float,
                        default=0.1,
                        help="TODO: Add description.")

    parser.add_argument('--mnar_quantile_q',
                        type=float,
                        default=0.1,
                        help="TODO: Add description.")

    parser.add_argument('--num_sampling',
                        type=int,
                        default=20,
                        help="The sampling sizes of the MCEM method.")

def add_dag_args(parser):
    """Add DAG arguments for parser.

    Args:
        parser (argparse.ArgumentParser): Parser.
    """
    parser.add_argument('--dag_method_type',
                        type=str,
                        default='notears_mlp',
                        help="Type of DAG learning method.")

    parser.add_argument('--lambda_1_ev',
                        type=float,
                        default=2e-2,
                        help="Coefficient of L1 penalty for GOLEM-EV, NOTEARS and GOLEM_NG.")

    parser.add_argument('--lambda_1_nv',
                        type=float,
                        default=2e-3,
                        help="Coefficient of L1 penalty for GOLEM-NV.")

    parser.add_argument('--lambda_2',
                        type=float,
                        default=5.0,
                        help="Coefficient of DAG penalty.")

    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-3,
                        help="Learning rate of Adam optimizer.")

    parser.add_argument('--golem_iter',
                        type=int,
                        default=1e+4,
                        help="Number of iterations for training GOLEM.")

    parser.add_argument('--graph_thres',
                        type=float,
                        default=0.3,
                        help="Threshold for weighted matrix.")

    parser.add_argument('--MLEScore',
                        type=str,
                        default='Sup-G',
                        help="Type of score function ['Sup-G', 'Sub-G'].")


def add_miss_args(parser):
    """Add missing method arguments for parser.

    Args:
        parser (argparse.ArgumentParser): Parser.
    """
    parser.add_argument('--miss_method_type',
                        type=str,
                        default='miss_forest_imputation',
                        help="Type of method for handling missing data.")

    parser.add_argument('--em_iter',
                        type=int,
                        default=10,
                        help="Number of iterations for EM algorithm in MissDAG.")

    parser.add_argument('--equal_variances',
                        dest='equal_variances',
                        action='store_true',
                        help="Assume equal noise variances for likelibood.")

    parser.add_argument('--non_equal_variances',
                        dest='equal_variances',
                        action='store_false',
                        help="Assume non-equal noise variances for likelibood.")


def add_other_args(parser):
    """Add other arguments for parser.

    Args:
        parser (argparse.ArgumentParser): Parser.
    """
    parser.add_argument('--seed',
                        type=int,
                        default=2021,
                        help="Random seed.")
