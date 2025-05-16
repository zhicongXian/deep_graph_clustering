import torch
import numpy as np
import os
import random
import argparse
from exp import Exp
from logger import create_logger
import json
from utils.train_utils import DotDict
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import Configuration, ConfigurationSpace
import gc
from smac.initial_design.sobol_design import SobolInitialDesign
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
import optuna
import logging
import sys

seed = 3047
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description='Lorentz Structural Entropy')

# Experiment settings
parser.add_argument('--dataset', type=str, default='FootBall')
parser.add_argument('--task', type=str, default='Clustering',
                    choices=['Clustering'])
parser.add_argument('--root_path', type=str, default='./datasets')
parser.add_argument('--eval_freq', type=int, default=10)
parser.add_argument('--exp_iters', type=int, default=5)
parser.add_argument('--version', type=str, default="run")
parser.add_argument('--log_path', type=str, default="./results/FootBall.log")

parser.add_argument('--pre_epochs', type=int, default=200, help='the training epochs for pretraining')
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--height', type=int, default=2)
parser.add_argument('--lr_pre', type=float, default=1e-3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--w_decay', type=float, default=0.3)
parser.add_argument('--decay_rate', type=float, default=None)
parser.add_argument('--max_nums', type=int, nargs='+', default=[10], help="such as [50, 7]")
parser.add_argument('--embed_dim', type=int, default=2)
parser.add_argument('--hidden_dim_enc', type=int, default=16)
parser.add_argument('--hidden_dim', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--nonlin', type=str, default=None)
parser.add_argument('--temperature', type=float, default=0.05)
parser.add_argument('--n_cluster_trials', type=int, default=5)
parser.add_argument('--t', type=float, default=1., help='for Fermi-Dirac decoder')
parser.add_argument('--r', type=float, default=2., help='Fermi-Dirac decoder')

parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--save_path', type=str, default='model.pt')
# GPU
parser.add_argument('--use_gpu', action='store_false', help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1',
                    help='device ids of multiple gpus')
parser.add_argument('--data_path', type=str, default='./datasets/affinity_matrix_from_senet_sparse_1000.npz')
parser.add_argument('--label_path', type=str, default='./datasets/senet_label_1000.csv')

configs = parser.parse_args()
# with open(f'./configs/{configs.dataset}.json', 'wt') as f:
#     json.dump(vars(configs), f, indent=4)
configs_dict = vars(configs)
with open(f'./configs/{configs.dataset}.json', 'rt') as f:
    configs_dict.update(json.load(f))
configs = DotDict(configs_dict)
f.close()
configs_dict_for_configspace = dict()
for k, v in configs_dict.items():
    if k not in ['height', "r", "t", "lr", "lr_pre", "decay_rate", "temperature"]:
        configs_dict_for_configspace[k] = [v]

configspace = ConfigurationSpace()  # configs_dict_for_configspace)

# n_cluster_trials_config])
# Scenario object specifying the optimization environment
scenario = Scenario(configspace, deterministic=False, n_trials=200)

log_path = f"./results/{configs.version}/{configs.dataset}.log"
configs.log_path = log_path
if not os.path.exists(f"./results"):
    os.mkdir("./results")
if not os.path.exists(f"./results/{configs.dataset}"):
    os.mkdir(f"./results/{configs.dataset}")
if not os.path.exists(f"./results/{configs.version}"):
    os.mkdir(f"./results/{configs.version}")
print(f"Log path: {configs.log_path}")
logger = create_logger(configs.log_path)
logger.info(configs)


def memory_stats():
    print(torch.cuda.memory_allocated() / 1024 ** 2)
    print(torch.cuda.memory_cached() / 1024 ** 2)


def train(trial) -> float:
    torch.cuda.empty_cache()
    hyper_config = dict()
    height = trial.suggest_categorical('height', [2, 3, 4, 5])
    r = trial.suggest_float('r', 0.001, 10, log=True)
    t = trial.suggest_float('t', 0.001, 10, log=True)
    lr_pre = trial.suggest_float('lr_pre', 0.0001, 0.05, log=True)
    lr = trial.suggest_float("lr", 0.0001, 0.05, log=True)
    decay_rate = trial.suggest_float("decay_rate", 0.0, 0.5)
    temperature = trial.suggest_float("temperature", 0.01, 1, log=True)

    hyper_config["height"] = height
    hyper_config["r"] = r
    hyper_config["t"] = t
    hyper_config["lr_pre"] = lr_pre
    hyper_config["lr"] = lr
    hyper_config["decay_rate"] = decay_rate
    hyper_config["temperature"] = temperature

    exp = Exp(configs, DotDict(hyper_config))
    ari = exp.train()

    torch.cuda.empty_cache()

    gc.collect()
    return 1 - ari


# Scenario object specifying the optimization environment
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "lse_optuna"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)

study = optuna.create_study(study_name=study_name, storage=storage_name)
study.enqueue_trial(
    {
        "r": 2.0,
        "t": 2.0,
        "lr": 1e-3,
        'lr_pre': 1e-3,
        'temperature': 1e-3,
        "height": 4
    }
)
study.optimize(train, n_trials=200)
# Use SMAC to find the best configuration/hyperparameters
