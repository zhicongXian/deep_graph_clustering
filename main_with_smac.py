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
    configs_dict_for_configspace[k] = [v]
    if k == 'height':
        configs_dict_for_configspace[k] = [2, 3, 4, 5, 6]
    # elif k == "r":
    #     configs_dict_for_configspace[k] = (0.001, 10)
    # elif k == "t":
    #     configs_dict_for_configspace[k] = (0.001, 10)
    elif k == "lr":
        configs_dict_for_configspace[k] = (0.0001, 0.02)
    elif k == "lr_pre":
        configs_dict_for_configspace[k] = (0.0001, 0.02)
    elif k == "decay_rate":  # regularization coefficient for zero-norm weights
        configs_dict_for_configspace[k] = (0.01, 0.5)
    elif k == "temperature":
        configs_dict_for_configspace[k] = (0.01, 1)
    # elif k == "n_cluster_trials":
    #     configs_dict_for_configspace[k] = np.arange(5, 20).tolist()

configspace = ConfigurationSpace(configs_dict_for_configspace)
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
    print(torch.cuda.memory_allocated()/1024**2)
    print(torch.cuda.memory_cached()/1024**2)
def train(config: Configuration, seed: int = 0) -> float:
    print("before training run")
    memory_stats()
    torch.cuda.empty_cache()
    memory_stats()
    exp = Exp(DotDict(dict(config)))
    ari = exp.train()
    print("after training run")
    memory_stats()
    torch.cuda.empty_cache()
    memory_stats()
    del exp
    gc.collect()
    return 1 - ari


# Scenario object specifying the optimization environment
scenario = Scenario(configspace, deterministic=False, n_trials=200)

# Use SMAC to find the best configuration/hyperparameters
inital_design = SobolInitialDesign(
    scenario=scenario,
# n_configs_per_hyperparameter=1,
    n_configs=1

)

smac = HyperparameterOptimizationFacade(scenario, train, initial_design=inital_design)
incumbent = smac.optimize()
best_score = smac.runhistory.get_min_cost(incumbent)
# best_parameters = incumbent.get_dictionary()
print(f"best ari found {1 - best_score}")
with open('hpo_best_results.json', 'w') as fp:
    json.dump(dict(incumbent), fp)
