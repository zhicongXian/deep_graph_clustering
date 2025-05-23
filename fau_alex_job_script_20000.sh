#!/usr/bin/bash -l
#
# SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --job-name=lsenet_senet
#SBATCH --output=res_lsenet_check_cuda_error.txt

unset SLURM_EXPORT_ENV

module load python/3.9-anaconda
module load gcc/11.2.0
# source $WORK/python_virtual_envs/doubly_robust_env/bin/activate # activate the python virtual environment
conda activate clustering_env
python3 ./main.py --dataset=SeNet --data_path=./datasets/affinity_matrix_from_senet_sparse_20000.npz --label_path=./datasets/senet_label_20000.csv>> output_main_lsenet.txt
# ./cuda_application
