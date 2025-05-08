#!/usr/bin/bash -l
#
# SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --job-name=lsenet_senet
#SBATCH --output=res_lsenet.txt

unset SLURM_EXPORT_ENV

module load python/3.9-anaconda
module load gcc/11.2.0
# source $WORK/python_virtual_envs/doubly_robust_env/bin/activate # activate the python virtual environment
conda activate clustering_env
python3 ./main.py --dataset=SeNet --data_path=./datasets/affinity_matrix_from_senet_sparse_1000.npz --label_path=./datasets/senet_label_1000.csv --decay_rate=0.1407704753242 --height=3 --lr=0.0001767579111 --lr_pre=0.0003071167291 --r=2.7884273773341 --t=0.0959539047695 --temperature=0.9603757594153 >> output_main_lsenet_single_run.txt

# ./cuda_application
