#!/bin/bash
#SBATCH --job-name=non_distribute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4     
#SBATCH --cpus-per-task=4
#SBATCH --output=non_distribute.out
#SBATCH --error=non_distribute.out
#SBATCH --mem=16GB
#SBATCH --time=07:00:00
#SBATCH --gres=gpu:1

pip install wandb
pip install torchsummary

cd /scratch/mnk2978/hpml/finalproj/non_distribute/

python main.py



