#!/bin/bash
#SBATCH --job-name=dparallel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4     
#SBATCH --cpus-per-task=4
#SBATCH --output=dparallel.out
#SBATCH --error=dparallel.out
#SBATCH --mem=16GB
#SBATCH --time=07:00:00
#SBATCH --gres=gpu:2

cd /scratch/mnk2978/hpml/finalproj/dataparallel

python3 -m torch.distributed.launch resdp.py -a resnet50 --dist-backend 'nccl' --world-size 1 --rank 0