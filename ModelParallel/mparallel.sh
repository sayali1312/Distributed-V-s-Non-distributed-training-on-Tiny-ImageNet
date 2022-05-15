#!/bin/bash
#SBATCH --job-name=mparallel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=4
#SBATCH --output=mparallel.out
#SBATCH --error=mparallel.out
#SBATCH --mem=16GB
#SBATCH --time=07:00:00
#SBATCH --gres=gpu:2

cd /scratch/mnk2978/hpml/finalproj/ModelParallel
python Resnet50_mp.py