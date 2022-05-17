#!/bin/bash
#SBATCH --job-name=gpipe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END
##SBATCH --mail-user=kp2670@nyu.edu
#SBATCH --output=gpipe.out
#SBATCH --error=gpipe.err
#SBATCH --mem=32GB
#SBATCH --time=07:00:00
#SBATCH --gres=gpu:rtx8000:2


singularity exec --nv --overlay $SCRATCH/singular/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c " source /ext3/env.sh;python3 resdp-mp2.py"
