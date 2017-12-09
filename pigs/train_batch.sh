#!/bin/sh
#
#SBATCH --account=dsi            # The account name for the job.
#SBATCH --job-name=pigs          # The job name.
#SBATCH --gres=gpu:1             # Request 1 gpu (Up to 4 on K80s, or up to 2 on P100s are valid).
#SBATCH -c 1                     # The number of cpu cores to use.
#SBATCH --time=8:00:00           # The time the job will take to run.
#SBATCH --mem-per-cpu=8gb        # The memory the job will use per cpu core.
#SBATCH --output=resnet50-augment2-lambda7e-1-lr1e-2.log
 
module load cuda80/toolkit cuda80/blas cudnn/6.0_8
module load anaconda/3-4.2.0

echo Starting training...
python -u train.py
