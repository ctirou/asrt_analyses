#!/bin/sh

# SLURM options:

#SBATCH --job-name=rsa           # Job name
#SBATCH --output=rsa_%j.log      # Standard output and error log

#SBATCH --partition=hpc
#SBATCH --cpus-per-task=5              # Run a single task (by default tasks == CPU)
#SBATCH --mem=10G                       # Memory in MB by default
#SBATCH --time=0-00:15:00                # 7 days by default on htc partition
#SBATCH --array=0-10                    # number of subjects (last num included)

# Commands to be submitted:

# export CLUSTER_ENV = 1
module load conda
conda activate mne

python ./sensors_/rsa_cluster.py                           # run the script