#!/bin/sh

# SLURM options:

#SBATCH --job-name=tgf_net           # Job name
#SBATCH --output=tgf-net-%j.log      # Standard output and error log

#SBATCH --partition=hpc
#SBATCH --cpus-per-task=20              # Run a single task (by default tasks == CPU)
#SBATCH --mem=100G                    # Memory in MB by default
#SBATCH --time=0-8:00:00                    # 7 days by default on htc partition
#SBATCH --array=0-14                    # number of subjects (last num included)

# Commands to be submitted:

module load conda
conda activate mne

python timeg_sess_net_fast.py