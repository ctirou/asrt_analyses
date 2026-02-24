#!/bin/sh

# SLURM options:

#SBATCH --job-name=rsaL_htc           # Job name
#SBATCH --output=rsaL_htc%j.log      # Standard output and error log

#SBATCH --partition=hpc
#SBATCH --cpus-per-task=20              # Run a single task (by default tasks == CPU)
#SBATCH --mem=80G                       # Memory in MB by default
#SBATCH --time=0-08:00:00                # 7 days by default on htc partition
#SBATCH --array=0-14                    # number of subjects (last num included)

# Commands to be submitted:

module load conda
conda activate mne

python rsa_lobo_htc.py                     # run the script
