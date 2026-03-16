#!/bin/sh

# SLURM options:

#SBATCH --job-name=s-epo           # Job name
#SBATCH --output=s-epo_%j.log      # Standard output and error log

#SBATCH --partition=htc
#SBATCH --cpus-per-task=20              # Run a single task (by default tasks == CPU)
#SBATCH --mem=40G                       # Memory in MB by default
#SBATCH --time=0-00:10:00               # 7 days by default on htc partition
#SBATCH --array=0-14                    # number of subjects (last num included)

# Commands to be submitted:

module load conda
conda activate mne
# python save_epochs.py
# python -m 02_preprocessing.save_reordered_epochs.py
python -m 02_preprocessing.save_reordered_epochs_practice.py