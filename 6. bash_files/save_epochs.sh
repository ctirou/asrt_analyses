#!/bin/sh

# SLURM options:

#SBATCH --job-name=s-epo           # Job name
#SBATCH --output=s-epo_%j.log      # Standard output and error log

#SBATCH --partition=hpc
#SBATCH --cpus-per-task=20              # Run a single task (by default tasks == CPU)
#SBATCH --mem=40G                       # Memory in MB by default
#SBATCH --time=0-02:00:00                # 7 days by default on htc partition
#SBATCH --array=0-14                    # number of subjects (last num included)

# Commands to be submitted:

module load conda
conda activate mne
python save_epochs.py                           # run the script