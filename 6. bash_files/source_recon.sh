#!/bin/sh

# SLURM options:

#SBATCH --job-name=src_rec           # Job name
#SBATCH --output=src_rec_%j.log      # Standard output and error log

#SBATCH --partition=htc
#SBATCH --cpus-per-task=20              # Run a single task (by default tasks == CPU)
#SBATCH --mem=15G                       # Memory in MB by default
#SBATCH --time=0-00:30:00                # 7 days by default on htc partition
#SBATCH --array=0-14                    # number of subjects (last num included)

# Commands to be submitted:

module load conda
conda activate mne
python source_recon.py                           # run the script
