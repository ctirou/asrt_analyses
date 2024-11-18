#!/bin/sh

# SLURM options:

#SBATCH --job-name=timeg           # Job name
#SBATCH --output=timeg-%j.log      # Standard output and error log

#SBATCH --partition=hpc
#SBATCH --cpus-per-task=10              # Run a single task (by default tasks == CPU)
#SBATCH --mem=40G                    # Memory in MB by default
#SBATCH --time=0-02:00:00                    # 7 days by default on htc partition
#SBATCH --array=0-10                    # number of subjects (last num included)

# Commands to be submitted:

module load conda
conda activate mne
python timeg.py
