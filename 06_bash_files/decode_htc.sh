#!/bin/sh

# SLURM options:

#SBATCH --job-name=dec_htc           # Job name
#SBATCH --output=dec_htc-%j.log      # Standard output and error log

#SBATCH --partition=htc
#SBATCH --cpus-per-task=20              # Run a single task (by default tasks == CPU)
#SBATCH --mem=90G                    # Memory in MB by default
#SBATCH --time=0-02:00:00                    # 7 days by default on htc partition
#SBATCH --array=0-14                    # number of subjects (last num included)

# Commands to be submitted:

module load conda
conda activate mne
python -m 04_source.time_gen.decode_htc.py