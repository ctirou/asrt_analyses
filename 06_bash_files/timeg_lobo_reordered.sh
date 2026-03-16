#!/bin/sh

# SLURM options:

#SBATCH --job-name=tg_reord           # Job name
#SBATCH --output=tg_reord-%j.log      # Standard output and error log

#SBATCH --partition=htc
#SBATCH --cpus-per-task=20              # Run a single task (by default tasks == CPU)
#SBATCH --mem=12G                    # Memory in MB by default
#SBATCH --time=0-00:10:00                    # 7 days by default on htc partition
#SBATCH --array=0-14                    # number of subjects (last num included)

# Commands to be submitted:

module load conda
conda activate mne
# python -m 03_sensors.time_gen.timeg_lobo_reordered.py
python -m 03_sensors.time_gen.timeg_lobo_reordered_practice.py