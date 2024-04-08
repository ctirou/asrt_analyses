#!/bin/bash

# Define your subjects array
subjects=('sub01' 'sub02' 'sub04' 'sub07' 'sub08' 'sub09' 'sub10' 'sub12' 'sub13' 'sub14' 'sub15')

# Max parallel jobs, set to the number of subjects
max_jobs=11

# Specify the path to the Python executable in your Conda environment
python_path="/Users/coum/opt/anaconda3/envs/mne/bin/python"

for i in "${!subjects[@]}"; do
    (
        # Use the specific Python executable to run your script
        export SLURM_ARRAY_TASK_ID=$i
        $python_path decoding_pred.py
    ) &
    
    # Wait for all processes to complete before exiting
    if (( i % max_jobs == 0 )); then
        wait
    fi
done

# Wait for any remaining jobs to complete
wait
