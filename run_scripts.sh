#!/bin/bash

# Activate conda environment
conda activate mne

# Array of values to pass as sys.argv[1]
values=("stim" "button")

# Loop through values and run the script with each value
for val in "${values[@]}"; do
    python /path/to/time_gen.py "$val"
done

# type this
# chmod +x run_script.sh
# and run