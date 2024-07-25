import os

# Define the directory path, the string to replace, and the new string
directory_path = '/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/results/fwd/stim'

# Change the current working directory to the specified path
os.chdir(directory_path)

# List all files in the directory
files = os.listdir()

# Loop through each file in the directory
for file in files:
    # Check if the file name contains the old string
    if 'lh-mixed-fwd' in file:
        new_file = file.replace('lh-mixed-fwd', 'lh-fwd')
    elif 'rh-mixed-fwd' in file:
        new_file = file.replace('rh-mixed-fwd', 'rh-fwd')
    elif 'others-mixed-fwd' in file:
        new_file = file.replace('others-mixed-fwd', 'others-fwd')
    # Rename the file
    os.rename(file, new_file)
    print(f'Renamed "{file}" to "{new_file}"')
        
import os
import shutil

def rename_and_move_scores(base_dir):
    for brain_region in os.listdir(base_dir):
        brain_region_path = os.path.join(base_dir, brain_region)
        
        # Ensure we're only processing directories
        if os.path.isdir(brain_region_path):
            for subject in os.listdir(brain_region_path):
                subject_path = os.path.join(brain_region_path, subject)
                
                if os.path.isdir(subject_path):
                    scores_file_path = os.path.join(subject_path, 'scores.npy')
                    
                    if os.path.isfile(scores_file_path):
                        # Define new filename and paths
                        new_filename = f"{subject}-scores.npy"
                        new_file_path = os.path.join(brain_region_path, new_filename)
                        
                        # Move and rename the file
                        os.rename(scores_file_path, new_file_path)
                        print(f"Renamed and moved {scores_file_path} to {new_file_path}")
                        
                        # Remove the subject folder
                        shutil.rmtree(subject_path)
                        print(f"Deleted {subject_path}")

# Define the base directory (adjust the path as needed)
base_dir = '/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/results/concatenated/source/stim/pattern'
rename_and_move_scores(base_dir)
