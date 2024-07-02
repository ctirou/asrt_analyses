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
        
