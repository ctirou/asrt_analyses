import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

# Simulation parameters
sigma = 0.1
dsize = 'big'
nreps = 100
max_count = 30
max_resamp = 15
max_ptps = 100
nseeds = 100
ntargets = 2
nchunks = 3
class_type = 'svm'

# Define function for creating pseudotrials
def create_pseudotrials(X, y, count, resampling, seed=None):
    np.random.seed(seed)
    n_trials, n_features = X.shape
    trial_indices = np.arange(n_trials)
    trial_usage_count = np.zeros(n_trials, dtype=int)
    n_pseudotrials = int(np.floor(n_trials * resampling / count))
    X_pseudo = np.zeros((n_pseudotrials, n_features))
    y_pseudo = np.zeros(n_pseudotrials, dtype=y.dtype)
    
    for pt in range(n_pseudotrials):
        valid_trials = trial_indices[trial_usage_count < resampling]
        chosen_trials = np.random.choice(valid_trials, size=count, replace=False)
        trial_usage_count[chosen_trials] += 1
        X_pseudo[pt] = X[chosen_trials].mean(axis=0)
        y_pseudo[pt] = np.bincount(y[chosen_trials]).argmax()
        
    return X_pseudo, y_pseudo

# Define classifier
if class_type == 'svm':
    classifier = SVC(kernel='linear')

# Placeholder for results
results = np.nan * np.zeros((max_count, max_resamp, nseeds, max_ptps))

# Main simulation loop
for ptp in range(max_ptps):
    # Generate synthetic dataset for each participant
    X, y = make_classification(n_samples=200, n_features=100, n_informative=50, n_classes=ntargets, flip_y=sigma)
    
    for count in range(1, max_count + 1):
        for resampling in range(1, max_resamp + 1):
            for seed in range(nseeds):
                # Create pseudotrials
                X_pseudo, y_pseudo = create_pseudotrials(X, y, count, resampling, seed=seed)
                
                # Perform classification and store results
                scores = cross_val_score(classifier, X_pseudo, y_pseudo, cv=5)  # 5-fold CV
                results[count-1, resampling-1, seed, ptp] = scores.mean()

# Analysis and plotting example
plt.figure(figsize=(10, 6))
mean_scores = np.nanmean(results, axis=(0, 1, 2))
sns.lineplot(data=mean_scores)
plt.xlabel('Participant')
plt.ylabel('Mean CV Score')
plt.title('Mean Cross-Validation Scores by Participant')
plt.show()
