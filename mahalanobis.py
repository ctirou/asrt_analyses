def cv_mahalanobis_parallel(X, y, n_jobs=-1, n_splits=10, verbose=True, shuffle=True):
    """
    Parallelized Cross-validated Mahalanobis distances with tqdm_joblib and NaN safety.

    Parameters:
        X: ndarray (n_trials, n_channels, n_times)
        y: array-like (n_trials,) condition labels
        n_splits: int, number of cross-validation folds
        n_jobs: int, number of parallel jobs (default: -1 = all CPUs)
        verbose: bool, show progress bars

    Returns:
        distances: ndarray (n_times, n_conditions, n_conditions)
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.covariance import LedoitWolf
    from scipy.linalg import solve
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm
    from tqdm_joblib import tqdm_joblib

    n_trials, n_channels, n_times = X.shape
    conditions = np.unique(y)
    n_conditions = len(conditions)

    if shuffle:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        skf = KFold(n_splits=n_splits, shuffle=False)    

    def compute_timepoint(t):
        X_t = X[:, :, t]
        dist_folds = np.zeros((n_conditions, n_conditions, n_splits))

        fold_iterator = skf.split(X_t, y)
        if verbose:
            fold_iterator = tqdm(fold_iterator, total=n_splits, desc=f"Time {t:03}", leave=False, position=t % 8)

        for fold, (train_idx, test_idx) in enumerate(fold_iterator):
            X_train, X_test = X_t[train_idx], X_t[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                lw = LedoitWolf()
                lw.fit(X_train)
                cov = lw.covariance_

                train_means = {c: X_train[y_train == c].mean(axis=0) for c in conditions}
                test_means = {c: X_test[y_test == c].mean(axis=0) for c in conditions}

                for i, ci in enumerate(conditions):
                    for j, cj in enumerate(conditions):
                        if j <= i:
                            continue

                        diff_train = train_means[ci] - train_means[cj]
                        diff_test = test_means[ci] - test_means[cj]

                        dist = diff_train.T @ solve(cov, diff_test, assume_a='pos')
                        dist_folds[i, j, fold] = dist
                        dist_folds[j, i, fold] = dist
            except (ValueError, np.linalg.LinAlgError, KeyError, ZeroDivisionError):
                # Catch common issues: singular covariances, missing classes, empty slices
                dist_folds[:, :, fold] = np.nan

        return np.nanmean(dist_folds, axis=2)  # <- safely ignore folds with NaNs

    time_iterator = range(n_times)

    if verbose:
        with tqdm_joblib(tqdm(desc="Overall Timepoints", total=n_times)) as progress_bar:
            distances = Parallel(n_jobs=n_jobs)(
                delayed(compute_timepoint)(t) for t in time_iterator
            )
    else:
        distances = Parallel(n_jobs=n_jobs)(
            delayed(compute_timepoint)(t) for t in time_iterator
        )

    return np.stack(distances, axis=0)

def train_test_mahalanobis_fast(X_train, X_test, y_train, y_test, n_jobs=-1, verbose=True):
    """
    Computes Mahalanobis distances between class means in training and testing sets.
    Handles missing conditions by filling with NaNs.

    Returns:
        distances: ndarray (n_times, n_conditions, n_conditions)
    """
    import numpy as np
    from sklearn.covariance import LedoitWolf
    from scipy.linalg import solve
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm
    from tqdm_joblib import tqdm_joblib

    n_trials_train, n_channels, n_times = X_train.shape
    n_trials_test, _, _ = X_test.shape
    conditions = [1, 2, 3, 4]
    n_conditions = len(conditions)
    cond_idx = {c: i for i, c in enumerate(conditions)}  # for indexing

    def compute_timepoint(t):
        try:
            Xtr = X_train[:, :, t]
            Xte = X_test[:, :, t]

            # Skip timepoints with too few trials
            if Xtr.shape[0] < 2 or Xte.shape[0] < 2:
                return np.full((n_conditions, n_conditions), np.nan)

            lw = LedoitWolf().fit(Xtr)
            cov = lw.covariance_

            # Only keep conditions present in both train and test
            present_train = set(y_train)
            present_test = set(y_test)
            valid_conditions = list(present_train & present_test)

            train_means = {}
            test_means = {}

            for c in valid_conditions:
                train_means[c] = Xtr[y_train == c].mean(0)
                test_means[c] = Xte[y_test == c].mean(0)

            dist = np.full((n_conditions, n_conditions), np.nan)

            for i, ci in enumerate(conditions):
                for j, cj in enumerate(conditions):
                    if j <= i:
                        continue
                    if ci in train_means and cj in train_means and ci in test_means and cj in test_means:
                        diff_train = train_means[ci] - train_means[cj]
                        diff_test = test_means[ci] - test_means[cj]
                        try:
                            mahal = diff_train.T @ solve(cov, diff_test, assume_a='pos')
                        except np.linalg.LinAlgError:
                            mahal = np.nan
                        dist[cond_idx[ci], cond_idx[cj]] = mahal
                        dist[cond_idx[cj], cond_idx[ci]] = mahal

            return dist

        except Exception as e:
            print(f"Error at time {t}: {e}")
            return np.full((n_conditions, n_conditions), np.nan)

    time_iterator = range(n_times)

    if verbose:
        with tqdm_joblib(tqdm(desc="Computing Mahalanobis", total=n_times)):
            distances = Parallel(n_jobs=n_jobs)(
                delayed(compute_timepoint)(t) for t in time_iterator
            )
    else:
        distances = Parallel(n_jobs=n_jobs)(
            delayed(compute_timepoint)(t) for t in time_iterator
        )

    return np.stack(distances, axis=0)  # shape: (n_times, n_conditions, n_conditions)
