import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, _class_cov, _class_means, linalg
from sklearn.base import BaseEstimator
from scipy.spatial.distance import euclidean, correlation


class Euclidean2(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        return euclidean(np.mean(X[y == self.classes_[0]], axis=0),
                         np.mean(X[y == self.classes_[1]], axis=0))**2


class CvEuclidean2(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        self.dist_train = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)

        return self

    def predict(self, X, y):

        X = np.array(X)
        y = np.array(y)

        dist_test = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)

        return self.dist_train @ dist_test


class Pearson(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0, return_1d=False):

        self.random_state = random_state
        self.verbose = verbose
        self.return_1d = return_1d

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        d = correlation(np.mean(X[y == self.classes_[0]], axis=0),
                        np.mean(X[y == self.classes_[1]], axis=0))

        if self.return_1d:
            d = np.atleast_1d(d)

        return d


class CvPearson(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0, regularize_var=True, regularize_denom=True,
                 reg_factor_var=0.1, reg_factor_denom=0.25, bounded=True, reg_bounding=1,
                 return_1d=False):

        self.random_state = random_state
        self.verbose = verbose
        self.regularize_var = regularize_var
        self.regularize_denom = regularize_denom
        self.reg_factor_var = reg_factor_var
        self.reg_factor_denom = reg_factor_denom
        self.bounded = bounded
        self.reg_bounding = reg_bounding
        self.return_1d = return_1d

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        self.A1 = np.mean(X[y == self.classes_[0]], axis=0)
        self.B1 = np.mean(X[y == self.classes_[1]], axis=0)
        self.var_A1 = np.var(self.A1)
        self.var_B1 = np.var(self.B1)
        self.denom_noncv = np.sqrt(self.var_A1 * self.var_B1)

        return self

    def predict(self, X, y=None):

        X = np.array(X)

        A2 = np.mean(X[y == self.classes_[0]], axis=0)
        B2 = np.mean(X[y == self.classes_[1]], axis=0)

        cov_a1b2 = np.cov(self.A1, B2)[0, 1]
        cov_b1a2 = np.cov(self.B1, A2)[0, 1]
        cov_ab = (cov_a1b2 + cov_b1a2) / 2

        var_A12 = np.cov(self.A1, A2)[0, 1]
        var_B12 = np.cov(self.B1, B2)[0, 1]

        if self.regularize_var:
            denom = np.sqrt(max(self.reg_factor_var * self.var_A1, var_A12) * max(self.reg_factor_var * self.var_B1, var_B12))
        else:
            denom = np.sqrt(var_A12 * var_B12)
        if self.regularize_denom:
            denom = max(self.reg_factor_denom * self.denom_noncv, denom)

        r = cov_ab / denom

        if self.bounded:
            r = min(max(-self.reg_bounding, r), self.reg_bounding)

        d = 1 - r
        if self.return_1d:
            d = np.atleast_1d(d)
        return d


class LDA(LinearDiscriminantAnalysis):
    """Wrapper to sklearn.discriminant_analysis.LinearDiscriminantAnalysis which allows passing
    a custom covariance matrix (sigma)
    """

    def __init__(self, solver='lsqr', shrinkage=None, priors=None, n_components=None,
                 tol=1e-4, sigma=None):

        super().__init__(solver=solver, shrinkage=shrinkage, priors=priors,
                         n_components=n_components, tol=tol)

        self.sigma = sigma

    def _solve_lsqr(self, X, y, shrinkage):

        self.means_ = _class_means(X, y)
        if self.sigma is not None:
            self.covariance_ = self.sigma
        else:
            self.covariance_ = _class_cov(X, y, self.priors_, shrinkage)
        self.coef_ = linalg.lstsq(self.covariance_, self.means_.T)[0].T
        self.intercept_ = (-0.5 * np.diag(np.dot(self.means_, self.coef_.T))
                           + np.log(self.priors_))
        
# Function to compute cross-validated Mahalanobis distances
def cv_mahalanobis(X, y, n_splits=10):
    """
    Compute cross-validated Mahalanobis distances between conditions for each time point.

    Parameters:
        X: ndarray of shape (n_trials, n_channels, n_times)
           Multivariate time-series data.
        y: ndarray of shape (n_trials,)
           Condition labels for each trial.
        n_splits: int
           Number of cross-validation folds.

    Returns:
        distances: ndarray of shape (n_times, n_conditions, n_conditions)
           Cross-validated Mahalanobis distances between conditions at each time point.
    """
    import numpy as np
    from sklearn.model_selection import KFold
    from scipy.linalg import inv
    n_trials, n_channels, n_times = X.shape
    conditions = np.unique(y)
    n_conditions = len(conditions)
    
    # Initialize array to store distances
    distances = np.zeros((n_times, n_conditions, n_conditions))
    
    # Cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for time_idx in range(n_times):
        X_time = X[:, :, time_idx]  # Data at this time point
        cv_distances = np.zeros((n_conditions, n_conditions, n_splits))
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            # Split data into training and testing
            X_train, X_test = X_time[train_idx], X_time[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Compute the mean and covariance for each condition in training data
            means = {cond: X_train[y_train == cond].mean(axis=0) for cond in conditions}
            cov = np.cov(X_train.T) + 1e-5 * np.eye(n_channels)  # Regularize covariance
            
            # Precompute covariance inverse
            cov_inv = inv(cov)
            
            # Compute Mahalanobis distances for each pair of conditions
            for i, cond1 in enumerate(conditions):
                for j, cond2 in enumerate(conditions):
                    diff_mean = means[cond1] - means[cond2]
                    cv_distances[i, j, fold_idx] = np.sqrt(diff_mean.T @ cov_inv @ diff_mean)
        
        # Average distances across folds
        distances[time_idx] = cv_distances.mean(axis=2)
    
    return distances


# Function to compute LOOCV Mahalanobis distances
def loocv_mahalanobis(X, y):
    """
    Compute leave-one-out cross-validated Mahalanobis distances between conditions for each time point.

    Parameters:
        X: ndarray of shape (n_trials, n_channels, n_times)
           Multivariate time-series data.
        y: ndarray of shape (n_trials,)
           Condition labels for each trial.

    Returns:
        distances: ndarray of shape (n_times, n_conditions, n_conditions)
           LOOCV Mahalanobis distances between conditions at each time point.
    """
    import numpy as np
    from scipy.linalg import inv
    n_trials, n_channels, n_times = X.shape
    conditions = np.unique(y)
    n_conditions = len(conditions)

    # Initialize array to store distances
    distances = np.zeros((n_times, n_conditions, n_conditions))

    for time_idx in range(n_times):
        X_time = X[:, :, time_idx]  # Data at this time point

        # Initialize temporary storage for distances
        temp_distances = np.zeros((n_trials, n_conditions, n_conditions))

        for test_idx in range(n_trials):
            # Split data into training and testing
            train_idx = np.setdiff1d(np.arange(n_trials), test_idx)
            X_train, X_test = X_time[train_idx], X_time[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Compute the mean and covariance for each condition in the training data
            means = {cond: X_train[y_train == cond].mean(axis=0) for cond in conditions}
            cov = np.cov(X_train.T) + 1e-5 * np.eye(n_channels)  # Regularize covariance

            # Precompute covariance inverse
            cov_inv = inv(cov)

            # Compute Mahalanobis distances for each pair of conditions
            for i, cond1 in enumerate(conditions):
                for j, cond2 in enumerate(conditions):
                    diff_mean = means[cond1] - means[cond2]
                    temp_distances[test_idx, i, j] = np.sqrt(diff_mean.T @ cov_inv @ diff_mean)

        # Average distances across LOOCV iterations
        distances[time_idx] = temp_distances.mean(axis=0)

    return distances


from sklearn.base import BaseEstimator
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv

class CvMahalanobis(BaseEstimator):
    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """
        Compute the class-wise mean vectors and the pooled covariance matrix.
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target labels.
        
        Returns:
        self : object
        """
        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)
        self.class_means_ = {}
        n_features = X.shape[1]

        # Compute mean vectors for each class
        for cls in self.classes_:
            self.class_means_[cls] = np.mean(X[y == cls], axis=0)

        # Compute pooled covariance matrix
        cov_matrices = [np.cov(X[y == cls].T) for cls in self.classes_]
        self.pooled_cov_ = np.mean(cov_matrices, axis=0)
        self.inv_pooled_cov_ = inv(self.pooled_cov_)

        return self

    def predict(self, X, y):
        """
        Compute the Mahalanobis distance between class means in the training
        and testing sets.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target labels.
        
        Returns:
        distances : dict
            A dictionary where keys are class pairs and values are their Mahalanobis distances.
        """
        X = np.array(X)
        y = np.array(y)

        test_class_means = {}
        for cls in self.classes_:
            test_class_means[cls] = np.mean(X[y == cls], axis=0)

        # Calculate Mahalanobis distances between all class pairs
        distances = {}
        for cls_train in self.classes_:
            for cls_test in self.classes_:
                dist = mahalanobis(
                    self.class_means_[cls_train], 
                    test_class_means[cls_test], 
                    self.inv_pooled_cov_
                )
                distances[(cls_train, cls_test)] = dist

        return distances
    
from sklearn.base import BaseEstimator
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv

class CvMahalanobisRDM(BaseEstimator):
    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """
        Compute the class-wise mean vectors and the pooled covariance matrix.
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target labels.
        
        Returns:
        self : object
        """
        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)
        self.class_means_ = {}
        n_features = X.shape[1]

        # Compute mean vectors for each class
        for cls in self.classes_:
            self.class_means_[cls] = np.mean(X[y == cls], axis=0)

        # Compute pooled covariance matrix
        cov_matrices = [np.cov(X[y == cls].T) for cls in self.classes_]
        self.pooled_cov_ = np.mean(cov_matrices, axis=0)
        self.inv_pooled_cov_ = inv(self.pooled_cov_)

        return self

    def predict(self, X, y):
        """
        Compute the Representational Dissimilarity Matrix (RDM) based on
        Mahalanobis distances between class means.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target labels.
        
        Returns:
        rdm : np.ndarray of shape (n_classes, n_classes)
            The Representational Dissimilarity Matrix (RDM) where element (i, j)
            is the Mahalanobis distance between class i and class j.
        """
        X = np.array(X)
        y = np.array(y)

        test_class_means = {}
        for cls in self.classes_:
            test_class_means[cls] = np.mean(X[y == cls], axis=0)

        # Initialize the RDM matrix
        n_classes = len(self.classes_)
        rdm = np.zeros((n_classes, n_classes))

        # Compute Mahalanobis distances for all pairs of classes
        for i, cls_train in enumerate(self.classes_):
            for j, cls_test in enumerate(self.classes_):
                dist = mahalanobis(
                    self.class_means_[cls_train],
                    test_class_means[cls_test],
                    self.inv_pooled_cov_
                )
                rdm[i, j] = dist

        return rdm
    
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv

class CvMahalanobisND:
    def __init__(self):
        self.class_means_ = {}
        self.inv_pooled_cov_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Compute the class-wise mean vectors and the pooled covariance matrix.
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target labels.
        
        Returns:
        self : object
        """
        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)
        self.class_means_ = {}

        # Compute mean vectors for each class
        for cls in self.classes_:
            self.class_means_[cls] = np.mean(X[y == cls], axis=0)

        # Compute pooled covariance matrix
        cov_matrices = [np.cov(X[y == cls].T) for cls in self.classes_]
        pooled_cov = np.mean(cov_matrices, axis=0)
        self.inv_pooled_cov_ = inv(pooled_cov)

        return self

    def predict(self, X, y):
        """
        Compute the RDM for each slice along the last dimension of X.

        Parameters:
        X : array-like of shape (n_samples, n_features, n_slices)
            Feature tensor.
        y : array-like of shape (n_samples,)
            Target labels.
        
        Returns:
        rdm_slices : list of np.ndarray
            A list of RDMs, one for each slice in the last dimension of X.
        """
        X = np.array(X)
        y = np.array(y)

        n_slices = X.shape[-1]
        rdm_slices = []

        for slice_idx in range(n_slices):
            X_slice = X[:, :, slice_idx]

            # Compute class means for the current slice
            slice_class_means = {}
            for cls in self.classes_:
                slice_class_means[cls] = np.mean(X_slice[y == cls], axis=0)

            # Initialize the RDM for this slice
            n_classes = len(self.classes_)
            rdm = np.zeros((n_classes, n_classes))

            # Compute Mahalanobis distances for all pairs of classes
            for i, cls_train in enumerate(self.classes_):
                for j, cls_test in enumerate(self.classes_):
                    dist = mahalanobis(
                        self.class_means_[cls_train],
                        slice_class_means[cls_test],
                        self.inv_pooled_cov_
                    )
                    rdm[i, j] = dist

            rdm_slices.append(rdm)

        return rdm_slices
