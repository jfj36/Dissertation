import numpy as np
import pandas as pd
import logging
from scipy.stats import norm

# Base classes
from sklearn.base import BaseEstimator, MetaEstimatorMixin, ClassifierMixin, clone as skclone
from sklearn.ensemble._base import _set_random_states
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, log_loss, precision_recall_curve, roc_curve
)
from sklearn.utils import (
    check_random_state, resample, check_X_y, check_array
)
from sslearn.utils import calculate_prior_probability

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SetredScratch(BaseEstimator, MetaEstimatorMixin):
    """
    SetredScratch is a semi-supervised learning wrapper that enhances self-training
    by applying a statistical rejection criterion for selecting pseudolabeled data.
    """

    def __init__(self,
                 base_estimator=KNeighborsClassifier(n_neighbors=3),
                 max_iterations=40,
                 distance="euclidean",
                 poolsize=0.25,
                 rejection_threshold=0.05,
                 graph_neighbors=7,
                 random_state=None,
                 n_jobs=None,
                 htunning=False,
                 param_grid=None,
                 n_simulations=100,
                 X_label_real=None,
                 y_label_real=None,
                 y_unlabel=None,
                 messages=True,
                 view=100):
        """
        Initialize the SetredScratch classifier with user-defined hyperparameters.

        Parameters define the base estimator, graph construction, simulation settings,
        hyperparameter tuning control, and logging options.
        """
        self.base_estimator = base_estimator
        self.max_iterations = max_iterations
        self.poolsize = poolsize
        self.distance = distance
        self.rejection_threshold = rejection_threshold
        self.graph_neighbors = graph_neighbors
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.htunning = htunning
        self.param_grid = param_grid
        self.n_simulations = n_simulations
        self.X_label_real = X_label_real
        self.y_label_real = y_label_real
        self.y_unlabel = y_unlabel
        self.view = view
        self.messages = messages

    def get_dataset(self, X, y):
        """
        Separate labeled and unlabeled data from input features and targets.

        Parameters:
            X (array-like): Feature matrix
            y (array-like): Target vector with -1 for unlabeled

        Returns:
            Tuple of labeled and unlabeled features and labels
        """
        is_df = isinstance(X, pd.DataFrame)
        columns = X.columns if is_df else None

        X = check_array(X)
        y = check_array(y, ensure_2d=False, dtype=y.dtype.type)

        X_label = X[y != -1]
        y_label = y[y != -1]
        X_unlabel = X[y == -1]

        X_label, y_label = check_X_y(X_label, y_label)

        if is_df:
            X_label = pd.DataFrame(X_label, columns=columns)
            X_unlabel = pd.DataFrame(X_unlabel, columns=columns)

        return X_label, y_label, X_unlabel

    def __create_neighborhood(self, X):
        """
        Construct a neighborhood graph from data using k-nearest neighbors.

        Parameters:
            X (array-like): Feature matrix

        Returns:
            Graph matrix representing neighborhood distances
        """
        return kneighbors_graph(
            X, n_neighbors=self.graph_neighbors,
            mode="distance", metric=self.distance, n_jobs=self.n_jobs
        ).toarray()

    def simulate_ji_matrix(self, p_wrong, weights, weights_sum, weights_square_sum, random_state):
        """
        Simulate the test statistic under the null hypothesis H0.

        Parameters:
            p_wrong (array): Probabilities of incorrect label
            weights (array): Edge weights in neighborhood graph
            weights_sum (array): Sum of weights per instance
            weights_square_sum (array): Sum of squared weights per instance
            random_state (RandomState): Random generator

        Returns:
            Dictionary of simulated statistics (ji_matrix, zi_matrix, oi_matrix)
        """
        n_instances, n_neighbors = weights.shape
        p_matrix = np.repeat(p_wrong, n_neighbors).reshape(weights.shape)

        ji_matrix = np.zeros((n_instances, self.n_simulations))
        zi_matrix = np.zeros_like(ji_matrix)
        oi_matrix = np.zeros_like(ji_matrix)

        mu_h0 = p_wrong * weights_sum
        sigma_h0 = np.sqrt((1 - p_wrong) * p_wrong * weights_square_sum)

        for s in range(self.n_simulations):
            iid_random = random_state.binomial(1, p_matrix)
            ji = (iid_random * weights).sum(axis=1)
            ji_matrix[:, s] = ji

            z_score = np.divide((ji - mu_h0), sigma_h0, out=np.zeros_like(sigma_h0), where=sigma_h0 != 0)
            zi_matrix[:, s] = z_score
            oi_matrix[:, s] = norm.sf(abs(z_score), 0, 1)

        return {
            "ji_matrix": ji_matrix,
            "zi_matrix": zi_matrix,
            "oi_matrix": oi_matrix
        }

    def compare_to_observed(self, jiobs, ji_matrix):
        """
        Compare observed ji statistics against simulated matrix.

        Parameters:
            jiobs (array): Observed test statistics
            ji_matrix (array): Simulated test statistics

        Returns:
            Array of empirical p-values
        """
        return 1 - np.mean(jiobs[:, None] < ji_matrix, axis=1)

    def predict(self, X, **kwargs):
        """
        Predict class labels for input samples.

        Parameters:
            X (array-like): Input features

        Returns:
            Predicted class labels
        """
        return self._base_estimator.predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        """
        Predict class probabilities for input samples.

        Parameters:
            X (array-like): Input features

        Returns:
            Predicted probabilities per class
        """
        return self._base_estimator.predict_proba(X, **kwargs)

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters:
            X (array-like): Test data
            y (array-like): True labels
            sample_weight (array-like, optional): Weights for samples

        Returns:
            Accuracy score
        """
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def fit(self, X, y, **kwargs):
        """
        Train the SetredScratch classifier on the input labeled and unlabeled data.

        Parameters:
            X (array-like): Input features
            y (array-like): Labels, with -1 indicating unlabeled instances

        Returns:
            self: Fitted estimator
        """
        logger.info("Starting training process...")
        random_state = check_random_state(self.random_state)
        y_unlabel = self.y_unlabel
        X_label, y_label, X_unlabel = self.get_dataset(X, y)

        is_df = isinstance(X_label, pd.DataFrame)
        self.classes_ = np.unique(y_label)
        each_iteration_candidates = X_label.shape[0]
        pool = int(len(X_unlabel) * self.poolsize)

        self._base_estimator = skclone(self.base_estimator)
        self._base_estimator.fit(X_label, y_label, **kwargs)

        y_probabilities = calculate_prior_probability(y_label)
        sort_idx = np.argsort(list(y_probabilities.keys()))

        if X_unlabel.shape[0] == 0:
            return self

        iteration = 0
        accuracy = []

        for _ in range(self.max_iterations):
            iteration += 1
            U_, yU_ = resample(X_unlabel, y_unlabel, replace=False, n_samples=pool, random_state=random_state)
            U_ = pd.DataFrame(U_, columns=X_label.columns) if is_df else U_

            raw_predictions = self._base_estimator.predict_proba(U_)
            predictions = np.max(raw_predictions, axis=1)
            class_predicted = np.argmax(raw_predictions, axis=1)
            indexes = predictions.argsort()[-each_iteration_candidates:]

            L_ = U_.iloc[indexes] if is_df else U_[indexes]
            y_ = np.array([self._base_estimator.classes_[x] for x in class_predicted[indexes]])
            yL_ = yU_[indexes]

            pre_L = pd.concat([X_label, L_]) if is_df else np.concatenate((X_label, L_), axis=0)
            pre_yL = pd.concat([y_label, pd.Series(y_)]) if is_df else np.concatenate((y_label, y_), axis=0)

            weights = self.__create_neighborhood(pre_L)
            iid_observed = (pre_yL[:, None] != pre_yL[None, :]).astype(int)
            weights = weights[-L_.shape[0]:, :]
            iid_observed = iid_observed[-L_.shape[0]:, :]

            idx = np.searchsorted(np.array(list(y_probabilities.keys())), y_, sorter=sort_idx)
            p_wrong = 1 - np.asarray(list(y_probabilities.values()))[sort_idx][idx]

            # Weights
            weights[weights != 0] += 1
            weights = np.divide(1, weights, out=np.zeros_like(weights), where=weights != 0)
            weights_sum = weights.sum(axis=1)
            weights_square_sum = (weights ** 2).sum(axis=1)
            
            # Hypothesis testing
            ## Observed statistics
            jiobs = (iid_observed * weights).sum(axis=1)
            ## Null hypothesis statistics
            mu_h0 = p_wrong * weights_sum
            sigma_h0 = np.sqrt((1 - p_wrong) * p_wrong * weights_square_sum)
            ## Standardized observed statistics
            zobs = np.divide((jiobs - mu_h0), sigma_h0, out=np.zeros_like(sigma_h0), where=sigma_h0 != 0)
            ## Observed p-values
            oiobs = norm.sf(abs(zobs), 0, 1)
            # Simulate under the null hypothesis
            sim_results = self.simulate_ji_matrix(p_wrong, weights, weights_sum, weights_square_sum, random_state)
            ji_matrix = sim_results["ji_matrix"]
            pvalue = self.compare_to_observed(jiobs, ji_matrix)
            to_add = (oiobs < self.rejection_threshold) & (zobs < mu_h0)
            # Filter instances based on p-values
            L_filtered = L_.iloc[to_add, :] if is_df else L_[to_add, :]
            y_filtered = y_[to_add]
            yL_filtered = yL_[to_add]
            # Update labeled and unlabeled datasets
            X_label = pd.concat([X_label, L_filtered]) if is_df else np.concatenate([X_label, L_filtered], axis=0)
            y_label = np.concatenate((y_label, y_filtered), axis=0)

            to_delete = indexes[to_add]
            if is_df:
                X_unlabel = X_unlabel.drop(index=X_unlabel.index[to_delete])
            else:
                X_unlabel = np.delete(X_unlabel, to_delete, axis=0)
                y_unlabel = np.delete(y_unlabel, to_delete)

            accuracy.append(self._base_estimator.score(L_filtered, yL_filtered))

            self._base_estimator = skclone(self.base_estimator)
            if self.htunning:
                grid_search = GridSearchCV(self._base_estimator, self.param_grid, scoring='accuracy', cv=5)
                X_retrain, X_reval, y_retrain, y_reval = train_test_split(X_label, y_label, test_size=0.3, random_state=random_state, stratify=y_label)
                grid_search.fit(X_reval, y_reval)
                best_params = grid_search.best_params_
                self._base_estimator.set_params(**best_params)
                self._base_estimator.fit(X_retrain, y_retrain, **kwargs)
            else:
                self._base_estimator.fit(X_label, y_label, **kwargs)

            if (iteration % self.view == 0) and self.messages:
                logger.info(f"Iteration {iteration} - {len(X_label)} labeled, {len(X_unlabel)} unlabeled")
                logger.info(f"Label distribution: {pd.Series(y_label).value_counts().to_dict()}")
                if self.htunning:
                    logger.info(f"Best parameters: {best_params}")
                if self.X_label_real is not None and len(self.X_label_real) > 0:
                    y_pred = self._base_estimator.predict(self.X_label_real)
                    report = classification_report(self.y_label_real, y_pred)
                    logger.info(f"Classification Report:\n{report}")

        self.accuracy_ = accuracy
        self.XU_ = L_filtered
        self.yU_ = yL_filtered
        self._base_estimator.fit(X_label, y_label, **kwargs)
        self.prior_probabilities_ = calculate_prior_probability(y_label)
        self.y_pseudolabel = y_
        self.p_wrong_ = p_wrong
        self.weights_ = weights
        self.iid_observed_ = iid_observed
        self.jiobs_ = jiobs
        self.ji_matrix_ = ji_matrix

        return self

    