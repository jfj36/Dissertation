

import numpy as np 
import pandas as pd 
from scipy.stats import norm 
# Base classes
from sklearn.base import BaseEstimator, MetaEstimatorMixin, ClassifierMixin
# The clone function creates a new copy of an estimator with the same parameters, but without any trained data, without any fitted attributes. 
from sklearn.base import clone as skclone 
from sklearn.base import is_classifier
from sklearn.ensemble._base import _set_random_states

from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.semi_supervised import SelfTrainingClassifier

# Utils
from sklearn.utils import check_random_state, resample
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.metaestimators import available_if

# Selection classifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    auc,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    log_loss)

# SSlearn library
from sslearn.utils import calculate_prior_probability, check_classifier
from sslearn.model_selection import artificial_ssl_dataset
from sslearn.wrapper import Setred

class Setred_scratch(BaseEstimator, MetaEstimatorMixin):
    def __init__( self,
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
        X_label_real =None,
        y_label_real =None,
        y_unlabel=None,
        messages=True,
        view=100
    ):
    
        """
        Create a SETRED classifier.
        It is a self-training algorithm that uses a rejection mechanism to avoid adding noisy samples to the training set.
        
        Parameters
        ----------
        base_estimator : ClassifierMixin, optional
            An estimator object implementing fit and predict_proba, by default KNeighborsClassifier(n_neighbors=3)
        max_iterations : int, optional
            Maximum number of iterations allowed. Should be greater than or equal to 0., by default 40
        distance : str, optional
            The distance metric to use for the graph.
            The default metric is euclidean, and with p=2 is equivalent to the standard Euclidean metric.
            For a list of available metrics, see the documentation of DistanceMetric and the metrics listed in sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
            Note that the `cosine` metric uses cosine_distances., by default `euclidean`
        poolsize : float, optional
            Max number of unlabel instances candidates to pseudolabel, by default 0.25
        rejection_threshold : float, optional
            significance level, by default 0.05
        graph_neighbors : int, optional
            Number of neighbors for each sample., by default 1
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None
        n_jobs : int, optional
            The number of parallel jobs to run for neighbors search. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors, by default None
        """
        self.base_estimator = base_estimator
        self.max_iterations = max_iterations
        self.poolsize  = poolsize
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
        """Check and divide dataset between labeled and unlabeled data.

        Parameters
        ----------
        X : ndarray or DataFrame of shape (n_samples, n_features)
            Features matrix.
        y : ndarray of shape (n_samples,)
            Target vector.

        Returns
        -------
        X_label : ndarray or DataFrame of shape (n_label, n_features)
            Labeled features matrix.
        y_label : ndarray or Serie of shape (n_label,)
            Labeled target vector.
        X_unlabel : ndarray or Serie DataFrame of shape (n_unlabel, n_features)
            Unlabeled features matrix.
        """

        is_df = False
        if isinstance(X, pd.DataFrame):
            is_df = True
            columns = X.columns

        X = check_array(X)
        y = check_array(y, ensure_2d=False, dtype=y.dtype.type)
        
        X_label = X[y != y.dtype.type(-1)]
        y_label = y[y != y.dtype.type(-1)]
        X_unlabel = X[y == y.dtype.type(-1)]

        X_label, y_label = check_X_y(X_label, y_label)

        if is_df:
            X_label = pd.DataFrame(X_label, columns=columns)
            X_unlabel = pd.DataFrame(X_unlabel, columns=columns)

        return X_label, y_label, X_unlabel

    def __create_neighborhood(self,X):
        """Create a neighborhood graph using the kneighbors_graph function."""
        return kneighbors_graph(X, n_neighbors=self.graph_neighbors, mode="distance", metric=self.distance, n_jobs=self.n_jobs).toarray()


    # Simulation functions
    def simulate_ji_matrix(self,
                            p_wrong,
                            weights,
                            weights_sum,
                            weights_square_sum,
                            random_state
                             ):
        n_instances, n_neighbors = weights.shape

        # Precompute simulation probabilities
        p_matrix = np.repeat(p_wrong, n_neighbors).reshape(weights.shape)

        # Matrix to store all simulated ji values
        ji_matrix = np.zeros((n_instances,self.n_simulations))
        zi_matrix = np.zeros((n_instances,self.n_simulations))
        oi_matrix = np.zeros((n_instances,self.n_simulations))

         # Expected value
        mu_h0 = p_wrong * weights_sum
        sigma_h0 = np.sqrt((1 - p_wrong) * p_wrong * weights_square_sum)

        for s in range(self.n_simulations):
            # Simulate binary decisions
            iid_random = random_state.binomial(1, p_matrix)            
            # Simulate test statistic
            ji = (iid_random * weights).sum(axis=1)
            ji_matrix[:, s] = ji           
            # Calculate the z-score
            z_score = np.divide((ji - mu_h0), sigma_h0, out=np.zeros_like(sigma_h0), where=sigma_h0 != 0)
            zi_matrix[:, s] = z_score
            # Calculate the p-value using the survival function
            #oi = norm.sf(abs(z_score), mu_h0, sigma_h0)
            oi = norm.sf(abs(z_score), 0, 1)
            oi_matrix[:, s] = oi
            #to_add = (oi < 0.05) & (z_score < mu_h0)  # or use self.rejection_threshold

            # If you're using to_add for filtering, handle it externally

        return {"ji_matrix": ji_matrix, 
                "zi_matrix": zi_matrix,
                "oi_matrix": oi_matrix                
                }

    def compare_to_observed(self, jiobs, ji_matrix):
    # Returns a count or proportion of times simulated ji > observed
        return 1 - np.mean(jiobs[:,None] < ji_matrix, axis=1)  # count
    

    # Fit function

    def fit(self, X, y, **kwargs):
        """ 
        Build a Setred classifier from the training set (X,y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. 
        y : array-like of shape (n_samples,)
            The target values (class labels), -1 if unlabeled.
        **kwargs : dict, optional

        Returns
        -------
        self : Setred 
            Fitted estimator

        """
        random_state = check_random_state(self.random_state)
        y_unlabel = self.y_unlabel
        X_label, y_label, X_unlabel = self.get_dataset(X, y)

        is_df = isinstance(X_label, pd.DataFrame)

        self.classes_ = np.unique(y_label)

        each_iteration_candidates = X_label.shape[0]
        #print(f"Each iteration will use {each_iteration_candidates} candidates to pseudolabel")

        pool = int(len(X_unlabel) * self.poolsize)
        self._base_estimator = skclone(self.base_estimator)        

        self._base_estimator.fit(X_label, y_label, **kwargs)

        # Should probabilities change every iteration or may it keep with the first L?
        y_probabilities = calculate_prior_probability(y_label) 

        sort_idx = np.argsort(list(y_probabilities.keys()))

        if X_unlabel.shape[0] == 0:
            return self
        iteration = 0
        accuracy = []
        for _ in range(self.max_iterations):
            iteration += 1
            # Resample unlabel candidates            
            U_, yU_ = resample(X_unlabel,y_unlabel, replace = False, n_samples = pool, random_state = random_state)


            if is_df:
                U_ = pd.DataFrame(U_, columns = X_label.columns)
            
            # Predictions
            raw_predictions = self._base_estimator.predict_proba(U_)
            predictions = np.max(raw_predictions, axis = 1)
            class_predicted = np.argmax(raw_predictions, axis = 1)
            # Unless a better understandinig is given, only the size of L will be used as maximal size of the candidate set. 
            indexes = predictions.argsort()[-each_iteration_candidates:]
            #indexes =  np.where(predictions > 0.90)[0]

            # L_ is a set with the most confident predictions according to the classifier
            if is_df:
                L_ = U_.iloc[indexes]
            else:
                L_ = U_[indexes]
                yL_ = yU_[indexes]

            # Vector with the predictions
            y_ = np.array(
                list(
                    map(
                        lambda x : self._base_estimator.classes_[x],
                        class_predicted[indexes],
                    )
                )
            )            

            if is_df:
                pre_L = pd.concat([X_label, L_])
                pre_yL = pd.concat([y_label, pd.Series(y_)])
            else:
                pre_L = np.concatenate((X_label, L_), axis = 0)
                pre_yL = np.concatenate((y_label, y_), axis = 0)

            
            weights = self.__create_neighborhood(pre_L)            
            iid_observed = (pre_yL[:, None] != pre_yL[None, :]).astype(int)
                       
            # Keep only weights for L_ the set that contains the most confident predictions according to the classifier
            # And keep the connections with the labelled instances
            weights = weights[-L_.shape[0]:, :] #: X_label.shape[0]]
            iid_observed = iid_observed[-L_.shape[0]:, :] # :X_label.shape[0]]
            

            idx = np.searchsorted(np.array(list(y_probabilities.keys())), y_, sorter = sort_idx )
            p_wrong = 1 - np.asarray(np.array(list(y_probabilities.values())))[sort_idx][idx] # idx is a vector with L_.shape[0] values
            # Must weights be the inverse of distance?
            # According to the paper about cut edge statistic the weights are 1/(1+dij)
            weights[weights != 0] += 1 # I have added this line to follow the paper
            weights = np.divide(1, weights, out=np.zeros_like(weights), where= weights != 0 )

            weights_sum = weights.sum(axis=1)
            weights_square_sum = (weights**2).sum(axis=1)  

            # Hypothesis testing    
            jiobs = (iid_observed*weights).sum(axis=1) # jiobs is the observed value of the test statistic
            
            # Expected value
            mu_h0 = p_wrong * weights_sum
            sigma_h0 = np.sqrt((1 - p_wrong) * p_wrong * weights_square_sum)

            # Z observed
            zobs = np.divide((jiobs - mu_h0), sigma_h0, out=np.zeros_like(sigma_h0), where=sigma_h0 != 0)
            # p-value observed
            oiobs = norm.sf(abs(zobs), 0, 1) # p-value observed, using the survival function

            # Simulation 
            sim_results = self.simulate_ji_matrix(
                p_wrong,
                weights,
                weights_sum,
                weights_square_sum,
                random_state
            )
            ji_matrix = sim_results["ji_matrix"]
            zi_matrix = sim_results["zi_matrix"]
            oi_matrix = sim_results["oi_matrix"]

            pvalue = self.compare_to_observed(jiobs, ji_matrix)
            #to_add = (pvalue < self.rejection_threshold) #& (zi_matrix[:,1] < mu_h0) # pvalue < alpha and Z_score < mu_h0
            to_add = (oiobs < self.rejection_threshold )& (zobs < mu_h0) # pvalue < alpha and Z_score < mu_h0

            #print(f"Number of instances to add: {to_add.sum()}")
            if is_df:
                L_filtered = L_.iloc[to_add, :]
            else:
                L_filtered = L_[to_add, :]
                yL_filtered = yL_[to_add]
            
            y_filtered = y_[to_add]
            

            if is_df:
                X_label = pd.concat([X_label, L_filtered])
            else: 
                X_label = np.concatenate([X_label, L_filtered], axis = 0)
            
            y_label = np.concatenate((y_label, y_filtered), axis = 0)

            # Remove the instances from the unlabeled set

            to_delete = indexes[to_add]

            if is_df:
                X_unlabel = X_unlabel.drop(index=X_unlabel.index[to_delete])
            else:
                X_unlabel = np.delete(X_unlabel, to_delete, axis=0)
                y_unlabel = np.delete(y_unlabel, to_delete)

            #Append accuracy
            accuracy.append(self._base_estimator.score(L_filtered, yL_filtered))
            #print(f"Iteration {iteration} - Accuracy: {accuracy[-1]:.4f}")
            
            #print(f"Iteration {iteration}: Report of the base estimator: {classification_report(yL_filtered, self._base_estimator.predict(L_filtered))}")

            # Why the classifier was not retrained in each iteration?            
            self._base_estimator = skclone(self.base_estimator)
            #print(f"Number of labeled instances for retraining: {len(y_label)}")
            if self.htunning:
                param_grid = self.param_grid
                grid_search = GridSearchCV(self._base_estimator, param_grid,scoring='accuracy', cv=5)
                # Train validation split
                X_retrain, X_reval, y_retrain, y_reval = train_test_split(X_label, y_label, 
                                                                          test_size=0.3, 
                                                                          random_state=random_state,
                                                                          stratify=y_label)                
                grid_search.fit(X_reval, y_reval)
                # Best parameters
                best_params = grid_search.best_params_                 
                self._base_estimator.set_params(**best_params)  # I, Juan Felipe, have added this line to set the best parameters found by GridSearchCV.
                self._base_estimator.fit(X_retrain, y_retrain, **kwargs)  # I, Juan Felipe, have added this line to retrain the base estimator with the best parameters.
            else:
                self._base_estimator.fit(X_label, y_label, **kwargs)    # I, Juan Felipe, have added this line to retrain the base estimator in each iteration.
            
            # Simulation checkings
            if (iteration % self.view  == 0 )and (self.messages):
                print(f"Iteration {iteration} - {len(X_label)} labeled instances, {len(X_unlabel)} unlabeled instances left")
                print("Distribution of labels in the labeled set:")
                print(pd.Series(y_label).value_counts())
                if self.htunning:
                    print(f"Best parameters found: {best_params}")
                if (len(self.X_label_real) > 0):
                    y_pred = self._base_estimator.predict(self.X_label_real)
                    # Generate the classification report
                    report = classification_report(self.y_label_real, y_pred)           
                    print(report)
        
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
        self.zi_matrix_ = zi_matrix
        self.oi_matrix_ = oi_matrix
        self.zobs_ = zobs
        self.oiobs_ = oiobs

                
        return self
    
    def predict(self, X, **kwargs):
        """ 
        Predict class value for X
        For a classification model, the predicted class for each sample in X is returned. 
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. 
        Returns
        -------
        y : array-like of shape (n_samples,)
        The predicted classes
        """
        return self._base_estimator.predict(X, **kwargs)
    
    def predict_proba(self, X, **kwargs):
        """
        Predict class probabilities of the input samples X.
        The predicted class probability depends on the ensemble estimator. 
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray of shape (n_samples, n_classes) or list of n_outputs such arrays if n_outputs > 1
            The class probabilities of the input samples.The predicted classes 
                    
        """

        return self._base_estimator.predict_proba(X, **kwargs)
    
    def score(self, X, y, sample_weight=None):
        """ 
        Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        
        Returns
        -------
        score : float 
            Mean Accuracy of ''self.predict(X)'' wrt ``y``.
        """
        return accuracy_score(y, self.predict(X),sample_weight=sample_weight)