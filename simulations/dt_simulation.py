# --- Setup logger ---
import logging
import sys
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# --- Standard libraries ---
sys.path.append(os.path.abspath(".."))

# --- Data manipulation ---
import numpy as np
import pandas as pd

# --- Models ---
from sklearn.tree import DecisionTreeClassifier

# --- Evaluation metrics ---
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
    log_loss,
)

# --- Custom modules and data ---
from setred_package import setred_scratch, simulated_data
from sklearn.model_selection import train_test_split, GridSearchCV

# --- Simulation parameters ---
m_stred = 100
n = 17000
K = 5
p = 5
kcenters = 5
label_rate = 0.01  # Label rate for semi-supervised learning

logger.info("Starting Decision Tree + SETRED simulation.")
logger.info(f"Simulation config - Samples: {n}, Classes: {K}, Features: {p}, Label rate: {label_rate}")

score_setred = {}

std_values = [1,1.5,2,3,4,5]
graph_neighbors_value = [2,5,10,15,20]


for std in std_values:
    logger.info(f"Generating data with std = {std}")
    
    X_ori, y_ori, X, y, X_unlabel, y_unlabel, X_test, y_test = simulated_data.create_data(
        n=n,
        kcenters=kcenters,
        K=K,
        p=p,
        std=std,
        label_rate=label_rate,
        export_path=None
    )

    X_val = X[y != -1]
    y_val = y[y != -1]
    logger.info(f"Labeled samples: {len(y_val)} | Unlabeled samples: {np.sum(y == -1)}")

    logger.info("Performing hyperparameter tuning for Decision Tree using GridSearchCV")
    dt = DecisionTreeClassifier(random_state=42)
    param_grid = {
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(dt, param_grid, cv=5)
    grid_search.fit(X_val, y_val)
    best_params = grid_search.best_params_
    logger.info(f"Best Decision Tree params: {best_params}")

    base_estimator_dt = DecisionTreeClassifier(**best_params, random_state=42)
    base_estimator_dt.fit(X_val, y_val)
    
    # Score the base estimator on validation, test, and unlabeled data
    y_pred_val = base_estimator_dt.predict(X_val)
    score_base_estimator_val = accuracy_score(y_val, y_pred_val)
    y_pred = base_estimator_dt.predict(X_test)
    score_base_estimator_test = accuracy_score(y_test, y_pred)
    y_pred_unlabel = base_estimator_dt.predict(X_unlabel)
    score_base_estimator_unlabel = accuracy_score(y_unlabel, y_pred_unlabel)

    logger.info(f"Base DT Accuracy (Val): {score_base_estimator_val:.4f} | (Test): {score_base_estimator_test:.4f}")

    # --- SETRED Simulation ---
    logger.info("Running SETRED with Decision Tree base estimator")

    score_graph_neighbors = {}
    for graph_neighbors in graph_neighbors_value:
        logger.info(f"Graph neighbors: {graph_neighbors}")

        score_test = []
        score_unlabel = []

        for i in range(m_stred):
            logger.info(f"SETRED simulation run {i+1}/{m_stred}")
            ssl_clf_dt = setred_scratch.Setred_scratch(
                base_estimator=base_estimator_dt,
                graph_neighbors=graph_neighbors,
                max_iterations=40,
                htunning=True,
                param_grid={
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4, 5, 15]
                },
                X_test=X_test,
                y_test=y_test,
                y_real_label=y_unlabel,
                messages=False,
                view=10
            )
            ssl_clf_dt.fit(X, y)
            
            # Scores after SETRED training
            score_i = ssl_clf_dt.score(X_test, y_test) # Test            
            score_unlabel.append(ssl_clf_dt.score(X_unlabel, y_unlabel)) # unlabel


            logger.info(f"Run {i+1} - SETRED Accuracy: {score_i:.4f}")
            score_test.append(score_i)

        score_mean = np.mean(score_test)
        score_std = np.std(score_test)

        logger.info(f"Completed std = {std}, graph_neighbors {graph_neighbors} | Mean SETRED accuracy: {score_mean:.4f} ± {score_std:.4f}")

        score_graph_neighbors[f"graph_neighbor_{graph_neighbors}"] = {
        'graph_neighbors': graph_neighbors,
        'score_base_estimator_val': score_base_estimator_val,
        'score_base_estimator_unlabel': score_base_estimator_unlabel,
        'score_base_estimator_test': score_base_estimator_test,
        'setred_score_mean_test': score_mean,
        'setred_score_std_test': score_std,
        'setred_score_mean_unlabel': np.mean(score_unlabel),
        'setred_score_std_unlabel': np.std(score_unlabel),
        'score_list_test': score_test,
        'score_list_unlabel': score_unlabel
          }
    
    score_setred[f"std_{std}"] = score_graph_neighbors


# Helper function to convert non-serializable types
def convert(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Save to 'results.json'
with open("results/results.json", "w") as f:
    json.dump(score_setred, f, indent=4, default=convert)

logger.info("Results exported to results.json")

logger.info("Simulation complete.")
