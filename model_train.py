"""
model_train.py — Model Training

Trains three models and saves versioned artifacts:
  1. Multinomial Logistic Regression (baseline — replicates R project)
  2. Random Forest
  3. XGBoost

Key practices:
  - Stratified K-fold cross-validation for honest performance estimates
  - MLOps model registry along with associated metadata, parameters, and
    metrics for reproducibility.
  - Training curves plotted to diagnose overfitting
"""

import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from feature_engineering import *
from ingest import * 
import numpy as np
import pandas as pd
import matplotlib
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import mlflow.sklearn
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRAIN_FILE, TEST_FILE, SMOTE_TRAIN_FILE, MODELS_DIR, FIGURES_DIR, REPORTS_DIR,
    TARGET_COL, MODEL_FEATURE_COLS,
    LOGISTIC_PARAMS, RANDOM_FOREST_PARAMS, XGBOOST_PARAMS,
    CV_FOLDS, RANDOM_STATE,
)

mlflow.set_tracking_uri("file:./mlruns")

warnings.filterwarnings("ignore")

# ─── Multinomial Logistic Regression ───────────────────────────────────────────────────────────
def build_mlr(X,
              y,
              param_grid,
              dataset_name,
              using_smote: bool=False):
    print(f'[MULTINOMIAL LOGISTIC REGRESSION] Building Model ...')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    if not using_smote:
        mlr = LogisticRegression(
            multi_class='multinomial',
            class_weight='balanced', # DELETE FOR SMOTE DATA
            max_iter=2000,
            random_state=RANDOM_STATE
        )
    else:
        mlr = LogisticRegression(
            multi_class='multinomial',
            max_iter=2000,
            random_state=RANDOM_STATE
            )  
    rand_search = RandomizedSearchCV(
        estimator=mlr,
        param_distributions=param_grid,
        n_iter=20,
        scoring='f1_macro',
        cv=skf,
        verbose=2,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    print(f'[MULTINOMIAL LOGISTIC REGRESSION] Searching for hyperparameters, fiting ...')
    with mlflow.start_run(run_name="Multinomial Logistic Regression"):
        rand_search.fit(X,y)
        # log best params
        mlflow.set_tag("model_type", "Multinomial Logistic Regression")
        mlflow.set_tag("dataset_type", dataset_name)
        mlflow.log_params(rand_search.best_params_)
        mlflow.log_param("using_smote", using_smote)
        # log metric
        mlflow.log_metric("best_f1_macro", rand_search.best_score_)
        # log model 
        mlflow.sklearn.log_model(
            rand_search.best_estimator_,
            name=f"mlr_{dataset_name}"
        )
    return rand_search.best_estimator_


# ─── Random Forest ───────────────────────────────────────────────────────────
def build_random_forest(X, y, param_grid, dataset_name, using_smote:bool=False):
    print(f'[RANDOM FOREST] Building Model ...')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    if not using_smote:
        rf = RandomForestClassifier(class_weight='balanced', # CHANGE FOR SMOTE
                                    random_state=RANDOM_STATE,
                                    n_jobs=-1)
    else:
        rf = RandomForestClassifier(random_state=RANDOM_STATE,
                                    n_jobs=-1)

    print(f'[RANDOM FOREST] Searching for hyperparameters, fiting ...')
    # random search for hyperparameters 
    rand_search = RandomizedSearchCV(
        rf,
        param_distributions=param_grid,
        n_iter=10,
        scoring='f1_macro', # handles imbalance, avg of f1 for all classes
        cv=skf,
        verbose=2,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    with mlflow.start_run(run_name="Random Forest"):
        rand_search.fit(X,y)
        # log best params
        mlflow.set_tag("model_type", "Random Forest")
        mlflow.set_tag("dataset_type", dataset_name)
        mlflow.log_params(rand_search.best_params_)
        mlflow.log_param("using_smote", using_smote)
        # log metric
        mlflow.log_metric("best_f1_macro", rand_search.best_score_)
        # log model 
        mlflow.sklearn.log_model(
            rand_search.best_estimator_,
            name=f"rf_{dataset_name}"
        )
    return rand_search.best_estimator_

# ─── XGBoost ───────────────────────────────────────────────────────────
def build_XGBoost(X, y, param_grid, dataset_name):
    # fixing zero index issue 
    label_map = {1: 0, 2: 1, 3: 2}
    y_xgb = y.map(label_map)
    print(f'[XGBOOST] Building Model ...')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y_xgb)),
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    print(f'[XGBOOST] Searching for hyperparameters, fiting ...')
    # random search for hyperparameters 
    rand_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=20,
        scoring='f1_macro', # handles imbalance, avg of f1 for all classes
        cv=skf,
        verbose=2,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    with mlflow.start_run(run_name=f"XGBoost_{dataset_name}"):
        rand_search.fit(X,y_xgb)
        # log best params
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("dataset_type", dataset_name)
        mlflow.log_params(rand_search.best_params_)
        # log metric
        mlflow.log_metric("best_f1_macro", rand_search.best_score_)
        # log model 
        mlflow.sklearn.log_model(
            rand_search.best_estimator_,
            name=f"xgboost_{dataset_name}"
        )
    return rand_search.best_estimator_


def plot_feature_importance(model,
                            feature_cols: list,
                            data_type,
                            top_n=20,
                            figsize=(10,6)):
    """
    Plot feature importance from a trained Random Forest Model

    Args:
        model: trained random forest classifier
        feature_names: list of feature names 
        top_n: number of top features to plot
        figsize: figure size 
    """

    importances = model.feature_importances_
    indicies = np.argsort(importances)[::-1] # sort descending order 

    # select top n features
    top_indicies = indicies[:top_n]
    top_features = [feature_cols[i] for i in top_indicies]
    top_importances = importances[top_indicies]

    # plot figure
    plt.figure(figsize=figsize)
    plt.barh(range(len(top_features)-1, -1, -1), top_importances, color='skyblue')
    plt.yticks(range(len(top_features)-1, -1, -1), top_features)
    plt.xlabel('Feature Importance')
    plt.title('Top Feature Importances - Random Forest')
    plt.tight_layout()
    path = FIGURES_DIR / f"03_{data_type}_feature_importance_rf.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved {path.name}")
    

from mlflow.entities import ViewType
import mlflow
import matplotlib.pyplot as plt
import pandas as pd

def comparing_metrics_across_runs(tracking_uri="file:./mlruns",
                                  top_n=3):
    """
    Compare f1_macro across all models over time,
    color-coding SMOTE vs no-SMOTE runs.
    """
    # set MLflow to local folder 
    mlflow.set_tracking_uri(tracking_uri)

    # list all experiments
    experiments = mlflow.search_experiments(view_type=ViewType.ALL)

    # collect all runs
    all_runs = []
    for exp in experiments:
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        if not runs.empty:
            runs['experiment_name'] = exp.name
            all_runs.append(runs)

    if not all_runs:
        print('No runs found in local MLflow Folder')
        return

    df = pd.concat(all_runs, ignore_index=True)

    # convert timestamp to datetime 
    df['start_time'] = pd.to_datetime(df['start_time'], unit='ms')

    # ensure 'using_smote' exists (fill missing as False)
    if 'params.using_smote' not in df.columns:
        df['params.using_smote'] = 'False'
    
    # create a label combining model name + SMOTE info
    df['model_label'] = df['tags.mlflow.runName'] + " | SMOTE: " + df['params.using_smote']

    # 7define colors: SMOTE True = orange, False = blue
    color_map = {'True': 'orange', 'False': 'blue'}

    # plot F1_macro over time
    plt.figure(figsize=(12,6))
    for label, grp in df.groupby('model_label'):
        grp_sorted = grp.sort_values('start_time').tail(top_n)  # latest top_n runs
        smote_status = label.split("SMOTE: ")[1]  # extract True/False
        plt.plot(grp_sorted['start_time'],
                 grp_sorted['metrics.best_f1_macro'],
                 marker='o',
                 label=label,
                 color=color_map.get(smote_status, 'gray'))

    plt.xlabel('Run Start Time')
    plt.ylabel('F1 Macro')
    plt.title('F1 Macro Across Models Over Time (SMOTE vs No-SMOTE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 9️⃣ Save figure
    path = FIGURES_DIR / "03_comparing_mlflow_metric_runs.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved {path.name}")


def run_model_training(X_train,
                       y_train,
                       X_train_smote,
                       y_train_smote,
                       X_test,
                       y_test,
                       feature_cols):
    """
    Run all model training
    """
    print("\n" + "="*60)
    print("STEP 7: MODEL BUILDING - ORIGINAL MODELS")
    print("="*60)

    # mlr - no smote 
    best_orig_mlr = build_mlr(X=X_train,
                              y=y_train,
                              param_grid=LOGISTIC_PARAMS,
                              dataset_name='original',
                              using_smote=False) 
    
    # rf - no smote 
    best_orig_rf = build_random_forest(X=X_train,
                                       y=y_train,
                                       param_grid=RANDOM_FOREST_PARAMS,
                                       using_smote=False,
                                       dataset_name='original')
    
    # xgb - no smote 
    best_orig_xgb = build_XGBoost(X=X_train,
                                  y=y_train,
                                  dataset_name='original',
                                  param_grid=XGBOOST_PARAMS)
    
    print(f'[PLOT] Plotting Feature Importance ...')
    plot_feature_importance(model=best_orig_rf,
                            feature_cols=feature_cols,
                            data_type='original')

    print("\n" + "="*60)
    print("STEP 8: MODEL BUILDING - SMOTE MODELS")
    print("="*60)

     # mlr - smote 
    best_smote_mlr = build_mlr(X=X_train_smote,
                              y=y_train_smote,
                              param_grid=LOGISTIC_PARAMS,
                              dataset_name='smote',
                              using_smote=True) 
    
    # rf - smote 
    best_smote_rf = build_random_forest(X=X_train_smote,
                                       y=y_train_smote,
                                       param_grid=RANDOM_FOREST_PARAMS,
                                       dataset_name='smote',
                                       using_smote=True)
    
    # xgb - smote 
    best_smote_xgb = build_XGBoost(X=X_train_smote,
                                  y=y_train_smote,
                                  dataset_name='smote',
                                  param_grid=XGBOOST_PARAMS)
    
    print(f'[PLOT] Plotting Feature Importance ...')
    plot_feature_importance(best_smote_rf,
                            feature_cols=feature_cols,
                            data_type='smote')
    
    print(f'[PLOT] Extracting MLFlow Recent Runs and Plotting ... ')
    comparing_metrics_across_runs()
    