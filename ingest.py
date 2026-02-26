"""
ingest.py - Data Acquisition and Validation

This file handles:
    1. Pulling statcast data via pybaseball package
    2. Strict schema validation using pydantic
    3. Outlier detection and flagging
    4. Reproducible train/test splitting 
    5. Data quality reporting

Run Standalone: 
    python src/01_ingest.py
"""

import sys
import json
import warnings
from pathlib import Path
from typing import Optional
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer

from config import (
    RAW_DATA_FILE, TRAIN_FILE, TEST_FILE, REPORTS_DIR, FIGURES_DIR, PROCESSED_DIR,
    SEASON_START, SEASON_END, HIT_OUTCOME_MAP, TARGET_COL,
    TEST_SIZE, RANDOM_STATE, RAW_FEATURE_COLS
)


def validate_schema(df: pd.DataFrame) -> dict:
    """
    Validate the raw statcast DataFrame for expected columns, dtypes,
    value ranges, and missingness. Return a report dict

    Warn if anything unexpected is found 
    """

    report = {
        "timestamp": datetime.now().isoformat(),
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "issues": [],
        "warnings": [],
        "passed": True
    }
    
    # ----- Expected Columns ------
    expected = set(RAW_FEATURE_COLS + ["events", "player_name", "game_date", "batter", "pitcher"])
    missing_cols = expected - set(df.columns)

    if missing_cols:
        report["issues"].append(f"Missing expected columns: {missing_cols}")
        report

    # ----- Value Range Checks ------
    range_checks = {
        "launch_angle": (-100,100),
        "launch_speed": (0, 200),
        "bat_speed": (0, 110),
        "swing_length": (0, 20),
        "release_speed": (40,115)
    }
    for col, (low, high) in range_checks.items():
        if col not in df.columns:
            continue
        # coerce to numeric - invalid parsing becomes NaN
        series = pd.to_numeric(df[col], errors='coerce')
        mask = ((df[col] < low) | (df[col] > high)) & series.notna() # ignore missing vals
        n_out = mask.sum()
        if n_out > 0:
            pct = 100 * n_out / len(df)
            report["warnings"].append(
                f"{col}: {n_out} values ({pct:.1f}%) outside expected range [{low}, {high}]"
            )
    
    # --- Missingness report ---
    miss = df[RAW_FEATURE_COLS].isnull().mean() * 100
    high_miss = miss[miss > 20]
    if len(high_miss) > 0:
        for col, pct in high_miss.items():
            report["warnings"].append(f"{col}: {pct:.1f}% missing values")

    # --- Class distribution ---
    if "events" in df.columns:
        hits_df = df[df["events"].isin(HIT_OUTCOME_MAP.keys())]
        class_counts = hits_df["events"].value_counts().to_dict()
        report["class_distribution_raw"] = class_counts

    return report


def load_statcast_data(filepath: Optional[Path] = None):
    """
    Load statcast data. If filepath exists, loads from csv. 
    Otherwise attempts to pull via pybaseball and saves to local csv
    """

    if filepath and Path(filepath).exists():
        print(f'[INGEST] Loading from {filepath}')
        return pd.read_csv(filepath)
    
    # load from pybaseball 
    try:
        from pybaseball import statcast
        print(f"[INGEST] Pulling Statcast {SEASON_START} -> {SEASON_END}")
        df = statcast(start_dt=SEASON_START,
                      end_dt=SEASON_END)
        df.to_csv(filepath, index=False)
        print(f"[INGEST] Saved {len(df)} rows to {filepath}")
        return df
    except Exception as e:
        print(f"[INGEST] Pull Failed ({e}) ")
        return None


def initial_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Initial data cleaning 
        1. Filter only used columns 
        2. Filter only hits 
        3. Impute missing values 
        4. K nearest neighbors and remove bunts 
    """
    # -------Filter Useful Cols -------------------
    print(f'[CLEANING] Filtering useful features and hits ...')
    SELECTED_COLS = RAW_FEATURE_COLS + ['events'] 
    df = df[SELECTED_COLS].copy()

    # -------Filter Only Hits -------------------
    # assign target variable hit_outcome
    df['hit_outcome'] = df['events'].map(HIT_OUTCOME_MAP).fillna(0).astype(int)
    df = df[df['hit_outcome'] != 0] # filter rows where hit_outcome is not 0
    df = df.drop(columns=['events'])

    # ------- KNN Impute Missing Vals of Numeric Cols -------------------
    print(f'[CLEANING] Using KNN to impute missing numeric data ...')
    numeric_cols_with_nas = ['release_speed',
                             'launch_angle',
                             'launch_speed',
                             'bat_speed',
                             'swing_length',
                             'estimated_woba_using_speedangle']
    imputer = KNNImputer(n_neighbors=3)
    imputed_data_array = imputer.fit_transform(df[numeric_cols_with_nas])
    # convert back to original df with col names and other cols 
    df_imputed_cols = pd.DataFrame(imputed_data_array,
                                   columns=numeric_cols_with_nas,
                                   index=df.index)
    df[numeric_cols_with_nas] = df_imputed_cols

    # ------- Detect and Remove Outliers -------------------
    print(f'[CLEANING] Removing outliers ...')
    outliers = detect_outliers(df,
                               cols=numeric_cols_with_nas,
                               k=4.0)
    # filter out outliers 
    df['outliers'] = outliers
    df = df[df['outliers'] == False]
    df = df.drop(columns=['outliers'])

    return df


def kmeans_analysis_bunt_cluster(df: pd.DataFrame):
    """
    K Means Cluster Analysis looking at scatterplot of standardized bat speed 
    and standardized swing length.

    Results are two figures
        1. scatter.png
        2. cluster_scatter.png

    The groups don't show significant groupings in the data- the bunts and check
    swings that were present seem to have been successfully removed while dealing 
    with outliers. If the parameters were changed on the outliers, it might signify 
    needed to remove a small cluster of check swings / bunts  
    """ 
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances

    # ------- Standardizing bat speed, swing length (temp) ---------------

    numeric_cols = ['bat_speed', 'swing_length']
    available = [c for c in numeric_cols if c in df.columns]
    scaler = StandardScaler()
    z = scaler.fit_transform(df[available])
    # turn into df 
    z_df = pd.DataFrame(z, columns=[f"z{c}" for c in available])

    # ------- Histogram of standardized variables ---------------
    plt.figure(figsize=(6, 6))
    plt.scatter(z_df.iloc[:, 0], z_df.iloc[:, 1], alpha=0.6)
    plt.xlabel(z_df.columns[0])
    plt.ylabel(z_df.columns[1])
    plt.title("Standardized Bat Speed vs Swing Length")
    plt.savefig(FIGURES_DIR / "scatter.png")
    print(f"[PLOT] Saved scatter.png")
    plt.close()

    # ------- K Means Clustering  ---------------
    k = 3
    kmeans = KMeans(n_clusters=k,
                    random_state=RANDOM_STATE,
                    n_init=10)
    clusters = kmeans.fit_predict(z)
    z_df["cluster"] = clusters

    # ------- Cluster Plot  ---------------
    plt.figure(figsize=(6, 6))
    for c in range(k):
        subset = z_df[z_df["cluster"] == c]
        plt.scatter(
            subset.iloc[:, 0],
            subset.iloc[:, 1],
            label=f"Cluster {c}",
            alpha=0.7
        )
    centers = kmeans.cluster_centers_
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        s=200,
        marker="X",
        label="Centroids"
    )
    plt.xlabel(z_df.columns[0])
    plt.ylabel(z_df.columns[1])
    plt.title("K-Means Clusters (Standardized)")
    plt.legend()
    plt.savefig(FIGURES_DIR / "cluster_scatter.png")
    print(f"[PLOT] Saved cluster_scatter.png")
    plt.close()


def detect_outliers(df: pd.DataFrame,
                    cols: list,
                    k: float=3.0) -> pd.Series:
    """
    Flag rows as outliers if any numeric column is beyone k*IQR from the median. 
    Uses IQR which is robust to skew rather than zscores 

    Return a boolean series, True = row is an outlier in one of the cols
    """
    outliers = pd.Series(False, index=df.index)
    for col in cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1 
        lower, upper = q1 - k * iqr, q3 + k * iqr
        col_outliers = (df[col] < lower) | (df[col] > upper)
        outliers = outliers | col_outliers
    
    return outliers


def plot_class_distributions(df: pd.DataFrame):
    """
    Bar chart of hit outcome class distributions

    The purpose of this is to show the class imbalance between the 
    different hit types. This is to be taken into account during model 
    development 

    Return 
        - 01_class_distributions.png
    """
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    fig.suptitle("Hit Outcome Class Distribution",
                 fontsize=15,
                 fontweight="bold")
    
    counts = df[TARGET_COL].value_counts().sort_index()
    labels = ["Single\n(1)", "Extra Base Hit\n(2)", "Home Run\n(3)"]
    colors = ["#4C72B0", "#DD8452", "#C44E52"]
    # Raw counts
    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Raw Counts")
    axes[0].set_ylabel("Number of observations")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 100, f"{v:,}", ha="center", fontsize=10)

    # Percentages
    pcts = 100 * counts / counts.sum()
    axes[1].bar(labels, pcts.values, color=colors, edgecolor="white", linewidth=1.5)
    axes[1].set_title("Class Proportions")
    axes[1].set_ylabel("Percentage (%)")
    for i, v in enumerate(pcts.values):
        axes[1].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10)

    plt.tight_layout()
    path = FIGURES_DIR / "01_class_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved {path.name}")


def run_ingestion() -> pd.DataFrame:
    """
    Full ingestion pipeline. Returns the processed split-ready DataFrame
    """

    print("\n" + "="*60)
    print("STEP 1: DATA INGESTION & VALIDATION")
    print("="*60)

    # Load Data 
    filepath = RAW_DATA_FILE
    df_raw = load_statcast_data(filepath=filepath)
    print(f"[INGEST] Raw data: {len(df_raw)} rows x {len(df_raw.columns)} cols")

    # Validate schema
    print("\n  [validate] Running schema validation...")
    report = validate_schema(df_raw)
    print(report)

    print("\n" + "="*60)
    print("STEP 2: INITIAL DATA CLEANING")
    print("="*60)

    # Initial Cleaning
    df = initial_cleaning(df_raw)

    print("\n" + "="*60)
    print("STEP 3: GENERATING EDA PLOTS")
    print("="*60)
    
    # EDA Distribution Plot
    plot_class_distributions(df)

    # K Means Cluster Plot
    kmeans_analysis_bunt_cluster(df)

    print("\n" + "="*60)
    print("STEP 4: TRAIN / TEST SPLIT")
    print("="*60)

    # Splitting the data 
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL]
    )
    print(f"[SPLIT] Train: {len(train_df)} | Test: {len(test_df)}")
    # Saving the splits
    df.to_csv(RAW_DATA_FILE.parent.parent / "processed" / "full_clean.csv", index=False)
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)
    print(f"[SAVED] Data saved to {PROCESSED_DIR}")

    print(f"[COMPLETE] Ingestion Complete")
    return train_df, test_df


if __name__ == "__main__":
    train, test = run_ingestion()