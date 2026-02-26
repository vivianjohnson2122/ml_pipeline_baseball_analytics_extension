"""
feature_engineering.py — Feature Engineering

This module builds on the R project by adding:
  1. Exit velocity (launch_speed) 
  2. Spray angle — horizontal direction of contact
  3. Count state encoding — ball-strike context
  4. Pitch type grouping — fastball / breaking / offspeed
  5. Handedness matchup — same-hand platoon effect
  6. Interaction terms — bat speed × swing length, EV × launch angle
  7. SMOTE balancing with comparison to original distribution

Key design principle: All transformations are fit ONLY on training data,
then applied to test data — no leakage.
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRAIN_FILE, TEST_FILE, SMOTE_TRAIN_FILE, FEATURES_FILE,
    PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR,
    TARGET_COL, MODEL_FEATURE_COLS, COLS_TO_STANDARDIZE,
    SMOTE_K_NEIGHBORS, SMOTE_RANDOM_STATE, RANDOM_STATE
)

warnings.filterwarnings("ignore")

# ─── Pitch Type Grouping ──────────────────────────────────────────────────────

PITCH_TYPE_GROUPS = {
    # Fastballs
    "FF": "fastball",   # 4-seam
    "SI": "fastball",   # Sinker
    "FC": "fastball",   # Cutter
    "FS": "fastball",   # Split-finger
    # Breaking balls
    "SL": "breaking",   # Slider
    "CU": "breaking",   # Curveball
    "KC": "breaking",   # Knuckle-curve
    "SV": "breaking",   # Sweeper
    "ST": "breaking",   # Sweeper variant
    # Offspeed
    "CH": "offspeed",   # Changeup
    "EP": "offspeed",   # Eephus
    "KN": "offspeed",   # Knuckleball
}


def encode_pitch_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw statcast pitch type codes to three groups:
        fastball / breaking / offspeed 
    Unknown types default to fastball
    """
    df = df.copy()
    if 'pitch_type' not in df.columns:
        return df # don't do anything 
    else:
        df['pitch_group'] = df['pitch_type'].map(PITCH_TYPE_GROUPS).fillna('fastball')
    # one hot encode col
    df = pd.get_dummies(df, columns=['pitch_group'], dtype=int)
    return df


# ─── Count State Features ─────────────────────────────────────────────────────
def encode_count_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode ball-strike count as:
        - count_state: integer code (balls*10 + strikes)
        - ahead_in_count: 1 if balls > strikes (hitters count)
        - behind_in_count: 1 if strikes > balls (pitcher's count) 
        - two_strike: 1 if 2 strikes (more pressure)

    Reasoning: different situations can lead to different outputs. For example
    a fastball on a 3-0 count is diffrent to the hitter and the pitcher than a
    breaking ball on a 0-2 count. This feature captures pitch selection / 
    situational context
    """
    # n/a
    if "balls" not in df.columns or "strikes" not in df.columns:
        df["count_state"]      = 0
        df["ahead_in_count"]   = 0
        df["behind_in_count"]  = 0
        df["two_strike"]       = 0
        return df
    # encode features 
    df["count_state"]     = df["balls"] * 10 + df["strikes"]
    df["ahead_in_count"]  = (df["balls"] > df["strikes"]).astype(int)
    df["behind_in_count"] = (df["strikes"] > df["balls"]).astype(int)
    df["two_strike"]      = (df["strikes"] == 2).astype(int)
    return df


# ─── Handedness Matchup ───────────────────────────────────────────────────────
def encode_handedness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode pitcher-batter handedness matchup.

    In baseball, same-handed matchups (R vs R, L vs L) favor the pitcher;
    opposite-handed matchups favor the batter. This is a meaningful
    contextual feature for hit outcome prediction.
    """
    if "stand" not in df.columns or "p_throws" not in df.columns:
        # cols not there - return
        df["same_hand"] = 0
        return df
    df["same_hand"] = (df["stand"] == df["p_throws"]).astype(int)
    return df


# ─── Standardization ───────────────────────────────────────────────────────
class StatcastScaler:
    """
    Fit z-score standardization on training data. Save / Store mean/stdev for
    reproducibility so it can be applied to any split / test data 
    """

    def __init__(self, cols:list):
        self.cols = [c for c in cols]
        self.means_ = {}
        self.stds_ = {}
        self._fitted = False

    def fit(self, df:pd.DataFrame):
        """
        Get the mean and standard deviation for each col
        """
        for col in self.cols:
            if col in df.columns:
                # get the mean of the col
                self.means_[col] = df[col].mean()
                # get the stdev of the col 
                self.stds_[col] = df[col].std()
        self._fitted=True
        return self
    
    def transform(self, df:pd.DataFrame):
        """
        Apply the standardization after fitting for each col
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        df = df.copy()
        for col in self.means_.keys():
            z_col = f"{col}_z"
            df[z_col] = (df[col] - self.means_[col]) / self.stds_[col]
        return df
    
    def fit_transform(self, df:pd.DataFrame):
        return self.fit(df).transform(df)
    
    def to_dict(self) -> dict:
        return {"means": self.means_,
                "stds": self.stds_}
    
    @classmethod
    def from_dict(cls, d: dict, cols: list) -> "StatcastScaler":
        scaler = cls(cols)
        scaler.means_ = d["means"]
        scaler.stds_  = d["stds"]
        scaler._fitted = True
        return scaler
    

# ─── Interaction Terms ────────────────────────────────────────────────────────
def build_interaction_terms(df: pd.DataFrame):
    """
    Create interaction terms motivated by domain knowledge

    1. bat_speed * swing_length: product captures 'total swing power' in a way neither alone does.

    2. launch_speed * launch_angle: A 100 mph ball at 5° is a line drive single;
       the same speed at 30° is a home run. The interaction captures this.
    """

    if "bat_speed_z" in df.columns and "swing_length_z" in df.columns:
        df["bat_speed_x_swing_length"] = df["bat_speed_z"] * df["swing_length_z"]

    if "launch_speed_z" in df.columns and "launch_angle" in df.columns:
        df["launch_speed_x_launch_angle"] = df["launch_speed_z"] * df["launch_angle"]

    return df


# ─── Full Feature Engineering Function ───────────────────────────────────────
def engineer_features(df: pd.DataFrame,
                      scaler: StatcastScaler,
                      fit_scaler: bool=False) -> pd.DataFrame:
    """
    Apply all feature engineering steps to a data frame

    Args:
        df: input data frame 
        scaler: StatcastScaler instance
        fit_scaler: if true, fit scaler on this data - training only 

    Returns:
        DataFrame with all engineered features + target col 
    """
    df = df.copy()
    # 1. pitch type
    df = encode_pitch_type(df)
    # 2. count state 
    df = encode_count_state(df)
    # 3. handedness 
    df = encode_handedness(df)
    # 4. standarize continuous features 
    if fit_scaler:
        df = scaler.fit_transform(df)
        for col, mean in scaler.means_.items():
            std = scaler.stds_[col]
            print(f"{col}: mean={mean:.4f}, std={std:.4f}")
    else:
        df = scaler.transform(df)
    # 5. Interaction terms (after standardization)
    df = build_interaction_terms(df)
    return df 


def get_available_features(df: pd.DataFrame) -> list:
    """Return which of MODEL_FEATURE_COLS are actually present in df."""
    return [c for c in MODEL_FEATURE_COLS if c in df.columns]


# ─── Correlation Plot ─────────────────────────────────────────────────────────

def plot_correlation_matrix(df: pd.DataFrame, feature_cols: list):
    """
    Correlation heatmap for engineered features.
    Flags any pair with |r| > 0.8 as potential multicollinearity.
    """
    available = [c for c in feature_cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, ax=ax,
        cbar_kws={"label": "Pearson r"},
        annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Matrix\n(|r| > 0.8 = potential multicollinearity)",
                 fontsize=12, fontweight="bold")

    # Highlight high correlations
    for i in range(len(corr)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.8:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                             edgecolor="yellow", linewidth=2))

    plt.tight_layout()
    path = FIGURES_DIR / "02_correlation_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved {path.name}")


def plot_smote_comparison(y_original: pd.Series, y_smote: pd.Series):
    """Compare class distributions before and after SMOTE."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Class Balance: Before vs. After SMOTE", fontsize=13, fontweight="bold")
    labels = ["Single (1)", "Extra Base (2)", "Home Run (3)"]
    colors = ["#4C72B0", "#DD8452", "#C44E52"]

    for ax, y, title in zip(axes, [y_original, y_smote], ["Before SMOTE", "After SMOTE"]):
        counts = y.value_counts().sort_index()
        ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_title(title)
        ax.set_ylabel("Count")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 50, f"{v:,}", ha="center", fontsize=9)
        ax.set_ylim(0, max(counts.values) * 1.15)

    plt.tight_layout()
    path = FIGURES_DIR / "02_smote_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved {path.name}")


def plot_pitch_type_breakdown(df: pd.DataFrame):
    """
    Show feature distributions segmented by pitch type group.
    """
    # Check which indicator columns exist
    indicator_cols = [col for col in df.columns if col.startswith("pitch_group_")]
    if not indicator_cols or "launch_speed_z" not in df.columns:
        return

    # Reconstruct pitch_group
    def get_group(row):
        for col in indicator_cols:
            if row[col] == 1:
                return col.replace("pitch_group_", "")
        return "unknown"

    df["pitch_group_recon"] = df.apply(get_group, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Exit Velocity (standardized) Distribution by Pitch Type Group & Hit Outcome",
                 fontsize=12, fontweight="bold")
    palette = {1: "#4C72B0", 2: "#DD8452", 3: "#C44E52"}
    groups = ["fastball", "breaking", "offspeed"]

    for ax, group in zip(axes, groups):
        sub = df[df["pitch_group_recon"] == group]
        sns.kdeplot(
            data=sub, x="launch_speed_z", hue=TARGET_COL,
            palette=palette, ax=ax, fill=True, alpha=0.3, common_norm=False
        )
        ax.set_title(f"{group.title()} (n={len(sub):,})")
        ax.set_xlabel("Exit Velocity (zscore)")
        ax.set_ylabel("Density")
        ax.legend(title="Outcome", labels=["Single", "XBH", "HR"])

    plt.tight_layout()
    path = FIGURES_DIR / "02_pitch_type_breakdown.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved {path.name}")


# ─── Main Feature Engineering Function ───────────────────────────────────────
def run_feature_engineering() -> tuple:
    """
    Full feature engineering pipeline 
    Returns 
        (X_train, y_train,
        X_train_smote, y_train_smote,
        X_test, y_test,
        feature_cols)
    """
    print("\n" + "="*60)
    print("STEP 5: FEATURE ENGINEERING")
    print("="*60)

    # load splits 
    print(f"[LOAD] Reading train/test splits ...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    print(f"[LOAD] Train: {len(train_df)} | Test: {len(test_df)}")
    # initialize sclaer - fit to train only
    scaler = StatcastScaler(COLS_TO_STANDARDIZE)
    # engineer features 
    print(f"[FEATURES] Engineering training features ...")
    train_feat = engineer_features(train_df, scaler, fit_scaler=True)
    print("[FEATURES] Engineering test features (using train statistics)...")
    test_feat  = engineer_features(test_df, scaler, fit_scaler=False)
    # identify available features 
    feature_cols = get_available_features(train_feat)
    print(f"[FEATURES] Using {len(feature_cols)} features:")
    for f in feature_cols:
        print(f"    - {f}")
    # plot pitch types 
    print(f'[PLOT] Generating feature engineering plots ... ')
    plot_pitch_type_breakdown(train_feat)
    plot_correlation_matrix(train_feat, feature_cols)
    # get rid of not used cols 
    all_cols = MODEL_FEATURE_COLS + [TARGET_COL]
    final_eng_train_df = train_feat[all_cols]
    final_eng_test_df = test_feat[all_cols]
    # save whole df files 
    final_eng_train_df.to_csv(TRAIN_FILE, index=False)
    final_eng_test_df.to_csv(TEST_FILE, index=False)
    print(f"[SAVE] Original Imbalanced Engineered train/test saved")
    # separate x and target 
    X_train = final_eng_train_df[feature_cols].fillna(0)
    y_train = final_eng_train_df[TARGET_COL]
    X_test = final_eng_test_df[feature_cols].fillna(0)
    y_test = final_eng_test_df[TARGET_COL]

    print("\n" + "="*60)
    print("STEP 6: APPLYING SMOTE")
    print("="*60)

    # apply smote
    print(f"[FEATURES] Applying SMOTE to training data ...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print("[FEATURES] Original X_train shape:", X_train.shape)
    print("[FEATURES] SMOTE Resampled X_train shape:", X_train_smote.shape)
    # saving smote data
    smote_train_df = pd.DataFrame(X_train_smote, columns=X_train_smote.columns)
    smote_train_df[TARGET_COL] = y_train_smote
    smote_train_df.to_csv(SMOTE_TRAIN_FILE, index=False)
    print(f"[SAVE] SMOTE-balanced training data saved")
    print("[PLOTS] Generating SMOTE feature engineering visualizations...")
    plot_smote_comparison(y_train, pd.Series(y_train_smote))
    print("[COMPLETE] Feature engineering")

    return X_train, y_train, X_train_smote, y_train_smote, X_test, y_test, feature_cols