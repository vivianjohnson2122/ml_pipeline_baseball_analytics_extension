"""
pipeline.py — End-to-End Pipeline Runner

Runs the full MLB hit outcome prediction pipeline in sequence:
  1. Data ingestion & validation
  2. Feature engineering & SMOTE balancing
  3. Model training (logistic, RF, XGBoost, LightGBM, ensemble)
  4. Evaluation (confusion matrices, ROC, calibration)
  5. SHAP explainability

Usage:
    python src/pipeline.py                    # Full pipeline
    python src/pipeline.py --steps 1,2,3      # Specific steps only
    python src/pipeline.py --skip-explain     # Skip SHAP (slower)
"""

import sys
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from ingest import *
from config import *
from model_train import *
from feature_engineering import *

sys.path.insert(0, str(Path(__file__).parent.parent))

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          MLB Hit Outcome Prediction Pipeline                 ║
║          Statcast 2024 · Python ML                           ║
╚══════════════════════════════════════════════════════════════╝
    """)

if __name__=="__main__":
    print_banner()
    train, test = run_ingestion()
    X_tr, y_tr, X_sm, y_sm, X_te, y_te, feats = run_feature_engineering()
    run_model_training(X_tr, y_tr, X_sm, y_sm, X_te, y_te, feats)