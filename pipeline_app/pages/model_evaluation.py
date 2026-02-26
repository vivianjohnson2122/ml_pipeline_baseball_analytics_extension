
import pickle
from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
MLRUNS_PATH = Path(__file__).parent.parent.parent / "mlruns"
EXPERIMENT_ID = "0"
PRIMARY_METRIC = "best_f1_macro"
# ─────────────────────────────────────────────────────────────────────────────


st.title("Model Evaluation")

st.markdown("""
## Models Trained

- Logistic Regression
- Random Forest
- XGBoost
""")

st.divider()

st.markdown("""
            ## Cross-Validation Results
            
            The results below show the performance of the model.
            """)

@st.cache_resource
def load_best_models(mlruns_path, experiment_id, primary_metric):
    exp_path = Path(mlruns_path) / experiment_id
    runs_by_model = {}

    for run_dir in exp_path.iterdir():
        # skip non-run dirs (models folder, .trash, etc.)
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        if not all(c in "0123456789abcdef" for c in run_dir.name):
            continue

        tags = {}
        tags_dir = run_dir / "tags"
        if tags_dir.exists():
            for f in tags_dir.iterdir():
                tags[f.name] = f.read_text().strip()

        model_type = tags.get("model_type", "unknown")
        dataset_type = tags.get("dataset_type", "original")

        metrics = {}
        metrics_dir = run_dir / "metrics"
        if metrics_dir.exists():
            for f in metrics_dir.iterdir():
                lines = f.read_text().strip().splitlines()
                if lines:
                    parts = lines[-1].split()
                    try:
                        metrics[f.name] = float(parts[1]) if len(parts) >= 2 else float(parts[0])
                    except ValueError:
                        pass

        # find linked model folder via outputs/m-* directory
        model_path = None
        outputs_dir = run_dir / "outputs"
        if outputs_dir.exists():
            for entry in outputs_dir.iterdir():
                if entry.name.startswith("m-"):
                    candidate = exp_path / "models" / entry.name
                    if candidate.exists():
                        model_path = candidate
                        break

        group_key = f"{model_type} ({'SMOTE' if dataset_type == 'smote' else 'Original'})"
        runs_by_model.setdefault(group_key, []).append({
            "run_id": run_dir.name,
            "run_name": tags.get("mlflow.runName", run_dir.name[:8]),
            "model_type": model_type,
            "dataset_type": dataset_type,
            "group_key": group_key,
            "metrics": metrics,
            "model_path": model_path,
        })

    # pick best run per model type by primary metric
    best_runs = {}
    for group_key, runs in runs_by_model.items():
        scored = [r for r in runs if primary_metric in r["metrics"]]
        best = max(scored, key=lambda r: r["metrics"][primary_metric]) if scored else runs[0]
        best_runs[group_key] = best

    # load models
    loaded = {}
    for group_key, run in best_runs.items():
        model, err = _load_model(run["model_path"])
        loaded[group_key] = {"run": run, "model": model, "load_error": err}

    return loaded


def _load_model(model_path):
    if model_path is None:
        return None, "No model folder linked to this run"
    pkl = model_path / "artifacts" / "model.pkl"
    if pkl.exists():
        try:
            import pickle
            return pickle.load(open(pkl, "rb")), None
        except Exception as e:
            return None, str(e)
    return None, f"model.pkl not found in {model_path}"



best_models = load_best_models(MLRUNS_PATH, EXPERIMENT_ID, PRIMARY_METRIC)

if not best_models:
    st.warning("No runs found — check MLRUNS_PATH.")
else:
    rows = []
    for model_type, data in best_models.items():
        metrics = data["run"]["metrics"]
        rows.append({
            "Model": model_type,
            "F1 Score": round(metrics.get("best_f1_macro", float("nan")), 4),
            "Run": data["run"]["run_name"],
            "Loaded": "✅" if data["model"] else f"❌ {data['load_error']}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    # Models available throughout the rest of the app via session state
    st.session_state["best_models"] = {k: v["model"] for k, v in best_models.items()}


st.divider()


st.markdown("""

## Feature Importance

Using the random forest models (both smote and original data), as well as the XGBoost trained on SMOTE data, we can evalutate the feature importances for models that were trained on the original imbalanced data, as well as the model that was trained on the balanced SMOTE data.
            
""")

tab1, tab2 , tab3= st.tabs(["Non-SMOTE RF", "SMOTE RF", "XGB SMOTE"])
ASSETS_DIR = Path(__file__).parent.parent / "assets"

with tab1:
    st.subheader("Random Forest Feature Importance (Non-SMOTE)")
    st.image(str(ASSETS_DIR / "03_original_feature_importance_rf.png"), use_container_width=True)

with tab2:
    st.subheader("Random Forest Feature Importance (SMOTE)")
    st.image(str(ASSETS_DIR / "03_smote_feature_importance_rf.png"), use_container_width=True)

with tab3:
    st.subheader("XGBoost Feature Importance (SMOTE) — SHAP")

    @st.cache_data
    def load_shap_data():
        DATA_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "train_smote.csv"
        df = pd.read_csv(DATA_PATH)
        target_col = df.columns[-1]  # assumes target is the last column
        X = df.drop(columns=[target_col])
        return X

    @st.cache_resource
    def compute_shap(_model, _X):
        explainer = shap.TreeExplainer(_model)
        shap_values = explainer.shap_values(_X)
        return shap_values

    xgb_model = st.session_state.get("best_models", {}).get("XGBoost (SMOTE)")

    if xgb_model is None:
        st.warning("XGBoost SMOTE model not loaded yet — make sure the results table above has run first.")
    else:
        with st.spinner("Computing SHAP values..."):
            X = load_shap_data()
            shap_values = compute_shap(xgb_model, X)

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X, plot_type="bar", show=False, ax=ax)
        st.pyplot(fig)
        plt.close()


st.markdown("""

The feature importance plots show how much each feature contributes tothe model's predictions. We can see which mechanics and game context have the most influence on hit outcomes, which is important information to have. 
            
The models show that launch angle has the biggest influence on predictions, as well as the power of the swing, which is captured in the interaction between launch angle and launch speed. This makes sense when we think about it logically, if a player is struggling with grounding out, they might want to work on increasing their launch angle. Having this information allows batters to train with this in mind.

The SHAP plot shows us how much each feature impacts the prediction of what class. We can see that launch angle has the larget impact on predicting a homerun, which aligns with practical baseball knowledge.  
            """)


st.divider()

st.markdown("""
## Model Selection Rationale

The XGBoost model trained on SMOTE balanced data achieved the highest macro F1 score of 0.884, which outperformed the othermodels. 
            
SMOTE addressed the class imbalance in the orginal dataset by generating minority class samples during training, which prevents the model from just predicting the majority class (simgles) each time. We can see that the XGBoost model improved greatly just by balancing the data, with the F1 score increasing from 0.763 to 0.884. 
            
XGBoost did better than both logistic regression and random forest because it builds trees sequentially, correcting the errors from the previous one. It is better at capturing nonlinear relationships, which is useful as there is not a linear relationship between launch and swing mechanics and hit outcomes. These are relationships that are captured in this model but are not caputred in the logistic regression model. (This is another reason why it outperforms  the model from the undergraduate project)
            
This model also uses built in regularization and the cross validation evaluates based on held out folds rather than the training data itself, which gives a more honest estimate of generalization. 
""")
