import streamlit as st
from pathlib import Path

st.title("ML Pipeline Architecture")

st.markdown("""
## Pipeline Stages

### 1. Data Ingestion
- Raw Statcast data is ingested and filtered to keep only events where there was an actual hit (a single, extra basehit, or a homerun). 
### 2. Cleaning & Feature Engineering
- Missing values did not make up a big percentage of the data, but were imputed using K-Neares Neighbors approach to preserve relationships between correlated swing and pitch metrics. Additional features and interactions were engineered to caputre other effects on a hit outcome that are rooted in game context. These capture nonlinear relationships between swing mechanics, pitch characteristics, and hit outcomes.
### 3. Outlier Removal
 - Extreme outliers were removed based on the interquartile range to reduce the influence of possible anomalies, such as check swings, bunts, accidental pitches, etc. This also ensures that anomalous tracking errors aren't disproportionately affecting model training. 
### 4. Scaling
- Continuous features were standardized (into Z-scores) using statistics derived from the training set only to ensure there was no leakage. This ensures consistent feature scales across models. 
### 5. SMOTE (Class Balancing)
 - Synthetic Minority Oversampling is applied to address the significant class imbalance across hit outcomes. This imbalance makes sense, as singles are more popular than homeruns. This allows the model to better learn decision boundaries for the underrepresented hit types while also preserving the overall data structure.
""")

ASSETS_DIR = Path(__file__).parent.parent / "assets"
st.image(ASSETS_DIR / "02_smote_comparison.png", use_container_width=True)


st.markdown("""
### 6. Model Training
 - Multiple model families are trained (Logistic Regression, Random Forest, XGBoost) to compare performance across different complexity levels.
### 7. Evaluation
 - Models are evaluated using stratified K-Fold cross validation and a performance metric of F1-score to asses overall performance. 
### 8. Model Tracking (MLflow)
 - Each experiment is logged using MLFlow, including parameters, metrics, and artifacts. This enables systematic comparison and reproducibility. 
""")

st.divider()

st.markdown("""
## Preprocessing Strategy

### Stratified K-Fold Validation: 
            
- This was used to preserve the original class distribution of hit outcomes within each fold. Given the inherent class imbalance, this was important to ensure that evaluation metrics remain stable and comparable, and model performance more accurately reflects real-world prediction scenarios. 
""")

st.divider()

st.markdown("## Architecture Diagram")

# optional image placeholder

st.image(ASSETS_DIR / "architecture_diagram.png", caption="Pipeline Diagram")