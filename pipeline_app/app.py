import streamlit as st

st.set_page_config(
    page_title="ML Pipeline Project",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("⚾ MLB Hit Outcome Prediction Pipeline")
st.subheader("End-to-End Machine Learning System in Python")

st.markdown("""
### Project Summary

  This project aims to study classification and its applications to Major League Baseball (MLB) batting data. The goal is to develop a model that is capable of classifying a hit based on various in-game metrics, including release speed, launch angle, bat speed, and swing length. The model development process incorporates different statistical techniques such as KNN Imputing, K-Means Cluster Analysis, feature engineering, and cross validation to understand the distributions of the data and how observations are being classified in the different models. Additionally, specific balancing techniques are explored to address inherent bias and imbalance in the data for the purpose of making more accurate classifications. 

This project helps shed insight into the question: "What differentiates a homerun from a single or an extra basehit?" We study the different mechanics and situations that go into each hit type, and attempt to shed light on how they are different, and make predictions given data.
            
This is an extension of my undergraduate Honors Capstone project, completed in R, that is more limited. This project explores more indepth model selection, trains and builds three models, not just logistic regression. It also engineers more features to be able to capture more signal out of other categorical variables that may have been missed. 
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Technical Stack
    - Python
    - scikit-learn
    - Feature Engineering
    - Logistic Regression, Random Forest, XGBoost
    - MLflow
    - SMOTE
    - Streamlit
    """)

with col2:
    st.markdown("""
    ### Key Features
    - Full ML pipeline
    - Cross-validation
    - Model tracking with MLflow
    - Reproducible preprocessing
    - Interactive prediction interface
    """)

st.divider()

st.info("Use the sidebar to navigate through the project sections.")