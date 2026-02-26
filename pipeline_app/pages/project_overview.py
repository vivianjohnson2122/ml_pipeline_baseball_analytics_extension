import streamlit as st
import pandas as pd
st.title("Project Overview")

st.markdown("""
## Background

- Originally devloped in R as an undergraduate honors capstone project
- Focused primarily on mulitnomial logistic regression 
- Emphasized statistical interpretation over system design 
- Preprocessing and modeling were more linear and less modular 
- Limited exploration of model comparison and reproducibility tooling 
            
My first project was more statistically focused, but not engineered as a scalable ML system. 

## Motivation for Python Extension

My goal for the extension was to enhance the framework I built. To make it scalable, more insightful, and more powerful. 
            
1. Move from statistical modeling to a full ML pipeline
2. Expand model exploration - to build more accurate and reliable models using more advanced techniques
3. Implement feature engineering to take into account more important variables 
            - Just using the hitting mechanics is a good start, but there are so many different factors that go into an at bat and the result of it. 

## Problem Definition

The data focuses on batting data from MLB games played from March 28, 2024 to October 30, 2024. The data was scraped from Baseball Savant using the `statcast()` function from the `pybaseball` package in python. Once compiled and cleaned, the data was saved to a CSV titled `full_clean.csv`.


## Dataset

""")
st.subheader("Target Variable Definition")

target_var = eature_definitions = pd.DataFrame([
    {
        "Target Variable": "hit_outcome",
        "Description": "Result of the at bat. 1=Single, 2=Double/Triple (Extra Basehit), 3=Homerun"
    }])

st.dataframe(
    target_var,
    use_container_width=True,
    hide_index=True
)

st.divider()

st.subheader("Model Feature Definitions")

feature_definitions = pd.DataFrame([
    {
        "Feature Name": "release_speed_z",
        "Category": "Swing Mechanics (Standardized)",
        "Description": "Pitch release speed standardized using training set mean and standard deviation."
    },
    {
        "Feature Name": "launch_angle",
        "Category": "Batted Ball Metrics",
        "Description": "Vertical launch angle of the batted ball in degrees."
    },
    {
        "Feature Name": "launch_speed_z",
        "Category": "Swing Mechanics (Standardized)",
        "Description": "Exit velocity standardized relative to the training distribution."
    },
    {
        "Feature Name": "bat_speed_z",
        "Category": "Swing Mechanics (Standardized)",
        "Description": "Bat speed standardized to capture relative swing intensity."
    },
    {
        "Feature Name": "swing_length_z",
        "Category": "Swing Mechanics (Standardized)",
        "Description": "Length of the swing path standardized to reduce scale effects."
    },
    {
        "Feature Name": "bat_speed_x_swing_length",
        "Category": "Interaction Term",
        "Description": "Interaction capturing combined effects of swing speed and swing length."
    },
    {
        "Feature Name": "launch_speed_x_launch_angle",
        "Category": "Interaction Term",
        "Description": "Interaction capturing nonlinear relationship between exit velocity and launch angle."
    },
    {
        "Feature Name": "count_state",
        "Category": "Game Context",
        "Description": "Encoded count state computed as (balls × 10 + strikes), ranging from 0–32."
    },
    {
        "Feature Name": "ahead_in_count",
        "Category": "Game Context",
        "Description": "Binary indicator equal to 1 if the batter is ahead in the count."
    },
    {
        "Feature Name": "behind_in_count",
        "Category": "Game Context",
        "Description": "Binary indicator equal to 1 if the batter is behind in the count."
    },
    {
        "Feature Name": "two_strike",
        "Category": "Game Context",
        "Description": "Binary indicator equal to 1 if there are two strikes in the count."
    },
    {
        "Feature Name": "same_hand",
        "Category": "Handedness Matchup",
        "Description": "Binary indicator equal to 1 if batter and pitcher share the same handedness."
    },
    {
        "Feature Name": "pitch_group_fastball",
        "Category": "Pitch Type",
        "Description": "One-hot encoded indicator for fastball-type pitches."
    },
    {
        "Feature Name": "pitch_group_breaking",
        "Category": "Pitch Type",
        "Description": "One-hot encoded indicator for breaking pitches."
    },
    {
        "Feature Name": "pitch_group_offspeed",
        "Category": "Pitch Type",
        "Description": "One-hot encoded indicator for offspeed pitches."
    },
])

st.dataframe(
    feature_definitions,
    use_container_width=True,
    hide_index=True
)

st.divider()

st.markdown("## Final Goal")

st.success("""
Predict MLB hit outcome classification using structured statcast data 
within a reproducible and scalable ML pipeline.
""")