import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle 
from pathlib import Path

MLRUNS_PATH = Path(__file__).parent.parent.parent / "mlruns"

@st.cache_resource
def load_xgb_smote():
    exp_path = MLRUNS_PATH / "0"
    for run_dir in exp_path.iterdir():
        if not run_dir.is_dir() or not all(c in "0123456789abcdef" for c in run_dir.name):
            continue
        tags = {}
        tags_dir = run_dir / "tags"
        if tags_dir.exists():
            for f in tags_dir.iterdir():
                tags[f.name] = f.read_text().strip()
        if tags.get("model_type") != "XGBoost" or tags.get("dataset_type") != "smote":
            continue
        outputs_dir = run_dir / "outputs"
        if outputs_dir.exists():
            for entry in outputs_dir.iterdir():
                if entry.name.startswith("m-"):
                    pkl = exp_path / "models" / entry.name / "artifacts" / "model.pkl"
                    if pkl.exists():
                        return pickle.load(open(pkl, "rb"))
    return None


st.title("Interactive Prediction Interface")

st.markdown("""
Use this interface to generate predictions using trained models.
""")

st.divider()

# ─────────────────────────────────────────────
# MODEL SELECTION
# ─────────────────────────────────────────────

# ── Load model ────────────────────────────────────────────────────────────────
model = load_xgb_smote()

if model is None:
    st.error("Could not load XGBoost SMOTE model — check that ../mlruns path is correct.")
    st.stop()


# ── Feature defaults ──────────────────────────────────────────────────────────
# All z-scored: 0 = average, positive = above average, negative = below average

st.markdown("## Swing Mechanics")
col1, col2 = st.columns(2)

with col1:
    release_speed = st.slider("Pitch Speed (mph)", 70.0, 105.0, 89.4, 0.1)
    launch_speed = st.slider("Exit Velocity (mph)", 50.0, 120.0, 94.0, 0.1)
    bat_speed = st.slider("Bat Speed (mph)", 55.0, 90.0, 71.6, 0.1)

with col2:
    swing_length = st.slider("Swing Length (ft)", 4.5, 9.5, 7.2, 0.1)
    launch_angle = st.slider("Launch Angle (degrees)", -30, 60, 12, 1)

st.markdown("## Count & Situation")
col3, col4 = st.columns(2)

with col3:
    ahead_in_count = st.toggle("Batter ahead in count", value=False)
    behind_in_count = st.toggle("Batter behind in count", value=False)
    two_strike = st.toggle("Two strikes", value=False)
    same_hand = st.toggle("Same handedness (batter/pitcher)", value=False)

with col4:
    balls = st.selectbox("Balls", [0, 1, 2, 3], index=0)
    strikes = st.selectbox("Strikes", [0, 1, 2], index=0)
    count_state = balls * 10 + strikes

st.markdown("## Pitch Type")
pitch_type = st.radio(
    "Pitch group",
    ["Fastball", "Breaking", "Offspeed"],
    horizontal=True
)
pitch_group_fastball = 1 if pitch_type == "Fastball" else 0
pitch_group_breaking = 1 if pitch_type == "Breaking" else 0
pitch_group_offspeed = 1 if pitch_type == "Offspeed" else 0

release_speed_z = (release_speed - 89.4125) / 5.7809
launch_speed_z  = (launch_speed - 94.0102) / 13.4586
bat_speed_z     = (bat_speed - 71.6416) / 5.4598
swing_length_z  = (swing_length - 7.2154) / 0.6922

# ── Compute interaction terms ─────────────────────────────────────────────────
bat_speed_x_swing_length = bat_speed_z * swing_length_z
launch_speed_x_launch_angle = launch_speed_z * launch_angle

# ── Build feature row ─────────────────────────────────────────────────────────
input_data = pd.DataFrame([{
    "release_speed_z":          release_speed_z,
    "launch_angle":             launch_angle,
    "launch_speed_z":           launch_speed_z,
    "bat_speed_z":              bat_speed_z,
    "swing_length_z":           swing_length_z,
    "bat_speed_x_swing_length": bat_speed_x_swing_length,
    "launch_speed_x_launch_angle": launch_speed_x_launch_angle,
    "count_state":              count_state,
    "ahead_in_count":           int(ahead_in_count),
    "behind_in_count":          int(behind_in_count),
    "two_strike":               int(two_strike),
    "same_hand":                int(same_hand),
    "pitch_group_fastball":     pitch_group_fastball,
    "pitch_group_breaking":     pitch_group_breaking,
    "pitch_group_offspeed":     pitch_group_offspeed,
}])

# ── Predict ───────────────────────────────────────────────────────────────────
st.divider()

if st.button("⚾ Predict Hit Outcome"):
    proba = model.predict_proba(input_data)[0]
    classes = model.classes_

    label_map = {0: "Single", 1: "Extra Base Hit", 2: "Home Run"}
    labels = [label_map.get(c, str(c)) for c in classes]

    pred_idx = np.argmax(proba)
    pred_label = label_map.get(classes[pred_idx], str(classes[pred_idx]))
    pred_proba = max(proba)

    st.markdown(f"### Most likely outcome: **{pred_label}** ({pred_proba:.1%})")

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4ade80" if i == pred_idx else "#334155" for i in range(len(classes))]
    bars = ax.bar(labels, proba, color=colors, width=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    for bar, p in zip(bars, proba):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{p:.1%}", ha="center", fontsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    st.pyplot(fig)
    plt.close()