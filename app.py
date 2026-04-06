from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_BUNDLE_PATH = Path("model_bundle.pkl")
LEGACY_MODEL_PATH = Path("final_model.pkl")
LEGACY_SCALER_PATH = Path("scaler.pkl")


def load_model_artifacts() -> dict:
    """Load the newest model bundle if available, otherwise fallback to legacy files."""
    if MODEL_BUNDLE_PATH.exists():
        bundle = joblib.load(MODEL_BUNDLE_PATH)
        return {
            "type": "bundle",
            "model": bundle["model"],
            "feature_names": bundle["feature_names"],
            "threshold": float(bundle.get("threshold", 0.5)),
            "metrics": bundle.get("metrics", {}),
        }

    if LEGACY_MODEL_PATH.exists() and LEGACY_SCALER_PATH.exists():
        legacy_model = joblib.load(LEGACY_MODEL_PATH)
        return {
            "type": "legacy",
            "model": legacy_model,
            "scaler": joblib.load(LEGACY_SCALER_PATH),
            "feature_count": int(legacy_model.n_features_in_),
            "threshold": 0.5,
            "metrics": {},
        }

    raise FileNotFoundError(
        "No model found. Train a model with train_model.py or provide final_model.pkl + scaler.pkl."
    )


def align_features(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Align uploaded dataframe to trained feature schema."""
    aligned = df.copy()

    for col in expected:
        if col not in aligned.columns:
            aligned[col] = np.nan

    return aligned[expected]


def risk_label(probability: float, threshold: float) -> tuple[str, str]:
    moderate_threshold = max(0.30, threshold * 0.7)
    high_threshold = threshold

    if probability >= high_threshold:
        return "HIGH", "🚨 High risk: seizure likely. Contact medical support immediately."
    if probability >= moderate_threshold:
        return "MODERATE", "⚠️ Moderate risk: keep close watch and seek medical advice soon."
    return "LOW", "✅ Low risk: continue normal monitoring and safety precautions."


st.set_page_config(page_title="Seizure Safety Assistant", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7fbff 0%, #eef3f9 100%);
    }
    .app-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.25rem 1.25rem 0.75rem 1.25rem;
        border: 1px solid #dbe6f3;
        box-shadow: 0 8px 20px rgba(44, 62, 80, 0.07);
        margin-bottom: 1rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        height: 2.8rem;
        background: #0c4a6e;
        color: white;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩺 Seizure Safety Assistant")
st.caption(
    "Simple tools to estimate seizure risk from a medical file or a quick symptom check."
)

try:
    artifacts = load_model_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

st.sidebar.header("Model Details")
st.sidebar.write(f"Loaded model type: **{artifacts['type']}**")
if artifacts["metrics"]:
    for metric_name, value in artifacts["metrics"].items():
        st.sidebar.write(f"{metric_name}: **{value:.3f}**")

mode = st.sidebar.radio("Choose a tool", ["Medical File Check", "Quick Health Check"])

if mode == "Medical File Check":
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.subheader("Medical File Check")
    st.write(
        "Upload a CSV file from your clinical system. The app will estimate average seizure risk from the data."
    )

    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded:
        data = pd.read_csv(uploaded)
        st.write("File preview")
        st.dataframe(data.head(), use_container_width=True)

        for col in ["label", "target", "seizure", "risk"]:
            if col in data.columns:
                data = data.drop(columns=[col])

        try:
            if artifacts["type"] == "bundle":
                input_df = align_features(data, artifacts["feature_names"])
                probabilities = artifacts["model"].predict_proba(input_df)[:, 1]
            else:
                if data.shape[1] != artifacts["feature_count"]:
                    st.error(
                        f"Expected {artifacts['feature_count']} columns, but got {data.shape[1]}."
                    )
                    st.stop()
                scaled = artifacts["scaler"].transform(data)
                probabilities = artifacts["model"].predict_proba(scaled)[:, 1]

            avg_probability = float(np.mean(probabilities))
            risk, message = risk_label(avg_probability, artifacts["threshold"])

            st.metric("Average seizure risk", f"{avg_probability * 100:.2f}%")
            st.progress(int(max(0, min(100, avg_probability * 100))))
            st.write(f"**Risk level: {risk}**")
            st.info(message)

            st.subheader("Risk by uploaded rows")
            st.bar_chart(pd.DataFrame({"probability": probabilities[:200]}))

        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction failed: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.subheader("Quick Health Check (easy language)")
    st.warning(
        "This check is for awareness only. It does not replace hospital tests or doctor advice."
    )

    vitals_col, symptoms_col = st.columns(2)

    with vitals_col:
        st.markdown("#### Body signs")
        heart_rate = st.slider("Heart rate (beats per minute)", 40, 180, 84)
        top_bp = st.slider("Blood pressure - top number", 80, 200, 122)
        bottom_bp = st.slider("Blood pressure - bottom number", 45, 130, 78)
        oxygen_level = st.slider("Oxygen level (%)", 75, 100, 97)
        body_temp = st.slider("Body temperature (°F)", 95.0, 106.0, 98.6, 0.1)

    with symptoms_col:
        st.markdown("#### How the person feels")
        high_stress = st.slider("Stress level (0 calm - 10 very stressed)", 0, 10, 4)
        sleep_hours = st.slider("Sleep in last 24 hours", 0, 14, 7)
        strong_tiredness = st.slider("Tiredness (0 none - 10 extreme)", 0, 10, 4)
        headache = st.checkbox("Strong headache")
        dizziness = st.checkbox("Dizziness")
        nausea = st.checkbox("Nausea or vomiting")
        confusion = st.checkbox("Confusion or memory gaps")
        body_jerks = st.checkbox("Sudden body jerks or shaking")
        chest_pain = st.checkbox("Chest pain or chest pressure")

    if st.button("Check risk now"):
        health_score = (
            (heart_rate > 115 or heart_rate < 50) * 0.12
            + (top_bp > 160 or top_bp < 90) * 0.10
            + (bottom_bp > 100 or bottom_bp < 55) * 0.08
            + (oxygen_level < 92) * 0.15
            + (body_temp >= 100.4) * 0.10
            + (high_stress >= 8) * 0.08
            + (sleep_hours <= 4) * 0.08
            + (strong_tiredness >= 8) * 0.06
            + headache * 0.05
            + dizziness * 0.05
            + nausea * 0.04
            + confusion * 0.10
            + body_jerks * 0.12
            + chest_pain * 0.07
        )

        quick_prob = float(min(1.0, health_score))
        quick_risk, quick_message = risk_label(quick_prob, 0.55)

        st.metric("Quick check risk", f"{quick_prob * 100:.2f}%")
        st.progress(int(quick_prob * 100))
        st.write(f"**Risk level: {quick_risk}**")
        st.info(quick_message)

        if quick_risk == "HIGH":
            st.error("Please contact emergency services or go to the nearest hospital now.")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("For emergencies, call local emergency services immediately.")
