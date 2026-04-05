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
        return {
            "type": "legacy",
            "model": joblib.load(LEGACY_MODEL_PATH),
            "scaler": joblib.load(LEGACY_SCALER_PATH),
            "feature_count": int(joblib.load(LEGACY_MODEL_PATH).n_features_in_),
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
        return "HIGH", "🚨 High risk: seizure likely. Immediate clinical action recommended."
    if probability >= moderate_threshold:
        return "MODERATE", "⚠️ Moderate risk: monitor closely and consider intervention."
    return "LOW", "✅ Low risk: continue standard monitoring."


st.set_page_config(page_title="Seizure Risk Prediction", page_icon="🧠", layout="wide")
st.title("🧠 Seizure Risk Prediction")
st.caption("Upload extracted EEG/ECG features and estimate short-term seizure risk.")

try:
    artifacts = load_model_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

st.sidebar.header("Model Information")
st.sidebar.write(f"Loaded model type: **{artifacts['type']}**")
if artifacts["metrics"]:
    for metric_name, value in artifacts["metrics"].items():
        st.sidebar.write(f"{metric_name}: **{value:.3f}**")

mode = st.sidebar.radio("Input Mode", ["Clinical CSV", "Symptom Triage"])

if mode == "Clinical CSV":
    st.subheader("Clinical CSV Prediction")
    uploaded = st.file_uploader("Upload feature CSV", type=["csv"])

    if uploaded:
        data = pd.read_csv(uploaded)
        st.write("Preview:")
        st.dataframe(data.head())

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

            st.metric("Average Seizure Risk", f"{avg_probability * 100:.2f}%")
            st.progress(int(max(0, min(100, avg_probability * 100))))
            st.write(f"**Risk Level: {risk}**")
            st.info(message)

            st.subheader("Per-row Risk Distribution")
            st.bar_chart(pd.DataFrame({"probability": probabilities[:200]}))

        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction failed: {exc}")

else:
    st.subheader("Symptom Triage (Non-diagnostic)")
    st.warning(
        "This mode is a safety triage tool and does not replace EEG/ECG model predictions."
    )

    col1, col2 = st.columns(2)
    with col1:
        heart_rate = st.slider("Heart Rate", 45, 180, 85)
        spo2 = st.slider("SpO2", 75, 100, 96)
        stress = st.slider("Stress", 0, 10, 4)
    with col2:
        sleep_quality = st.slider("Sleep Quality", 0, 10, 6)
        fatigue = st.slider("Fatigue", 0, 10, 4)
        aura = st.checkbox("Aura / unusual pre-seizure sensations")
        confusion = st.checkbox("Confusion episodes")

    if st.button("Run Triage"):
        triage_score = (
            (heart_rate > 115) * 0.20
            + (spo2 < 90) * 0.20
            + (stress > 7) * 0.15
            + (sleep_quality < 4) * 0.10
            + (fatigue > 7) * 0.10
            + aura * 0.15
            + confusion * 0.10
        )
        triage_prob = float(min(1.0, triage_score))
        triage_risk, triage_message = risk_label(triage_prob, 0.6)

        st.metric("Triage Risk", f"{triage_prob * 100:.2f}%")
        st.progress(int(triage_prob * 100))
        st.write(f"**Risk Level: {triage_risk}**")
        st.info(triage_message)

st.markdown("---")
st.caption("For medical emergencies, contact local emergency services immediately.")
