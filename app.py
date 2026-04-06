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
        legacy_scaler = joblib.load(LEGACY_SCALER_PATH)

        model_feature_count = int(getattr(legacy_model, "n_features_in_", 0) or 0)
        scaler_feature_count = int(getattr(legacy_scaler, "n_features_in_", 0) or 0)
        expected_feature_count = model_feature_count or scaler_feature_count

        return {
            "type": "legacy",
            "model": legacy_model,
            "scaler": legacy_scaler,
            "model_feature_count": model_feature_count,
            "scaler_feature_count": scaler_feature_count,
            "feature_count": expected_feature_count,
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


def prepare_legacy_input(
    data: pd.DataFrame, expected_feature_count: int
) -> tuple[np.ndarray, list[str]]:
    """Prepare numeric matrix for legacy scaler/model with soft alignment."""
    warnings: list[str] = []
    numeric_data = data.select_dtypes(include=[np.number]).copy()

    if numeric_data.shape[1] == 0:
        raise ValueError("No numeric columns found in uploaded file.")

    if numeric_data.shape[1] > expected_feature_count:
        warnings.append(
            f"Found {numeric_data.shape[1]} numeric columns. Using first {expected_feature_count} for legacy model."
        )
        numeric_data = numeric_data.iloc[:, :expected_feature_count]
    elif numeric_data.shape[1] < expected_feature_count:
        missing = expected_feature_count - numeric_data.shape[1]
        warnings.append(
            f"Found only {numeric_data.shape[1]} numeric columns. Padding {missing} missing columns with 0.0."
        )
        for idx in range(missing):
            numeric_data[f"_pad_{idx}"] = 0.0

    numeric_data = numeric_data.fillna(numeric_data.median(numeric_only=True)).fillna(0.0)
    matrix = numeric_data.to_numpy(dtype=float)
    return matrix, warnings


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
        background: linear-gradient(180deg, #f3f7ff 0%, #e8eef8 100%);
        color: #10233f;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label, .stApp li, .stApp div {
        color: #10233f;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border: 1px solid rgba(148, 163, 184, 0.45);
        border-radius: 10px;
        margin-bottom: 0.3rem;
        padding: 0.45rem 0.55rem;
        background: rgba(15, 23, 42, 0.4);
    }
    .app-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.25rem 1.25rem 0.75rem 1.25rem;
        border: 1px solid #c6d5ea;
        box-shadow: 0 8px 26px rgba(15, 23, 42, 0.08);
        margin-bottom: 1rem;
    }
    .stMetric {
        background: #f8fbff;
        border: 1px solid #d7e4f3;
        border-radius: 12px;
        padding: 0.4rem 0.75rem;
    }
    .stFileUploader {
        background: #f8fbff;
        border: 1px solid #d7e4f3;
        border-radius: 12px;
        padding: 0.25rem 0.75rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #0ea5e9 0%, #2563eb 100%);
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        height: 2.8rem;
        background: linear-gradient(90deg, #0369a1 0%, #1d4ed8 100%);
        color: white;
        border: none;
        box-shadow: 0 8px 16px rgba(29, 78, 216, 0.2);
    }
    .stButton > button:hover {
        filter: brightness(1.05);
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

        source_target = None
        for col in ["label", "target", "seizure", "risk"]:
            if col in data.columns:
                if source_target is None:
                    source_target = pd.to_numeric(data[col], errors="coerce")
                data = data.drop(columns=[col])

        try:
            if artifacts["type"] == "bundle":
                input_df = align_features(data, artifacts["feature_names"])
                probabilities = artifacts["model"].predict_proba(input_df)[:, 1]
            else:
                legacy_input, legacy_warnings = prepare_legacy_input(
                    data, artifacts["feature_count"]
                )
                for warning_text in legacy_warnings:
                    st.warning(warning_text)

                scaler_feature_count = artifacts.get("scaler_feature_count", 0)
                if scaler_feature_count and scaler_feature_count == legacy_input.shape[1]:
                    model_input = artifacts["scaler"].transform(legacy_input)
                else:
                    model_input = legacy_input
                    if scaler_feature_count:
                        st.warning(
                            "Legacy scaler expects "
                            f"{scaler_feature_count} features, but legacy model expects "
                            f"{legacy_input.shape[1]}. Skipping scaler to keep model input valid."
                        )

                probabilities = artifacts["model"].predict_proba(model_input)[:, 1]

            avg_probability = float(np.mean(probabilities))
            risk, message = risk_label(avg_probability, artifacts["threshold"])
            row_predictions = (probabilities >= artifacts["threshold"]).astype(int)

            st.metric("Average seizure risk", f"{avg_probability * 100:.2f}%")
            st.progress(int(max(0, min(100, avg_probability * 100))))
            st.write(f"**Risk level: {risk}**")
            st.info(message)

            if source_target is not None:
                valid_target = source_target.dropna().astype(int)
                if len(valid_target) > 0:
                    seizure_ratio = float((valid_target == 1).mean())
                    if seizure_ratio > 0:
                        st.error(
                            f"Seizure found in uploaded file ({seizure_ratio * 100:.1f}% positive rows)."
                        )
                    else:
                        st.success("No seizure label found in uploaded file (all rows are non-seizure).")

            if int(row_predictions.max()) == 1:
                st.error("Model decision: **SEIZURE DETECTED**. Contact emergency services immediately.")
            else:
                st.success("Model decision: **NO RISK**.")

            st.subheader("Risk by uploaded rows")
            st.bar_chart(pd.DataFrame({"probability": probabilities[:200]}))

            st.subheader("Row-level model output")
            preview = pd.DataFrame(
                {
                    "row_index": np.arange(len(probabilities)),
                    "probability": probabilities,
                    "prediction": np.where(row_predictions == 1, "SEIZURE", "NO RISK"),
                }
            ).head(200)
            st.dataframe(preview, use_container_width=True)

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
