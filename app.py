import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =========================
# LOAD MODEL + SCALER
# =========================
signal_model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Seizure Prediction AI",
    layout="wide",
    page_icon="🧠"
)

# =========================
# HEADER
# =========================
st.markdown("""
# 🧠 Multimodal Seizure Prediction System
### Predict seizure risk (5–30 min prior) using EEG & ECG signals
""")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Input Mode")
option = st.sidebar.radio("Choose Mode", [
    "📂 Clinical (CSV Upload)",
    "🩺 Real-Time Symptoms"
])

# =========================
# OPTION 1 — CSV (REAL MODEL)
# =========================
if option == "📂 Clinical (CSV Upload)":

    st.subheader("📂 Upload Signal Features CSV")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file is not None:
        data = pd.read_csv(file)

        st.write("### 🔍 Data Preview")
        st.dataframe(data.head())

        # Remove label if exists
        if "label" in data.columns:
            data = data.drop("label", axis=1)

        required_features = signal_model.n_features_in_

        if data.shape[1] != required_features:
            st.error(f"❌ Expected {required_features} features, got {data.shape[1]}")
        else:
            try:
                # 🔥 APPLY SCALING
                data_scaled = scaler.transform(data)

                # 🔥 PREDICTION
                prob = signal_model.predict_proba(data_scaled)[:, 1].mean()

                st.markdown("## 📊 Prediction Result")

                st.metric("🧠 Seizure Risk", f"{prob*100:.2f}%")
                st.progress(int(prob * 100))

                if prob > 0.8:
                    st.error("🚨 HIGH RISK: Seizure likely within 5–10 minutes!")
                elif prob > 0.6:
                    st.warning("⚠️ MODERATE RISK: Possible seizure within 30 minutes")
                else:
                    st.success("✅ LOW RISK")

                # Visualization
                st.subheader("📈 Signal Visualization")
                st.line_chart(data.iloc[:100])

            except Exception as e:
                st.error(f"❌ Prediction Error: {e}")

# =========================
# OPTION 2 — SYMPTOMS (SUPPORT ONLY)
# =========================
else:

    st.subheader("🩺 Real-Time Health Monitoring")

    col1, col2 = st.columns(2)

    with col1:
        heart_rate = st.slider("Heart Rate", 60, 150, 80)
        spo2 = st.slider("Oxygen Level (SpO2)", 80, 100, 95)
        stress = st.slider("Stress Level", 0, 10, 5)

    with col2:
        sleep = st.slider("Sleep Quality", 0, 10, 5)
        fatigue = st.slider("Fatigue Level", 0, 10, 5)
        dizziness = st.checkbox("Dizziness")
        headache = st.checkbox("Headache")

    st.warning("⚠️ Symptom-based prediction is less accurate. Use signal data for reliable results.")

    if st.button("Predict Seizure Risk"):

        # Simple heuristic (NOT ML — safer than fake model)
        risk_score = (
            (heart_rate > 110) * 0.2 +
            (spo2 < 90) * 0.2 +
            (stress > 7) * 0.2 +
            (sleep < 4) * 0.1 +
            (fatigue > 7) * 0.1 +
            dizziness * 0.1 +
            headache * 0.1
        )

        prob = min(risk_score, 1.0)

        st.markdown("## 📊 Prediction Result")

        st.metric("🧠 Estimated Risk", f"{prob*100:.2f}%")
        st.progress(int(prob * 100))

        if prob > 0.7:
            st.error("🚨 HIGH RISK: Consult doctor immediately!")
        elif prob > 0.4:
            st.warning("⚠️ MODERATE RISK")
        else:
            st.success("✅ LOW RISK")

# =========================
# HEALTHCARE PANEL
# =========================
st.markdown("---")
st.subheader("🏥 Emergency & Healthcare Support")

col1, col2 = st.columns(2)

with col1:
    st.error("📞 Emergency: 108")
    st.info("🏥 Nearby Hospitals: Apollo / AIIMS")

with col2:
    st.warning("👨‍⚕️ Consult Neurologist Immediately if High Risk")
    st.success("📍 Recommended: Government Hospital / Specialist Clinic")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("⚕️ AI-powered healthcare system for early seizure prediction")
