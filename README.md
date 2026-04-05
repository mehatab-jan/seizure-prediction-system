# Seizure Prediction System

A cleaner seizure-risk project with:

- Robust model training for imbalanced data.
- Probability calibration and threshold tuning.
- Streamlit app with safer prediction flow.
- Backward compatibility with legacy `final_model.pkl` + `scaler.pkl`.

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Train a new model

```bash
python train_model.py --data your_dataset.csv --output model_bundle.pkl
```

### Dataset expectations

- CSV with numeric feature columns.
- Target column name can be one of: `label`, `target`, `seizure`, `risk`.
- Target values should be binary (`0` or `1`).

`train_model.py` will:

- impute missing values,
- scale features robustly,
- train a class-balanced random forest,
- calibrate probabilities,
- tune threshold for best F1 on test split,
- save all artifacts in `model_bundle.pkl`.

## 3) Run app

```bash
streamlit run app.py
```

### Prediction modes

1. **Clinical CSV**
   - Uses trained model (`model_bundle.pkl` preferred).
   - If absent, falls back to legacy model files.

2. **Symptom Triage**
   - Non-diagnostic safety mode for rough triage only.

## Why this fixes "always low risk"

The previous setup often outputs low-risk probabilities when:

- class imbalance is severe,
- probabilities are uncalibrated,
- decision threshold is fixed at `0.5` for all datasets.

This version addresses all three with balanced training, calibration, and tuned thresholds.

## Important medical note

This project is for research/support workflows and is **not** a standalone clinical diagnosis tool.
