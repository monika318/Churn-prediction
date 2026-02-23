import io
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Churn Risk Dashboard", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("src/churn_model.joblib")

model = load_model()

st.title("📉 Customer Churn Risk Dashboard")
st.write(
    "Upload customer data to predict churn probability (risk score). "
    "This app uses a trained Logistic Regression pipeline with preprocessing."
)

with st.sidebar:
    st.header("Upload")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    threshold = st.slider("Churn threshold", 0.10, 0.90, 0.50, 0.01)
    st.caption("Customers with risk ≥ threshold will be flagged as likely to churn.")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Make the app robust to small column-name differences
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def prep_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare input exactly as training expected:
    - drops customerID if present
    - keeps TotalCharges numeric conversion if present
    """
    df = normalize_columns(df_raw)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # If dataset includes target, drop it for prediction
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    # Common issue in Telco dataset
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df

if uploaded_file is None:
    st.info("Upload the Telco-style customer CSV to begin.")
    st.markdown(
        "**Tip:** Use the same columns as the training dataset. "
        "You can export a sample from your notebook."
    )
    st.stop()

try:
    df_raw = pd.read_csv(uploaded_file)
except Exception:
    # If encoding issues
    uploaded_file.seek(0)
    df_raw = pd.read_csv(uploaded_file, encoding="latin-1")

X = prep_input(df_raw)

# Handle missing TotalCharges created by coercion
if "TotalCharges" in X.columns and X["TotalCharges"].isna().any():
    st.warning(
        f"Found {int(X['TotalCharges'].isna().sum())} rows with invalid TotalCharges. "
        "They will be filled with the median."
    )
    X["TotalCharges"] = X["TotalCharges"].fillna(X["TotalCharges"].median())

# Predict
proba = model.predict_proba(X)[:, 1]
pred = (proba >= threshold).astype(int)

results = df_raw.copy()
results["churn_risk_score"] = np.round(proba, 4)
results["churn_prediction"] = np.where(pred == 1, "Yes", "No")

# Layout
col1, col2, col3 = st.columns(3)
col1.metric("Rows uploaded", len(results))
col2.metric("Predicted churners", int((results["churn_prediction"] == "Yes").sum()))
col3.metric("Predicted churn rate", f"{(results['churn_prediction'].eq('Yes').mean()*100):.1f}%")

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("📋 Predictions (sorted by highest risk)")
    show_cols = list(results.columns)
    # Put score + prediction at front
    show_cols = ["churn_risk_score", "churn_prediction"] + [c for c in show_cols if c not in ["churn_risk_score", "churn_prediction"]]
    st.dataframe(
        results.sort_values("churn_risk_score", ascending=False)[show_cols],
        use_container_width=True,
        height=520
    )

    # Download
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download predictions as CSV",
        data=csv_bytes,
        file_name="churn_predictions.csv",
        mime="text/csv"
    )

with right:
    st.subheader("📊 Risk distribution")
    st.bar_chart(pd.Series(proba).round(2).value_counts().sort_index())

    st.subheader("🚩 High-risk customers")
    high_risk = results[results["churn_risk_score"] >= threshold].copy()
    st.write(high_risk[["churn_risk_score", "churn_prediction"]].head(10))

st.divider()
st.caption("Note: Risk score is the model’s predicted probability of churn.")