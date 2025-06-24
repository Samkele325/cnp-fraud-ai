import streamlit as st
import pandas as pd
import joblib
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime

# --- Full Feature Engineering Function ---
def create_cnp_features(df):
    df = df.sort_values("transaction_time").set_index("transaction_time")
    df["card_ip_province_match"] = (df["card_ip_province"] == df["transaction_province"]).astype(int)
    df["transaction_hour"] = df.index.hour
    df["card_transaction_count_last_1h"] = df.groupby("card_number")["card_number"].transform(
        lambda x: x.rolling("1h").count()
    )
    df["card_transaction_count_last_10min"] = df.groupby("card_number")["card_number"].transform(
        lambda x: x.rolling("10min").count()
    )
    df["is_new_device"] = df.duplicated(subset=["card_number", "device_id"]).apply(lambda x: 0 if x else 1)
    df["card_type_code"] = df["card_type"].astype("category").cat.codes
    df["transaction_type_code"] = df["transaction_type"].astype("category").cat.codes
    df = df.reset_index()
    return df[[
        "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
        "card_ip_province_match", "transaction_hour", "card_transaction_count_last_1h",
        "card_transaction_count_last_10min", "is_new_device", "card_type_code",
        "transaction_type_code", "label"
    ]]

# --- Load Model ---
model = joblib.load("model.pkl")
explainer = shap.TreeExplainer(model)

st.title("🔎 CNP Fraud Detection Demo")

# --- Input method ---
option = st.radio("Select input method:", ["Upload CSV", "Manual Entry"])

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload transaction CSV file", type="csv")
    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file, parse_dates=["transaction_time"])
        df = create_cnp_features(df_raw)
        df = df.fillna(0)
        dmatrix = xgb.DMatrix(df.drop("label", axis=1))
        pred_proba = model.predict(dmatrix)
        df["fraud_probability"] = pred_proba
        st.write("📊 Prediction Results:", df[["fraud_probability"]])

        # SHAP Explanation for first row
        st.subheader("🔍 SHAP Explanation for First CSV Row")
        shap_exp = explainer(df.drop(["label", "fraud_probability"], axis=1))
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_exp[0], max_display=10, show=False)
        st.pyplot(fig)


else:
    st.subheader("Manual Transaction Entry")
    card_number = st.text_input("Card Number", "CARD1234")
    card_ip_province = st.selectbox("Card IP Province", [
        "Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape", "Limpopo",
        "Mpumalanga", "North West", "Free State", "Northern Cape"
    ])
    transaction_province = st.selectbox("Transaction Province", [
        "Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape", "Limpopo",
        "Mpumalanga", "North West", "Free State", "Northern Cape"
    ])
    device_id = st.text_input("Device ID", "DEVICE1")
    card_type = st.selectbox("Card Type", ["VISA", "MasterCard", "AMEX"])
    transaction_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "DEBIT", "PAYMENT"])
    amount = st.number_input("Transaction Amount", min_value=0.0)
    oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0)
    newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0)
    oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0)
    newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0)

    if st.button("Predict"):
        now = datetime.now()
        df_manual = pd.DataFrame([{
            "card_number": card_number,
            "card_ip_province": card_ip_province,
            "transaction_province": transaction_province,
            "transaction_time": now,
            "device_id": device_id,
            "card_type": card_type,
            "transaction_type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
            "label": 0
        }])
        df = create_cnp_features(df_manual)
        df = df.fillna(0)
        dmatrix = xgb.DMatrix(df.drop("label", axis=1))
        prob = model.predict(dmatrix)[0]
        st.success(f"Fraud Probability: {prob:.4f}")

        # SHAP explanation
        shap_values = explainer.shap_values(df.drop("label", axis=1))
        top_features = pd.DataFrame({
            "Feature": df.drop("label", axis=1).columns,
            "SHAP Impact": shap_values[0]
        }).sort_values("SHAP Impact", key=abs, ascending=False).head(5)

        st.subheader("Top Fraud Indicators for This Prediction")
        st.bar_chart(top_features.set_index("Feature"))
