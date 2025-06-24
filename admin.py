import streamlit as st
import pandas as pd
import datetime

st.set_page_config(page_title="Client Usage Admin Dashboard", layout="wide")
st.title("📊 Client Usage Tracker")

# --- Load Usage Logs ---
try:
    df = pd.read_csv("usage_log.csv", names=["timestamp", "client_id", "source", "transactions"], parse_dates=["timestamp"])
    df["transactions"] = df["transactions"].astype(int)
except FileNotFoundError:
    st.warning("No usage logs found yet.")
    st.stop()

# --- Filter by Month ---
df["month"] = df["timestamp"].dt.to_period("M")
months = df["month"].astype(str).unique()
selected_month = st.selectbox("Select Month", sorted(months, reverse=True))

monthly_df = df[df["month"].astype(str) == selected_month]

# --- Summary ---
st.subheader(f"📆 Summary for {selected_month}")
summary = monthly_df.groupby("client_id")["transactions"].sum().reset_index()
summary.columns = ["Client", "Total Transactions"]
summary["Over Quota"] = summary["Total Transactions"] > 5000
st.dataframe(summary.sort_values("Total Transactions", ascending=False), use_container_width=True)

# --- Detailed Log Viewer ---
st.subheader("🧾 Raw Usage Log")
st.dataframe(monthly_df.sort_values("timestamp", ascending=False), use_container_width=True)
