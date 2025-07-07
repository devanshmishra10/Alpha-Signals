import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Alpha Signals Dashboard", layout="wide")
st.title("📈 Alpha Signals: Stock Movement Prediction")

st.markdown("""
This dashboard presents the results of a machine learning model designed to predict the **next-day movement** of a stock based on technical indicators.
The key goal was to **outperform a random prediction baseline** in both accuracy and Sharpe Ratio.
""")

# Sample results from your notebook
results = {
    'Logistic Regression': {'Accuracy': 0.532, 'Sharpe Ratio': 0.58},
    'Random Forest': {'Accuracy': 0.551, 'Sharpe Ratio': 0.93},
    'XGBoost': {'Accuracy': 0.574, 'Sharpe Ratio': 1.12},
    'Random Baseline': {'Accuracy': None, 'Sharpe Ratio': 0.02}
}

results_df = pd.DataFrame(results).T
st.subheader("📊 Model Performance Comparison")
st.dataframe(results_df.style.format({"Accuracy": "{:.2%}", "Sharpe Ratio": "{:.2f}"}))

# Bar plot for Sharpe Ratios
st.subheader("📈 Sharpe Ratio Comparison")
fig, ax = plt.subplots(figsize=(8, 4))
sharpe_values = results_df['Sharpe Ratio'].dropna()
sharpe_values.sort_values().plot(kind='barh', ax=ax, color=['grey', 'skyblue', 'dodgerblue', 'orange'])
ax.set_xlabel("Sharpe Ratio")
ax.set_title("Risk-Adjusted Return by Model")
st.pyplot(fig)

# Optional: Add commentary
st.markdown("""
- The **XGBoost** model outperformed other approaches with the highest **Sharpe Ratio of 1.12**.
- Even a small increase in accuracy (**57.4% vs ~50%**) leads to **significant outperformance** when applied to trading strategies.
- **Random predictions** perform near-zero Sharpe, emphasizing that our model adds **real predictive value**.
""")

# Add an expandable section with explanation
with st.expander("Why is Sharpe Ratio More Important than Accuracy?"):
    st.markdown("""
    In financial markets, the goal is not just to be right more often, but to be **right when it matters** —
    during volatile or high-return periods. **Sharpe Ratio** captures the **risk-adjusted return**, reflecting
    how consistent and profitable a strategy is.

    A model with 57% accuracy but good Sharpe is more valuable than one with 60% accuracy and poor return consistency.
    """)
