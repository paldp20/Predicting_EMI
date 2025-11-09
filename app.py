import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load models and scaler
@st.cache_resource
def load_models():
    clf_path = "models/best_classifier.joblib"
    reg_path = "models/best_regressor.joblib"
    scaler_path = "artifacts/scaler.joblib"

    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)
    scaler = joblib.load(scaler_path)

    return clf, reg, scaler

clf_model, reg_model, scaler = load_models()

# Streamlit Page Setup
st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide",
)

st.title("EMIPredict AI — Intelligent Financial Risk Assessment Platform")
st.markdown("### Empowering Smarter EMI Decisions with Data-Driven Insights")

# Sidebar Input Fields
st.sidebar.header("Applicant Financial Details")

age = st.sidebar.number_input("Age", 18, 70, 30)
monthly_salary = st.sidebar.number_input("Monthly Salary (₹)", 1000, 500000, 50000)
monthly_rent = st.sidebar.number_input("Monthly Rent (₹)", 0, 100000, 10000)
groceries_utilities = st.sidebar.number_input("Groceries & Utilities (₹)", 0, 100000, 10000)
other_expenses = st.sidebar.number_input("Other Monthly Expenses (₹)", 0, 100000, 5000)
existing_loans = st.sidebar.selectbox("Existing Loans?", ["Yes", "No"])
current_emi_amount = st.sidebar.number_input("Current EMI Amount (₹)", 0, 100000, 5000)
credit_score = st.sidebar.number_input("Credit Score", 300, 900, 700)
years_of_employment = st.sidebar.number_input("Years of Employment", 0.0, 40.0, 5.0)
bank_balance = st.sidebar.number_input("Bank Balance (₹)", 0, 1000000, 100000)
emergency_fund = st.sidebar.number_input("Emergency Fund (₹)", 0, 1000000, 20000)
dependents = st.sidebar.number_input("Dependents", 0, 10, 2)

# Derived Features
total_expenses = monthly_rent + groceries_utilities + other_expenses + current_emi_amount
debt_to_income = (current_emi_amount / monthly_salary) if monthly_salary > 0 else 0
expense_to_income = (total_expenses / monthly_salary) if monthly_salary > 0 else 0
affordability_ratio = (monthly_salary - total_expenses) / monthly_salary if monthly_salary > 0 else 0

# Prepare Input for Prediction
input_data = pd.DataFrame([{
    "age": age,
    "monthly_salary": monthly_salary,
    "monthly_rent": monthly_rent,
    "groceries_utilities": groceries_utilities,
    "other_monthly_expenses": other_expenses,
    "existing_loans": 1 if existing_loans == "Yes" else 0,
    "current_emi_amount": current_emi_amount,
    "credit_score": credit_score,
    "years_of_employment": years_of_employment,
    "bank_balance": bank_balance,
    "emergency_fund": emergency_fund,
    "dependents": dependents,
    "debt_to_income": debt_to_income,
    "expense_to_income": expense_to_income,
    "affordability_ratio": affordability_ratio
}])

expected_features = getattr(scaler, "feature_names_in_", None)
if expected_features is not None:
    input_data = input_data.reindex(columns=expected_features, fill_value=0)

scaled_input = scaler.transform(input_data)

# Main Prediction Area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("EMI Eligibility Prediction")
    if st.button("Run Prediction"):
        pred_class = clf_model.predict(scaled_input)[0]
        if hasattr(clf_model, "predict_proba"):
            proba = clf_model.predict_proba(scaled_input).max()
        else:
            proba = None

        class_map = {0: "Not Eligible", 1: "Eligible", 2: "High Risk"}
        eligibility = class_map.get(pred_class, "Unknown")

        # Display results
        st.markdown(f"### Prediction: **{eligibility}**")
        if proba:
            st.progress(int(proba * 100))
            st.caption(f"Confidence: {proba*100:.2f}%")

        # Color-coded recommendation
        if eligibility == "High Risk":
            st.warning("**High Risk Applicant** — recommend additional credit evaluation.")
        elif eligibility == "Not Eligible":
            st.error("**Not Eligible** — financial profile does not currently meet EMI criteria.")
        else:
            st.success("**Eligible for EMI** — applicant shows strong financial capacity.")

with col2:
    st.subheader("Predicted Maximum EMI Amount")
    pred_emi = reg_model.predict(scaled_input)[0]
    st.metric("Maximum Affordable EMI", f"₹ {pred_emi:,.0f}")

# Financial Health Visualization
st.write("---")
st.subheader("Financial Health Overview")

health_score = (affordability_ratio * 100)
health_score = np.clip(health_score, 0, 100)

fig, ax = plt.subplots(figsize=(6, 1.5))
ax.barh(["Affordability"], [health_score], color="green" if health_score > 60 else "orange" if health_score > 30 else "red")
ax.set_xlim(0, 100)
ax.set_xlabel("Affordability Ratio (%)")
st.pyplot(fig)

st.caption(f"Financial Health Score: {health_score:.2f}% — Higher means more disposable income after expenses.")

# Detailed Insights
st.write("---")
st.subheader("Financial Insights & Ratios")
c1, c2, c3 = st.columns(3)
c1.metric("Debt-to-Income Ratio", f"{debt_to_income:.2f}")
c2.metric("Expense-to-Income Ratio", f"{expense_to_income:.2f}")
c3.metric("Affordability Ratio", f"{affordability_ratio:.2f}")

st.markdown(
    """
    **Interpretation Guide:**
    - 🟢 *Debt-to-Income Ratio* < 0.35 → Healthy  
    - 🟠 0.35–0.5 → Moderate risk  
    - 🔴 > 0.5 → High risk of EMI stress  
    """
)

# Footer
st.write("---")
st.markdown("<center>Developed by <b>Pal</b> | Powered by Streamlit + PyTorch + MLflow</center>", unsafe_allow_html=True)
