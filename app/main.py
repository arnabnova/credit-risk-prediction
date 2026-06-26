"""
app.py
------
Streamlit UI for the Loan Default Risk Predictor.
All prediction logic lives in prediction_helper.py.
"""

import streamlit as st
from Prediction_Helper import (
    load_model,
    build_input_df,
    preprocess,
    predict,
    get_risk_level,
    get_credit_rating,
    get_decision,
)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Metric cards */
    [data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 700; }
    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1f4e79;
        border-bottom: 2px solid #d0e4f7;
        padding-bottom: 4px;
        margin-bottom: 12px;
    }
    /* Badge pill */
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model (cached) ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def get_model():
    return load_model()

model_data = get_model()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=72)
    st.title("🏦 Loan Risk Predictor")
    st.divider()

    with st.expander("📌 About Project", expanded=True):
        st.markdown("""
        This tool predicts the **probability of loan default** for a given
        applicant profile using a trained machine-learning model.

        It helps credit officers make faster, data-driven decisions
        and estimate an indicative credit score.
        """)

    with st.expander("🤖 Model Used"):
        st.markdown("""
        **Algorithm:** Logistic Regression  
        **Scaling:** StandardScaler (selected numerical features)  
        **Output:** Default probability via `predict_proba()`  
        **Framework:** scikit-learn
        """)

    with st.expander("📊 Dataset Summary"):
        st.markdown("""
        | Property | Value |
        |---|---|
        | Target variable | Loan Default (0/1) |
        | Features used | 13 (post-encoding) |
        | Categorical cols | Residence, Purpose, Type |
        | Numeric cols | Age, Tenure, Ratios … |
        """)

    st.divider()
    st.caption("Built with ❤️ using Streamlit")

# ── Main title ─────────────────────────────────────────────────────────────────

st.title("🏦 Loan Default Risk Predictor")
st.markdown(
    "Fill in the applicant details below and click **Predict Risk** "
    "to get an instant assessment."
)
st.divider()

# ── Input form ─────────────────────────────────────────────────────────────────

with st.form("input_form"):
    st.markdown('<p class="section-header">📋 Applicant & Loan Details</p>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Numerical Inputs**")
        age = st.number_input(
            "Age (years)", min_value=18, max_value=80, value=35, step=1
        )
        loan_tenure_months = st.number_input(
            "Loan Tenure (months)", min_value=6, max_value=360, value=60, step=6
        )
        number_of_open_accounts = st.number_input(
            "Number of Open Accounts", min_value=0, max_value=30, value=3, step=1
        )
        credit_utilization_ratio = st.slider(
            "Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0,
            value=30.0, step=0.5,
            help="Percentage of available credit currently in use."
        )

    with col2:
        st.markdown("**Risk Metrics**")
        loan_income_ratio = st.number_input(
            "Loan-to-Income Ratio", min_value=0.0, max_value=50.0,
            value=3.0, step=0.1, format="%.2f",
            help="Total loan amount divided by annual income."
        )
        delinquency_ratio = st.slider(
            "Delinquency Ratio", min_value=0.0, max_value=1.0,
            value=0.10, step=0.01,
            help="Fraction of past payments that were delinquent."
        )
        avg_dpd_per_delinquency = st.number_input(
            "Average DPD per Delinquency", min_value=0, max_value=200,
            value=0, step=1,
            help="Average Days Past Due for delinquent accounts."
        )

    with col3:
        st.markdown("**Categorical Inputs**")
        residence_type = st.selectbox(
            "Residence Type",
            options=["Owned", "Rented", "Mortgage"],
            index=0,
        )
        loan_purpose = st.selectbox(
            "Loan Purpose",
            options=["Home", "Education", "Personal", "Auto"],
            index=0,
        )
        loan_type = st.selectbox(
            "Loan Type",
            options=["Secured", "Unsecured"],
            index=0,
        )

    st.divider()
    submitted = st.form_submit_button(
        "🔍 Predict Risk", use_container_width=True, type="primary"
    )

# ── Prediction & output ────────────────────────────────────────────────────────

if submitted:
    # ── Preprocessing ──────────────────────────────────────────────────────
    df = build_input_df(
        age=age,
        loan_tenure_months=loan_tenure_months,
        number_of_open_accounts=number_of_open_accounts,
        credit_utilization_ratio=credit_utilization_ratio,
        loan_income_ratio=loan_income_ratio,
        delinquency_ratio=delinquency_ratio,
        avg_dpd_per_delinquency=avg_dpd_per_delinquency,
        residence_type=residence_type,
        loan_purpose=loan_purpose,
        loan_type=loan_type,
    )
    X = preprocess(df, model_data)
    probability, credit_score = predict(X, model_data)

    # ── Derived labels ─────────────────────────────────────────────────────
    risk_label, risk_card   = get_risk_level(probability)
    rating_label, rating_color = get_credit_rating(credit_score)
    decision_text, dec_card = get_decision(probability)

    st.divider()
    st.subheader("📊 Prediction Results")

    # ── Row 1 – Key metrics ────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            label="📉 Default Probability",
            value=f"{probability * 100:.2f} %",
        )
        st.progress(probability, text=f"{probability * 100:.1f}% chance of default")

    with m2:
        st.metric(
            label="💳 Estimated Credit Score",
            value=str(credit_score),
        )
        score_norm = (credit_score - 300) / 550          # normalise 300-850 → 0-1
        st.progress(score_norm, text=f"Score: {credit_score} / 850")

    with m3:
        st.metric(label="⚠️ Risk Level", value=risk_label)
        # Color-coded decision badge
        badge_color = "#2ecc71" if dec_card == "success" else "#e74c3c"
        st.markdown(
            f'<span class="badge" style="background:{badge_color}">'
            f'{decision_text}</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Row 2 – Risk & credit rating cards ────────────────────────────────
    r1, r2 = st.columns(2)

    with r1:
        st.markdown('<p class="section-header">🚦 Risk Assessment</p>',
                    unsafe_allow_html=True)
        msg = (
            f"**{risk_label}**  \n"
            f"Default probability: **{probability * 100:.2f}%**"
        )
        getattr(st, risk_card)(msg)

        with st.expander("ℹ️ What does risk level mean?"):
            st.markdown("""
            | Range | Level |
            |---|---|
            | 0 – 20 % | 🟢 Low Risk |
            | 20 – 50 % | 🟡 Moderate Risk |
            | 50 – 75 % | 🟠 High Risk |
            | > 75 % | 🔴 Very High Risk |
            """)

    with r2:
        st.markdown('<p class="section-header">💎 Credit Rating</p>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<span class="badge" style="background:{rating_color};font-size:1rem;">'
            f'  {rating_label} &nbsp;({credit_score})  </span>',
            unsafe_allow_html=True,
        )
        st.markdown("&nbsp;")
        with st.expander("ℹ️ Credit rating scale"):
            st.markdown("""
            | Score Range | Rating |
            |---|---|
            | 800 – 850 | Excellent |
            | 740 – 799 | Very Good |
            | 670 – 739 | Good |
            | 580 – 669 | Fair |
            | 300 – 579 | Poor |
            """)

    st.divider()

    # ── Row 3 – Final decision banner ──────────────────────────────────────
    st.markdown('<p class="section-header">🏛️ Final Decision</p>',
                unsafe_allow_html=True)
    getattr(st, dec_card)(f"### {decision_text}")

    # ── Row 4 – Input summary (expander) ──────────────────────────────────
    with st.expander("🔎 View Input Summary"):
        summary = {
            "Age": age,
            "Loan Tenure (months)": loan_tenure_months,
            "Open Accounts": number_of_open_accounts,
            "Credit Utilization (%)": credit_utilization_ratio,
            "Loan-to-Income Ratio": loan_income_ratio,
            "Delinquency Ratio": delinquency_ratio,
            "Avg DPD per Delinquency": avg_dpd_per_delinquency,
            "Residence Type": residence_type,
            "Loan Purpose": loan_purpose,
            "Loan Type": loan_type,
        }
        import pandas as pd
        summary_df = pd.DataFrame(summary.items(), columns=["Feature", "Value"]).astype(str)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

else:
    # Placeholder before prediction
    st.info(
        "👆 Fill in the applicant details in the form above and click "
        "**Predict Risk** to see the analysis.",
        icon="ℹ️",
    )