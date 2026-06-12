import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(
    page_title="Churn Risk Predictor",
    page_icon="🎯",
    layout="wide"
)

# --- Styling ---
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    color: #1f77b4;
    margin-bottom: 0.5rem;
}
.sub-header {
    font-size: 1rem;
    text-align: center;
    color: #666;
    margin-bottom: 2rem;
}
.risk-high {
    background: #ffebee;
    border-left: 5px solid #f44336;
    padding: 15px;
    border-radius: 8px;
    font-size: 1.2rem;
    font-weight: bold;
    color: #c62828;
}
.risk-medium {
    background: #fff8e1;
    border-left: 5px solid #ff9800;
    padding: 15px;
    border-radius: 8px;
    font-size: 1.2rem;
    font-weight: bold;
    color: #e65100;
}
.risk-low {
    background: #e8f5e9;
    border-left: 5px solid #4caf50;
    padding: 15px;
    border-radius: 8px;
    font-size: 1.2rem;
    font-weight: bold;
    color: #1b5e20;
}
.recommendation-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 12px;
    margin: 5px 0;
    border-left: 3px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header">🎯 Smart Churn Risk Predictor</div>',
            unsafe_allow_html=True)
st.markdown('''<div class="sub-header">Enter customer details to predict
            churn probability and get personalized retention strategies</div>''',
            unsafe_allow_html=True)

# --- Build Model ---
@st.cache_resource
def build_model():
    np.random.seed(42)
    n = 5000

    # Generate synthetic training data
    # based on real telco churn patterns
    tenure = np.random.exponential(30, n).clip(1, 72).astype(int)
    monthly_charges = np.random.normal(65, 25, n).clip(20, 120)
    contract = np.random.choice([0, 1, 2], n, p=[0.55, 0.25, 0.20])
    internet_service = np.random.choice([0, 1, 2], n, p=[0.22, 0.44, 0.34])
    payment_method = np.random.choice([0, 1, 2, 3], n, p=[0.34, 0.22, 0.22, 0.22])
    online_security = np.random.choice([0, 1], n, p=[0.50, 0.50])
    tech_support = np.random.choice([0, 1], n, p=[0.50, 0.50])
    senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
    dependents = np.random.choice([0, 1], n, p=[0.70, 0.30])
    partner = np.random.choice([0, 1], n, p=[0.52, 0.48])
    paperless_billing = np.random.choice([0, 1], n, p=[0.41, 0.59])
    multiple_lines = np.random.choice([0, 1], n, p=[0.52, 0.48])
    streaming_tv = np.random.choice([0, 1], n, p=[0.50, 0.50])
    streaming_movies = np.random.choice([0, 1], n, p=[0.50, 0.50])
    num_services = (online_security + tech_support +
                   streaming_tv + streaming_movies + multiple_lines)

    # Churn probability based on real patterns
    churn_prob = (
        0.45 * (contract == 0) +
        0.20 * (monthly_charges > 75) / 1 +
        0.15 * (tenure < 12) / 1 +
        0.10 * (internet_service == 1) +
        0.08 * (payment_method == 0) +
        0.07 * (paperless_billing == 1) +
        0.06 * (senior_citizen == 1) -
        0.15 * (online_security == 1) -
        0.15 * (tech_support == 1) -
        0.10 * (contract == 2) -
        0.08 * (num_services > 3) -
        0.05 * (dependents == 1) -
        0.05 * (partner == 1)
    )

    churn_prob = np.clip(churn_prob, 0.02, 0.98)
    churn = (np.random.random(n) < churn_prob).astype(int)

    X = pd.DataFrame({
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'Contract': contract,
        'InternetService': internet_service,
        'PaymentMethod': payment_method,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support,
        'SeniorCitizen': senior_citizen,
        'Dependents': dependents,
        'Partner': partner,
        'PaperlessBilling': paperless_billing,
        'MultipleLines': multiple_lines,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'NumServices': num_services,
        'TotalCharges': tenure * monthly_charges
    })

    model = GradientBoostingClassifier(
        n_estimators=100, random_state=42)
    model.fit(X, churn)

    return model, X.columns.tolist()

# --- Load Model ---
with st.spinner("Loading prediction model..."):
    model, feature_names = build_model()

st.success("✅ Model ready!")

# --- Risk Gauge ---
def create_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Risk %", 'font': {'size': 18}},
        delta={'reference': 26.5,
               'increasing': {'color': "red"},
               'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100],
                    'tickwidth': 1,
                    'tickcolor': "darkblue"},
            'bar': {'color': "darkblue", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 60], 'color': '#fff3cd'},
                {'range': [60, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(t=60, b=0, l=30, r=30),
        font={'size': 14}
    )
    return fig

# --- Feature Importance Chart ---
def create_importance_chart(input_data, model, feature_names):
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True).tail(8)

    fig = px.bar(
        feat_imp,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top Risk Factors',
        color='Importance',
        color_continuous_scale='RdYlGn_r'
    )
    fig.update_layout(height=300,
                     margin=dict(t=40, b=0, l=0, r=0),
                     showlegend=False)
    return fig

# --- Recommendations Engine ---
def get_recommendations(inputs, probability):
    recs = []

    if probability >= 0.6:
        recs.append(("🚨 URGENT", "Assign dedicated account manager immediately"))
        recs.append(("📞 ACTION", "Schedule personal call within 24 hours"))

    if inputs['Contract'] == 0:
        recs.append(("📋 CONTRACT",
                    "Offer 20% discount to upgrade to annual contract"))
    if inputs['MonthlyCharges'] > 75:
        recs.append(("💰 PRICING",
                    "Review pricing — charges above average may be driving churn"))
    if inputs['tenure'] < 12:
        recs.append(("🤝 ONBOARDING",
                    "New customer — enroll in loyalty program immediately"))
    if inputs['OnlineSecurity'] == 0:
        recs.append(("🔒 SECURITY",
                    "Offer free 3-month OnlineSecurity trial"))
    if inputs['TechSupport'] == 0:
        recs.append(("🛠️ SUPPORT",
                    "Offer complimentary TechSupport upgrade"))
    if inputs['PaymentMethod'] == 0:
        recs.append(("💳 PAYMENT",
                    "Incentivize switch to auto-payment — 5% monthly discount"))
    if inputs['SeniorCitizen'] == 1:
        recs.append(("👴 SENIOR",
                    "Enroll in senior loyalty program with dedicated support"))
    if inputs['NumServices'] < 3:
        recs.append(("📦 BUNDLE",
                    "Offer bundled services package at discounted rate"))
    if inputs['tenure'] > 24 and probability > 0.4:
        recs.append(("⭐ LOYALTY",
                    "Long-term customer at risk — offer VIP loyalty reward"))

    if not recs:
        recs.append(("✅ HEALTHY",
                    "Customer appears stable — maintain regular engagement"))
        recs.append(("📧 NURTURE",
                    "Send monthly satisfaction survey to maintain relationship"))

    return recs

# --- Main Layout ---
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 👤 Customer Details")

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            tenure = st.slider("Tenure (Months)", 1, 72, 12)
            monthly_charges = st.slider("Monthly Charges ($)", 20, 120, 65)
            contract = st.selectbox(
                "Contract Type",
                options=[0, 1, 2],
                format_func=lambda x: ["Month-to-Month",
                                       "One Year",
                                       "Two Year"][x])
            internet_service = st.selectbox(
                "Internet Service",
                options=[0, 1, 2],
                format_func=lambda x: ["DSL",
                                       "Fiber Optic",
                                       "No Internet"][x])
            payment_method = st.selectbox(
                "Payment Method",
                options=[0, 1, 2, 3],
                format_func=lambda x: ["Electronic Check",
                                       "Mailed Check",
                                       "Bank Transfer",
                                       "Credit Card"][x])
            senior_citizen = st.selectbox(
                "Senior Citizen",
                options=[0, 1],
                format_func=lambda x: ["No", "Yes"][x])
            paperless_billing = st.selectbox(
                "Paperless Billing",
                options=[0, 1],
                format_func=lambda x: ["No", "Yes"][x])

        with col2:
            online_security = st.selectbox(
                "Online Security",
                options=[0, 1],
                format_func=lambda x: ["No", "Yes"][x])
            tech_support = st.selectbox(
                "Tech Support",
                options=[0, 1],
                format_func=lambda x: ["No", "Yes"][x])
            partner = st.selectbox(
                "Has Partner",
                options=[0, 1],
                format_func=lambda x: ["No", "Yes"][x])
            dependents = st.selectbox(
                "Has Dependents",
                options=[0, 1],
                format_func=lambda x: ["No", "Yes"][x])
            multiple_lines = st.selectbox(
                "Multiple Lines",
                options=[0, 1],
                format_func=lambda x: ["No", "Yes"][x])
            streaming_tv = st.selectbox(
                "Streaming TV",
                options=[0, 1],
                format_func=lambda x: ["No", "Yes"][x])
            streaming_movies = st.selectbox(
                "Streaming Movies",
                options=[0, 1],
                format_func=lambda x: ["No", "Yes"][x])

    # Predict Button
    predict_btn = st.button("🔍 Predict Churn Risk",
                            use_container_width=True,
                            type="primary")

with col_right:
    st.markdown("### 📊 Prediction Results")

    if predict_btn:
        num_services = (online_security + tech_support +
                       streaming_tv + streaming_movies + multiple_lines)
        total_charges = tenure * monthly_charges

        inputs = {
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'Contract': contract,
            'InternetService': internet_service,
            'PaymentMethod': payment_method,
            'OnlineSecurity': online_security,
            'TechSupport': tech_support,
            'SeniorCitizen': senior_citizen,
            'Dependents': dependents,
            'Partner': partner,
            'PaperlessBilling': paperless_billing,
            'MultipleLines': multiple_lines,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'NumServices': num_services,
            'TotalCharges': total_charges
        }

        input_df = pd.DataFrame([inputs])
        probability = model.predict_proba(input_df)[0][1]

        # Risk Level
        if probability >= 0.6:
            risk_level = "HIGH RISK"
            risk_class = "risk-high"
            risk_emoji = "🔴"
        elif probability >= 0.3:
            risk_level = "MEDIUM RISK"
            risk_class = "risk-medium"
            risk_emoji = "🟡"
        else:
            risk_level = "LOW RISK"
            risk_class = "risk-low"
            risk_emoji = "🟢"

        # Gauge
        st.plotly_chart(create_gauge(probability),
                       use_container_width=True)

        # Risk Badge
        st.markdown(f"""
        <div class="{risk_class}">
            {risk_emoji} {risk_level} — {probability*100:.1f}% Churn Probability
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Key Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Tenure", f"{tenure} months",
                   delta=f"{'Stable' if tenure > 24 else 'New'}")
        col2.metric("Monthly Bill", f"${monthly_charges:.0f}",
                   delta=f"{'High' if monthly_charges > 75 else 'Normal'}",
                   delta_color="inverse")
        col3.metric("Services", f"{num_services}",
                   delta=f"{'Engaged' if num_services > 3 else 'Low'}")

        # Feature Importance
        st.plotly_chart(
            create_importance_chart(inputs, model, feature_names),
            use_container_width=True)

    else:
        st.info("👈 Fill in customer details and click Predict!")
        st.markdown("""
        **This app predicts:**
        - 🎯 Churn probability score
        - 🚦 Risk level (Low/Medium/High)
        - 📊 Key risk factors driving prediction
        - 💡 Personalized retention strategies
        """)

# --- Recommendations Section ---
if predict_btn:
    st.markdown("---")
    st.markdown("### 💼 Personalized Retention Recommendations")

    inputs_local = {
        'Contract': contract,
        'MonthlyCharges': monthly_charges,
        'tenure': tenure,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support,
        'PaymentMethod': payment_method,
        'SeniorCitizen': senior_citizen,
        'NumServices': (online_security + tech_support +
                       streaming_tv + streaming_movies + multiple_lines)
    }

    probability_local = model.predict_proba(
        pd.DataFrame([{
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'Contract': contract,
            'InternetService': internet_service,
            'PaymentMethod': payment_method,
            'OnlineSecurity': online_security,
            'TechSupport': tech_support,
            'SeniorCitizen': senior_citizen,
            'Dependents': dependents,
            'Partner': partner,
            'PaperlessBilling': paperless_billing,
            'MultipleLines': multiple_lines,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'NumServices': (online_security + tech_support +
                           streaming_tv + streaming_movies + multiple_lines),
            'TotalCharges': tenure * monthly_charges
        }])
    )[0][1]

    recommendations = get_recommendations(inputs_local, probability_local)

    cols = st.columns(2)
    for i, (tag, rec) in enumerate(recommendations):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="recommendation-card">
            <strong>{tag}</strong><br>{rec}
            </div>""", unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.8rem'>
Built with ❤️ using Streamlit & ML | Churn Risk Predictor v1.0
</div>""", unsafe_allow_html=True)
