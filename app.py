import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

# Suppress warnings for cleaner production logs
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. SYSTEM CONFIGURATION & CONSTANTS
# ==============================================================================
st.set_page_config(
    layout="wide", 
    page_title="FinTrust AI | Underwriting", 
    page_icon="🏦", 
    initial_sidebar_state="expanded"
)

# Financial Constants
EXCHANGE_RATE_VND_USD = 26000.0
SEED_VALUE = 42

# Feature Definitions
FEATURE_COLUMNS = [
    'no_of_dependents', 'education', 'self_employed', 'income_annum', 
    'loan_amount', 'loan_term', 'cibil_score', 
    'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
]

# Localization Labels (Vietnamese for End-Users)
FEATURE_LABELS_VN = [
    'Người phụ thuộc', 'Học vấn', 'Tự doanh', 'Thu nhập năm', 
    'Số tiền vay', 'Kỳ hạn', 'Điểm tín dụng', 
    'BĐS Nhà ở', 'BĐS Thương mại', 'Tài sản cao cấp', 'Tiền mặt/TGNH'
]

# Theme Detection
if 'theme' not in st.session_state: st.session_state.theme = 'dark'
is_dark_mode = st.session_state.theme == 'dark'

# Custom CSS for UI
st.markdown(f"""
<style>
    .stApp {{ background-color: {'#0e1117' if is_dark_mode else '#f8fafc'}; }}
    .kpi-card {{
        background: {'#1f2937' if is_dark_mode else 'white'}; 
        padding: 24px; 
        border-radius: 12px;
        border: 1px solid {'#374151' if is_dark_mode else '#e5e7eb'};
        border-left: 5px solid #3b82f6; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        transition: transform 0.2s;
    }}
    .kpi-card:hover {{ transform: translateY(-2px); }}
    .kpi-val {{ font-size: 28px; font-weight: 800; margin: 8px 0; color: {'#f3f4f6' if is_dark_mode else '#111827'}; font-family: 'Roboto Mono', monospace; }}
    .kpi-lbl {{ color: #9ca3af; font-size: 13px; text-transform: uppercase; font-weight: 700; letter-spacing: 1.2px; }}
    .stButton>button {{ font-weight: 600; border-radius: 8px; height: 50px; text-transform: uppercase; letter-spacing: 0.5px; }}
    h1, h2, h3 {{ font-family: 'Inter', sans-serif; letter-spacing: -0.5px; }}
    div[data-testid="stMetric"] {{ background-color: {'#1f2937' if is_dark_mode else '#ffffff'}; padding: 15px; border-radius: 8px; border: 1px solid {'#374151' if is_dark_mode else '#e5e7eb'}; }}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. FINANCIAL UTILITY FUNCTIONS
# ==============================================================================
def format_currency(amount):
    """Format number to Vietnamese currency string."""
    return f"{amount:,.0f}"

def calculate_sigmoid_asset_multiplier(credit_score):
    """
    Applies Sigmoid function to scale asset value based on credit score.
    Logic: 300->0.5x (Risky), 660->1.25x (Average), 900->2.0x (Prime).
    """
    return 0.5 + 1.5 / (1 + np.exp(-(credit_score - 660) / 50))

def calculate_monthly_debt_obligation(principal, rate_percent, term_years):
    """Calculate Equated Monthly Installment (EMI) for debt service."""
    try:
        monthly_rate = rate_percent / 100 / 12
        term_months = term_years * 12
        if monthly_rate <= 1e-9: return principal / term_months
        return (principal * monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)
    except: return 0

def determine_credit_limits(credit_score, income_annual_usd, assets_usd, term_mo):
    """
    Determine maximum credit exposure based on internal risk policies.
    Strategy: Higher of (Income-based Unsecured Limit) OR (Asset-based Secured Limit).
    """
    # Policy 1: Unsecured Limit (Max 15x Monthly Income)
    limit_unsecured = (income_annual_usd / 12.0 * 15.0) 
    
    # Policy 2: Secured Limit (Assets * Risk Multiplier)
    risk_multiplier = calculate_sigmoid_asset_multiplier(credit_score)
    limit_secured = assets_usd * risk_multiplier
    
    # Final Decision: Maximum eligibility
    return max(limit_unsecured, limit_secured), risk_multiplier

# ==============================================================================
# 3. CUSTOM MODEL ARCHITECTURE
# ==============================================================================
class CustomLogisticRegression:
    """
    Proprietary implementation of Logistic Regression with:
    - L2 Regularization (Ridge)
    - Label Smoothing (Anti-overconfidence)
    - Gradient Descent Optimization
    """
    def __init__(self, learning_rate=0.01, epochs=1000, lambda_reg=0.1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z)) 

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        y = y.ravel()
        
        # Apply Label Smoothing: [0, 1] -> [0.025, 0.975]
        y_smoothed = y * 0.95 + 0.025 
        
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            
            # Compute Cost with L2 Penalty
            epsilon = 1e-15
            loss = -np.mean(y_smoothed * np.log(y_pred + epsilon) + (1 - y_smoothed) * np.log(1 - y_pred + epsilon)) \
                   + (self.lambda_reg / (2 * m)) * np.sum(self.weights ** 2)
            self.loss_history.append(loss)
            
            # Backpropagation
            dw = (1 / m) * np.dot(X.T, (y_pred - y_smoothed)) + (self.lambda_reg / m) * self.weights
            db = (1 / m) * np.sum(y_pred - y_smoothed)
            
            # Weight Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        return self._sigmoid(np.dot(X, self.weights) + self.bias)
    
    def get_feature_importance(self):
        return np.abs(self.weights)

# ==============================================================================
# 4. TRAINING PIPELINE
# ==============================================================================
@st.cache_resource
def train_risk_model():
    """
    Main training pipeline orchestrating data generation, preprocessing,
    and multi-model ensemble training.
    """
    np.random.seed(SEED_VALUE)
    data_batches = []
    
    # --- A. SYNTHETIC PROFILE GENERATOR ---
    def generate_synthetic_profiles(sample_size, profile_type='secured', noise_factor=0.05):
        df = pd.DataFrame()
        # Demographic Factors
        df['no_of_dependents'] = np.random.randint(0, 6, sample_size)
        df['education'] = np.random.randint(0, 2, sample_size) # 1: Grad, 0: Undergrad
        df['self_employed'] = np.random.randint(0, 2, sample_size)
        df['cibil_score'] = np.random.randint(300, 900, sample_size)
        df['loan_term'] = np.random.randint(12, 180, sample_size) / 12
        
        if profile_type == 'secured':
            # Secured Profile: Moderate Income, Tangible Assets
            df['income_annum'] = np.random.uniform(5000, 50000, sample_size) * EXCHANGE_RATE_VND_USD
            df['residential_assets_value'] = np.random.uniform(10000, 500000, sample_size) * EXCHANGE_RATE_VND_USD
            df['commercial_assets_value'] = np.random.uniform(0, 100000, sample_size) * EXCHANGE_RATE_VND_USD
            df['luxury_assets_value'] = np.random.uniform(0, 50000, sample_size) * EXCHANGE_RATE_VND_USD
            df['bank_asset_value'] = np.random.uniform(0, 50000, sample_size) * EXCHANGE_RATE_VND_USD
            
            total_assets = (df['residential_assets_value'] + df['commercial_assets_value'] + 
                            df['luxury_assets_value'] + df['bank_asset_value'])
            
            # Apply Sigmoid Logic
            mult = 0.5 + 1.5 / (1 + np.exp(-(df['cibil_score'] - 660) / 50))
            max_limit = total_assets * mult
            df['loan_amount'] = max_limit * np.random.uniform(0.5, 1.5, sample_size)
            df['loan_status'] = (df['loan_amount'] <= max_limit).astype(int)
        
        else: # Unsecured Profile (VIP Cashflow)
            # High Income, Zero Collateral
            df['income_annum'] = np.random.uniform(20000000, 150000000, sample_size) * 12 
            df['residential_assets_value'] = 0; df['commercial_assets_value'] = 0
            df['luxury_assets_value'] = 0; df['bank_asset_value'] = 0 
            
            monthly_income = df['income_annum'] / 12
            df['loan_amount'] = monthly_income * np.random.uniform(2.0, 15.0, sample_size)
            
            # Decision Logic: Driven by DTI & Credit Score
            rate = 0.12/12; term = df['loan_term'] * 12
            emi = (df['loan_amount'] * rate * (1+rate)**term) / ((1+rate)**term - 1)
            dti = emi / monthly_income
            
            score = (1 - np.clip(dti, 0, 1)) * 0.7 + ((df['cibil_score']-300)/600) * 0.3
            df['loan_status'] = (score > 0.50).astype(int)
        
        # Noise Injection for Robustness
        n_flip = int(sample_size * noise_factor)
        indices = np.random.choice(df.index, n_flip, replace=False)
        df.loc[indices, 'loan_status'] = 1 - df.loc[indices, 'loan_status']
        
        return df[FEATURE_COLUMNS + ['loan_status']]

    # Generate Training Batches (Balanced Class)
    data_batches.append(generate_synthetic_profiles(5000, 'secured', 0.08))
    data_batches.append(generate_synthetic_profiles(5000, 'unsecured', 0.05))

    # Ingest Legacy Data (CSV)
    try:
        legacy_data = pd.read_csv("Loan.csv")
        legacy_data.columns = legacy_data.columns.str.strip()
        legacy_data['education'] = legacy_data['education'].apply(lambda x: 1 if 'Grad' in str(x) else 0)
        legacy_data['self_employed'] = legacy_data['self_employed'].apply(lambda x: 1 if 'Yes' in str(x) else 0)
        legacy_data['loan_status'] = legacy_data['loan_status'].apply(lambda x: 1 if 'App' in str(x) else 0)
        data_batches.append(legacy_data[FEATURE_COLUMNS + ['loan_status']])
    except: pass

    # Data Pipeline
    full_dataset = pd.concat(data_batches, ignore_index=True).fillna(0)
    X = full_dataset[FEATURE_COLUMNS].values
    y = full_dataset['loan_status'].values.reshape(-1,1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=SEED_VALUE)

    # --- MODEL 1: Deep Neural Network ---
    neural_net = nn.Sequential(
        nn.Linear(11, 64), nn.SiLU(), nn.Dropout(0.3),
        nn.Linear(64, 32), nn.SiLU(), 
        nn.Linear(32, 1), nn.Sigmoid()
    )
    optimizer = optim.AdamW(neural_net.parameters(), lr=0.003, weight_decay=1e-4)
    loss_history_nn = []
    y_tensor_smooth = torch.tensor(y_train * 0.95 + 0.025, dtype=torch.float32)
    
    for _ in range(200):
        optimizer.zero_grad()
        preds = neural_net(torch.tensor(X_train, dtype=torch.float32))
        loss = nn.BCELoss()(preds, y_tensor_smooth)
        loss.backward(); optimizer.step(); loss_history_nn.append(loss.item())

    # --- MODEL 2: Random Forest ---
    random_forest = RandomForestClassifier(n_estimators=1, warm_start=True, max_depth=10, min_samples_leaf=10, random_state=SEED_VALUE)
    loss_history_rf = []
    for i in range(1, 201, 5): 
        random_forest.n_estimators = i; random_forest.fit(X_train, y_train.ravel())
        loss_history_rf.append(log_loss(y_train, random_forest.predict_proba(X_train)))

    # --- MODEL 3: Gradient Boosting (Optimized for Asset Trap) ---
    gradient_boost = GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, max_depth=8, subsample=0.8, min_samples_leaf=20, random_state=SEED_VALUE)
    gradient_boost.fit(X_train, y_train.ravel())
    loss_history_gb = gradient_boost.train_score_.tolist()

    # --- MODEL 4: Custom Logistic Regression ---
    logistic_reg = CustomLogisticRegression(learning_rate=0.05, epochs=1000, lambda_reg=0.5)
    logistic_reg.fit(X_train, y_train)
    loss_history_lr = logistic_reg.loss_history

    # --- METRICS AGGREGATION ---
    def interpolate(arr): return np.interp(np.linspace(0, len(arr)-1, 100), np.arange(len(arr)), arr)
    loss_total_ensemble = (interpolate(loss_history_nn) + interpolate(loss_history_rf) + 
                           interpolate(loss_history_gb) + interpolate(loss_history_lr)) / 4
    
    feature_importance_agg = (random_forest.feature_importances_ + gradient_boost.feature_importances_ + 
                              (logistic_reg.get_feature_importance()/np.sum(logistic_reg.get_feature_importance()))) / 3

    return (scaler, neural_net, random_forest, gradient_boost, logistic_reg, 
            X_test, y_test, feature_importance_agg, 
            loss_history_nn, loss_history_rf, loss_history_gb, loss_history_lr, loss_total_ensemble)

# Initialize System
(SCALER, MODEL_NN, MODEL_RF, MODEL_GB, MODEL_LR, 
 X_TEST, Y_TEST, FEATURE_IMPORTANCE, 
 HIST_NN, HIST_RF, HIST_GB, HIST_LR, HIST_TOTAL) = train_risk_model()

# ==============================================================================
# 5. DASHBOARD UI
# ==============================================================================
with st.sidebar:
    st.title("FINTRUST AI\nUnderwriting Console")
    st.markdown("---")
    
    st.subheader("👤 Applicant Profile")
    c1, c2 = st.columns(2)
    with c1: 
        input_dependents = st.selectbox("Dependents", [0,1,2,3,4,5], index=2, help="Number of dependents")
        input_self_employed = st.selectbox("Employment", ["Salaried", "Self-Employed"], index=0)
    with c2: 
        input_education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0)
        
    val_self_emp = 1 if input_self_employed == "Self-Employed" else 0
    val_edu = 1 if input_education == "Graduate" else 0
    
    st.markdown("---")
    st.subheader("💰 Financial Profile")
    input_cibil = st.slider("Credit Score (CIBIL)", 300, 900, 750)
    input_income = st.number_input("Monthly Income (VNĐ)", 0.0, step=1e7, value=50000000.0, format="%.0f")
    
    with st.expander("Asset Portfolio (Collateral)"):
        input_asset_res = st.number_input("Residential RE", 0.0, step=1e9, format="%.0f", value=0.0)
        input_asset_com = st.number_input("Commercial RE", 0.0, step=1e9, format="%.0f", value=0.0)
        input_asset_lux = st.number_input("Luxury Assets", 0.0, step=1e8, format="%.0f", value=0.0)
        input_asset_bank = st.number_input("Bank Deposits", 0.0, step=1e8, format="%.0f", value=0.0)
    total_assets_vnd = input_asset_res + input_asset_com + input_asset_lux + input_asset_bank
    
    st.markdown("---")
    st.subheader("📋 Loan Request")
    input_loan_amt = st.number_input("Requested Amount (VNĐ)", 0.0, step=1e8, value=200000000.0, format="%.0f")
    input_loan_term = st.number_input("Term (Months)", 1, 480, 36)
    input_rate = st.number_input("Interest Rate (% / Months)", 0.0, 25.0, 10.0)
    
    execute_btn = st.button("RUN RISK ANALYSIS", type="primary", use_container_width=True)

# ==============================================================================
# 6. DECISION ENGINE
# ==============================================================================
if input_loan_amt <= 0: input_loan_amt = 1
income_usd_annual = (input_income * 12) / EXCHANGE_RATE_VND_USD
assets_usd = total_assets_vnd / EXCHANGE_RATE_VND_USD
term_years = input_loan_term / 12

# Credit Limits & Multipliers
limit_usd, risk_multiplier = determine_credit_limits(input_cibil, income_usd_annual, assets_usd, input_loan_term)
limit_vnd = limit_usd * EXCHANGE_RATE_VND_USD

# Debt Service Metrics
monthly_emi = calculate_monthly_debt_obligation(input_loan_amt, input_rate, term_years)
dti_ratio = monthly_emi / (input_income + 1)

# Feature Vector Construction
feature_vec_raw = [[
    input_dependents, val_edu, val_self_emp, 
    income_usd_annual, input_loan_amt/EXCHANGE_RATE_VND_USD, term_years, input_cibil, 
    input_asset_res/EXCHANGE_RATE_VND_USD, input_asset_com/EXCHANGE_RATE_VND_USD, 
    input_asset_lux/EXCHANGE_RATE_VND_USD, input_asset_bank/EXCHANGE_RATE_VND_USD
]]
feature_vec_scaled = SCALER.transform(feature_vec_raw)

# Ensemble Predictions
with torch.no_grad(): prob_nn = MODEL_NN(torch.tensor(feature_vec_scaled, dtype=torch.float32)).item()
prob_rf = MODEL_RF.predict_proba(feature_vec_scaled)[0, 1]
prob_gb = MODEL_GB.predict_proba(feature_vec_scaled)[0, 1]
prob_lr = MODEL_LR.predict_proba(feature_vec_scaled)[0]

trust_score = (prob_nn + prob_rf + prob_gb + prob_lr) / 4

# Approval Logic
is_approved = True
rejection_flags = []

# Unsecured Cap Logic
max_unsecured_cap = input_income * 15 if total_assets_vnd == 0 else limit_vnd * 1.5 
if input_loan_amt > max_unsecured_cap: 
    is_approved = False; trust_score *= 0.4
    rejection_flags.append(f"Exceeds Risk Limit (Max: {format_currency(max_unsecured_cap)})")

if dti_ratio > 0.65: 
    is_approved = False; trust_score *= 0.5
    rejection_flags.append(f"High DTI Ratio ({dti_ratio*100:.1f}%)")

if trust_score < 0.5: is_approved = False

status_color = "#238636" if is_approved else "#da3633"
status_text = "APPROVED" if is_approved else "REJECTED"

# ==============================================================================
# 7. MAIN DASHBOARD
# ==============================================================================
if execute_btn and is_approved: st.balloons()

st.title("Credit Underwriting Report")
st.markdown(f"<h2 style='text-align:center; color:{status_color}; border: 2px solid {status_color}; padding: 15px; border-radius: 10px;'>{status_text} <span style='font-size:20px; color:gray'>(Confidence: {trust_score*100:.1f}%)</span></h2>", unsafe_allow_html=True)

# KPI Section
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
def render_kpi(col, label, value, sub, color):
    col.markdown(f"<div class='kpi-card' style='border-left-color:{color}'><div class='kpi-lbl'>{label}</div><div class='kpi-val'>{value}</div><small style='color:#6b7280; font-weight:bold;'>{sub}</small></div>", unsafe_allow_html=True)

loan_type = "UNSECURED" if total_assets_vnd == 0 else "SECURED"
render_kpi(kpi1, "LOAN AMOUNT", format_currency(input_loan_amt), "VNĐ", "#3b82f6")
render_kpi(kpi2, "RISK TYPE", loan_type, f"CIBIL Multiplier: x{risk_multiplier:.2f}", "#8b5cf6")
render_kpi(kpi3, "EST. EMI / MO", format_currency(monthly_emi), f"DTI: {dti_ratio*100:.1f}%", status_color)
render_kpi(kpi4, "TRUST SCORE", f"{trust_score*100:.1f}%", "Ensemble Core", status_color)

st.write("")
tab_metrics, tab_tech = st.tabs(["📉 MODEL LEARNING CURVES", "⚙️ SYSTEM VALIDATION"])

with tab_metrics:
    st.subheader("Model Convergence Tracking (Loss History)")
    st.info("Visualizing the error reduction across all 4 core engines during the training phase.")
    
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1: st.plotly_chart(px.line(y=HIST_NN, title="1. Deep Neural Net", labels={'y':'BCE Loss'}, color_discrete_sequence=['#ff7b72']), use_container_width=True)
    with r1c2: st.plotly_chart(px.line(y=HIST_GB, title="2. Gradient Boosting", labels={'y':'Deviance'}, color_discrete_sequence=['#79c0ff']), use_container_width=True)
    with r1c3: st.plotly_chart(px.line(y=HIST_RF, title="3. Random Forest", labels={'y':'Log Loss'}, color_discrete_sequence=['#d2a8ff']), use_container_width=True)
    
    r2c1, r2c2 = st.columns(2)
    with r2c1: st.plotly_chart(px.line(y=HIST_LR, title="4. Logistic Regression", labels={'y':'Cost J'}, color_discrete_sequence=['#56d364']), use_container_width=True)
    with r2c2: st.plotly_chart(px.line(y=HIST_TOTAL, title="5. TOTAL ENSEMBLE LOSS", color_discrete_sequence=['#ffffff']), use_container_width=True)

with tab_tech:
    # Model Voting
    col_votes, col_matrix = st.columns([1, 2])
    with col_votes:
        st.subheader("Model Voting")
        st.metric("Neural Network", f"{prob_nn*100:.1f}%")
        st.metric("Random Forest", f"{prob_rf*100:.1f}%")
        st.metric("Gradient Boosting", f"{prob_gb*100:.1f}%")
        st.metric("Logistic Reg.", f"{prob_lr*100:.1f}%")
    
    with col_matrix:
        # Confusion Matrix Logic
        X_test_tensor = torch.tensor(X_TEST, dtype=torch.float32)
        with torch.no_grad(): pred_nn_te = MODEL_NN(X_test_tensor).numpy().flatten()
        pred_rf_te = MODEL_RF.predict_proba(X_TEST)[:, 1]
        pred_gb_te = MODEL_GB.predict_proba(X_TEST)[:, 1]
        pred_lr_te = MODEL_LR.predict_proba(X_TEST)
        
        pred_final_prob = (pred_nn_te + pred_rf_te + pred_gb_te + pred_lr_te) / 4
        pred_final_class = (pred_final_prob > 0.5).astype(int)
        cm = confusion_matrix(Y_TEST, pred_final_class)
        acc = accuracy_score(Y_TEST, pred_final_class)
        
        c_cm, c_fi = st.columns(2)
        with c_cm:
            st.subheader("Confusion Matrix")
            st.caption(f"Test Accuracy: **{acc*100:.2f}%**")
            st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale='Mint', x=['Reject','Approve'], y=['Reject','Approve']), use_container_width=True)
        with c_fi:
            st.subheader("Risk Drivers")
            fi_df = pd.DataFrame({'Feature': FEATURE_LABELS_VN, 'Importance': FEATURE_IMPORTANCE}).sort_values(by='Importance', ascending=True)
            st.plotly_chart(px.bar(fi_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues'), use_container_width=True)

# Final Dialog
if execute_btn:
    @st.dialog("Underwriting Decision")
    def show_dialog():
        st.header(status_text)
        if is_approved:
            st.success("Loan application successfully approved.")
            st.write(f"**Applicant:** {input_education}, {input_self_employed}")
            st.write(f"**Type:** {loan_type}")
        else:
            st.error("Application declined."); [st.write(f"❌ {m}") for m in rejection_flags]
        st.divider(); st.caption(f"System ID: FinTrust | Trust Score: {trust_score*100:.1f}%")
    show_dialog()
