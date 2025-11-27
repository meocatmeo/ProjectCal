import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. CẤU HÌNH HỆ THỐNG
# ==========================================
st.set_page_config(layout="wide", page_title="Risk AI V27 (Sigmoid Scaling)", page_icon="📈", initial_sidebar_state="expanded")
EX_RATE = 26000.0 
FEATURES = [
    'no_of_dependents', 'education', 'self_employed', 'income_annum', 
    'loan_amount', 'loan_term', 'cibil_score', 
    'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
]
FEATURE_VN = [
    'Người phụ thuộc', 'Học vấn', 'Tự kinh doanh', 'Thu nhập', 
    'Số tiền vay', 'Kỳ hạn', 'Điểm CIBIL', 
    'BĐS Nhà', 'BĐS TMại', 'Tài sản Lux', 'Tiền mặt/Bank'
]

if 'theme' not in st.session_state: st.session_state.theme = 'dark'
is_dark = st.session_state.theme == 'dark'

st.markdown(f"""
<style>
    .stApp {{ background-color: {'#0b0e11' if is_dark else '#f8fafc'}; color: {'#f0f2f6' if is_dark else '#333'}; }}
    .kpi-card {{
        background: {'#161b22' if is_dark else 'white'}; padding: 15px; border-radius: 10px;
        border: 1px solid {'#30363d' if is_dark else '#ddd'};
        border-left: 4px solid #3b82f6; box-shadow: 0 4px 10px rgba(0,0,0,0.2); margin-bottom: 10px;
    }}
    .kpi-val {{ font-size: 26px; font-weight: 800; margin: 5px 0; color: #fff; font-family: 'Courier New'; }}
    .kpi-lbl {{ color: #8b949e; font-size: 11px; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; }}
    div[data-testid="stNumberInput"] input {{ color: #f59e0b !important; font-weight: bold; }}
</style>
""", unsafe_allow_html=True)

def fmt(x): return f"{x:,.0f}"

# === HÀM SIGMOID QUYẾT ĐỊNH HỆ SỐ TÀI SẢN ===
def get_asset_multiplier(cibil):
    # Midpoint 660 để đạt 1.25x. Steepness 50 để curve mượt từ 300->900
    return 0.5 + 1.5 / (1 + np.exp(-(cibil - 660) / 50))

def calc_emi(p, r, n):
    try:
        rm = r/100.0
        if rm <= 1e-9: return p/n
        return (p * rm * (1+rm)**n)/((1+rm)**n - 1)
    except: return 0

def banking_limits(cibil, inc_yr_usd, asset_usd, term_mo):
    # 1. Hạn mức theo Thu nhập (Unsecured Logic - V26)
    cap_inc = (inc_yr_usd / 12.0 * 15.0) # Tối đa 15 lần lương (như đã fix ở V26)
    
    # 2. Hạn mức theo Tài sản (Sigmoid Logic - V27)
    sigmoid_factor = get_asset_multiplier(cibil)
    cap_ass = asset_usd * sigmoid_factor
    
    # Lấy cái lớn hơn (Nếu có thế chấp thì ưu tiên thế chấp, nếu không thì xét tín chấp)
    base_limit = max(cap_inc, cap_ass)
    
    return base_limit, sigmoid_factor

# ==========================================
# 1. CLASS MANUAL LOGISTIC
# ==========================================
class HandMadeLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, lambda_reg=0.1):
        self.lr = learning_rate; self.epochs = epochs; self.lambda_reg = lambda_reg
        self.w = None; self.b = None; self.cost_history = []

    def sigmoid(self, z): return 1 / (1 + np.exp(-z)) 

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n); self.b = 0; y = y.ravel()
        for i in range(self.epochs):
            z = np.dot(X, self.w) + self.b; a = self.sigmoid(z)
            epsilon = 1e-15
            loss = -np.mean(y * np.log(a + epsilon) + (1 - y) * np.log(1 - a + epsilon)) + (self.lambda_reg / (2 * m)) * np.sum(self.w ** 2)
            self.cost_history.append(loss)
            dw = (1 / m) * np.dot(X.T, (a - y)) + (self.lambda_reg / m) * self.w
            db = (1 / m) * np.sum(a - y)
            self.w -= self.lr * dw; self.b -= self.lr * db

    def predict_proba(self, X): return self.sigmoid(np.dot(X, self.w) + self.b)
    def get_feature_importance(self): return np.abs(self.w)

# ==========================================
# 2. TRAINING ENGINE (SIGMOID LOGIC INJECTED)
# ==========================================
@st.cache_resource
def train_sigmoid_core():
    np.random.seed(42)
    dfs = []
    
    # --- A. NHÓM THẾ CHẤP (ÁP DỤNG SIGMOID) ---
    def generate_secured_sigmoid(size):
        df = pd.DataFrame()
        df['income_annum'] = np.random.uniform(5000, 50000, size) * EX_RATE
        
        # Tạo tài sản đa dạng
        df['residential_assets_value'] = np.random.uniform(10000, 500000, size) * EX_RATE
        df['commercial_assets_value'] = np.random.uniform(0, 100000, size) * EX_RATE
        df['luxury_assets_value'] = np.random.uniform(0, 50000, size) * EX_RATE
        df['bank_asset_value'] = np.random.uniform(0, 50000, size) * EX_RATE
        
        df['cibil_score'] = np.random.randint(300, 900, size)
        df['loan_term'] = np.random.randint(24, 180, size) / 12
        df['no_of_dependents'] = np.random.randint(0, 4, size)
        df['education'] = np.random.randint(0, 2, size)
        df['self_employed'] = np.random.randint(0, 2, size)
        
        total_assets = df['residential_assets_value'] + df['commercial_assets_value'] + df['luxury_assets_value'] + df['bank_asset_value']
        
        # === ÁP DỤNG CÔNG THỨC SIGMOID VÀO VIỆC SINH DỮ LIỆU ===
        # Tính hệ số nhân dựa trên CIBIL từng người
        multipliers = 0.5 + 1.5 / (1 + np.exp(-(df['cibil_score'] - 660) / 50))
        
        # Hạn mức tối đa cho phép
        max_loan_limit = total_assets * multipliers
        
        # Sinh khoản vay ngẫu nhiên quanh ngưỡng này
        # Nếu vay <= max_limit * 0.9 -> Duyệt (Safe)
        # Nếu vay > max_limit * 1.1 -> Rớt (Risky)
        # Vùng giữa -> Hên xui
        
        df['loan_amount'] = max_loan_limit * np.random.uniform(0.5, 1.5, size)
        
        # Logic Labeling
        ratio = df['loan_amount'] / max_loan_limit
        df['loan_status'] = (ratio < 1.0).astype(int)
        
        return df[FEATURES + ['loan_status']]

    # --- B. NHÓM TÍN CHẤP (GIỮ LOGIC V26 CHO BẠN) ---
    def generate_unsecured_v26(size):
        df = pd.DataFrame()
        df['income_annum'] = np.random.uniform(20000000, 100000000, size) * 12 
        df['residential_assets_value'] = 0; df['commercial_assets_value'] = 0
        df['luxury_assets_value'] = 0; df['bank_asset_value'] = 0 
        df['cibil_score'] = np.random.randint(600, 900, size)
        df['loan_term'] = np.random.randint(12, 60, size) / 12
        df['no_of_dependents'] = np.random.randint(0, 3, size)
        df['education'] = np.random.randint(0, 2, size); df['self_employed'] = np.random.randint(0, 2, size)
        
        monthly_income = df['income_annum'] / 12
        df['loan_amount'] = monthly_income * np.random.uniform(3.0, 12.0, size)
        
        rate = 0.12 / 12
        term_mo = df['loan_term'] * 12
        emi = (df['loan_amount'] * rate * (1+rate)**term_mo) / ((1+rate)**term_mo - 1)
        dti = emi / monthly_income
        
        score = (1 - np.clip(dti, 0, 1)) * 0.6 + ((df['cibil_score']-300)/600) * 0.4
        df['loan_status'] = (score > 0.50).astype(int)
        return df[FEATURES + ['loan_status']]

    dfs.append(generate_secured_sigmoid(6000)) # 6k mẫu học Sigmoid
    dfs.append(generate_unsecured_v26(4000))   # 4k mẫu học Tín chấp

    try:
        r = pd.read_csv("Loan.csv"); r.columns = r.columns.str.strip()
        r['education'] = r['education'].apply(lambda x: 1 if 'Grad' in str(x) else 0)
        r['self_employed'] = r['self_employed'].apply(lambda x: 1 if 'Yes' in str(x) else 0)
        r['loan_status'] = r['loan_status'].apply(lambda x: 1 if 'App' in str(x) else 0)
        dfs.append(r[FEATURES + ['loan_status']])
    except: pass

    FULL = pd.concat(dfs, ignore_index=True).fillna(0)
    
    X = FULL[FEATURES].values; y = FULL['loan_status'].values.reshape(-1,1)
    scl = StandardScaler(); X_s = scl.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.2, random_state=42)

    # --- TRAIN 4 MODELS ---
    # Neural Net
    model_nn = nn.Sequential(
        nn.Linear(11, 64), nn.SiLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.SiLU(), 
        nn.Linear(32, 1), nn.Sigmoid()
    )
    opt = optim.AdamW(model_nn.parameters(), lr=0.005)
    for _ in range(250):
        opt.zero_grad(); p = model_nn(torch.tensor(X_train, dtype=torch.float32))
        loss = nn.BCELoss()(p, torch.tensor(y_train, dtype=torch.float32)); loss.backward(); opt.step()

    # Random Forest (Depth vừa phải để học curve sigmoid mà không overfit noise)
    model_rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=8, random_state=42)
    model_rf.fit(X_train, y_train.ravel())

    # G-Boosting
    model_gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=42)
    model_gb.fit(X_train, y_train.ravel())

    # Manual Logic
    model_manual = HandMadeLogisticRegression(learning_rate=0.05, epochs=2000, lambda_reg=0.2)
    model_manual.fit(X_train, y_train)

    # Feature Importance
    imp_rf = model_rf.feature_importances_
    imp_gb = model_gb.feature_importances_
    imp_man = model_manual.get_feature_importance(); imp_man /= np.sum(imp_man)
    final_importance = (imp_rf + imp_gb + imp_man) / 3
        
    return scl, model_nn, model_rf, model_gb, model_manual, len(FULL), X_test, y_test, final_importance

SCALER, NET, RF, GB, MAN, DATA_LEN, X_TE, Y_TE, F_IMP = train_sigmoid_core()

# ==========================================
# 3. UI & CALCULATION
# ==========================================
with st.sidebar:
    st.header("📝 THẨM ĐỊNH (SIGMOID)")
    st.info(f"Curve: 300(0.5x) -> 660(1.25x) -> 900(2.0x)")
    i_cibil = st.slider("CIBIL", 300, 900, 750)
    i_inc_mo = st.number_input("Thu nhập Tháng (VNĐ)", 0.0, step=1e7, value=50000000.0, format="%.0f")
    
    with st.expander("Tài sản (Để trống = 0)"):
        ra = st.number_input("BĐS Nhà", 0.0, step=1e9, format="%.0f", value=0.0)
        ca = st.number_input("Kinh doanh", 0.0, step=1e9, format="%.0f", value=0.0)
        la = st.number_input("Xe/Lux", 0.0, step=1e8, format="%.0f", value=0.0)
        ba = st.number_input("Bank/CK", 0.0, step=1e8, format="%.0f", value=0.0)
    i_tot = ra+ca+la+ba
    
    st.markdown("---")
    i_loan = st.number_input("Số Tiền Vay (VNĐ)", 0.0, step=1e8, value=200000000.0, format="%.0f")
    i_term = st.number_input("Thời hạn (Tháng)", 1, 480, 36)
    i_rate = st.number_input("Lãi suất (%)", 0.0, 20.0, 10.0)
    btn = st.button("KÍCH HOẠT HỆ THỐNG", type="primary", use_container_width=True)

# LOGIC CALCULATOR
if i_loan<=0: i_loan=1
inc_yr = i_inc_mo * 12
lim_usd, sig_factor = banking_limits(i_cibil, (inc_yr/EX_RATE), (i_tot/EX_RATE), i_term)
limit_vnd = lim_usd * EX_RATE
emi = calc_emi(i_loan, i_rate, i_term)
dti = emi / (i_inc_mo + 1)

# PREDICT
v_raw = [[2, 1, 0, inc_yr/EX_RATE, i_loan/EX_RATE, i_term/12.0, i_cibil, ra/EX_RATE, ca/EX_RATE, la/EX_RATE, ba/EX_RATE]]
v = SCALER.transform(v_raw)

with torch.no_grad(): score_nn = NET(torch.tensor(v, dtype=torch.float32)).item()
score_rf = RF.predict_proba(v)[0, 1]
score_gb = GB.predict_proba(v)[0, 1]
score_man = MAN.predict_proba(v)[0]

final_score = (score_nn + score_rf + score_gb + score_man) / 4

is_ok=True; reasons=[]
# Logic duyệt
if i_loan > limit_vnd*1.1: is_ok=False; final_score*=0.4; reasons.append(f"Vượt hạn mức ({fmt(limit_vnd)})")
if dti > 0.65: is_ok=False; final_score*=0.5; reasons.append(f"DTI quá cao ({dti*100:.1f}%)")
if final_score < 0.5: is_ok=False

col = "#238636" if is_ok else "#da3633"
txt = "DUYỆT" if is_ok else "TỪ CHỐI"

# ==========================================
# 4. DASHBOARD
# ==========================================
if btn and is_ok:
    st.balloons()
st.markdown(f"<h1 style='text-align:center;color:{col}'>{txt} ({final_score*100:.1f}%)</h1>", unsafe_allow_html=True)

m1,m2,m3,m4 = st.columns(4)
def kpi(c,l,v,s,co): c.markdown(f"<div class='kpi-card' style='border-left-color:{co}'><div class='kpi-lbl'>{l}</div><div class='kpi-val' style='color:{'#fff' if is_dark else '#333'}'>{v}</div><small style='color:#aaa'>{s}</small></div>", unsafe_allow_html=True)

kpi(m1, "SỐ TIỀN", fmt(i_loan), "VNĐ", "#3b82f6")
kpi(m2, "LOẠI HÌNH", "TÍN CHẤP" if i_tot==0 else "THẾ CHẤP", f"Hệ số CIBIL: x{sig_factor:.2f}", "#a371f7")
kpi(m3, "EMI/THÁNG", fmt(emi), f"DTI: {dti*100:.1f}%", col)
kpi(m4, "AI TRUST", f"{final_score*100:.1f}", "Sigmoid V27", col)

st.write("")
t1, t2 = st.tabs(["📉 BIỂU ĐỒ SIGMOID & FEATURE", "⚙️ CHI TIẾT MODEL"])

with t1:
    c_left, c_right = st.columns(2)
    with c_left:
        # VẼ CURVE SIGMOID ĐỂ USER THẤY TRỰC QUAN
        x_vals = np.linspace(300, 900, 100)
        y_vals = get_asset_multiplier(x_vals)
        
        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Multiplier Curve', line=dict(color='#3b82f6', width=3)))
        # Điểm hiện tại của User
        fig_sig.add_trace(go.Scatter(x=[i_cibil], y=[sig_factor], mode='markers', name='You are here', marker=dict(color='red', size=12)))
        
        fig_sig.update_layout(title="Hệ số nhân Tài sản theo CIBIL (Sigmoid)", xaxis_title="CIBIL Score", yaxis_title="Multiplier (x lần)", template="plotly_dark")
        st.plotly_chart(fig_sig, use_container_width=True)
        st.caption("Công thức: 0.5 + 1.5 / (1 + exp(-(CIBIL - 660)/50))")
        
    with c_right:
        fi_df = pd.DataFrame({'Feature': FEATURE_VN, 'Importance': F_IMP})
        fi_df = fi_df.sort_values(by='Importance', ascending=True)
        fig_imp = px.bar(fi_df, x='Importance', y='Feature', orientation='h', 
                        title="Yếu tố quan trọng", color='Importance', color_continuous_scale='Blues')
        st.plotly_chart(fig_imp, use_container_width=True)

with t2:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1. Neural Net", f"{score_nn*100:.1f}%")
    c2.metric("2. Random Forest", f"{score_rf*100:.1f}%")
    c3.metric("3. G-Boosting", f"{score_gb*100:.1f}%")
    c4.metric("4. Manual Code", f"{score_man*100:.1f}%")
    
    X_te_torch = torch.tensor(X_TE, dtype=torch.float32)
    with torch.no_grad(): prob_nn = NET(X_te_torch).numpy().flatten()
    prob_rf = RF.predict_proba(X_TE)[:, 1]
    prob_gb = GB.predict_proba(X_TE)[:, 1]
    prob_man = MAN.predict_proba(X_TE)
    
    prob_final = (prob_nn + prob_rf + prob_gb + prob_man) / 4
    pred_final = (prob_final > 0.5).astype(int)
    cm = confusion_matrix(Y_TE, pred_final)
    acc = accuracy_score(Y_TE, pred_final)

    k_acc, k_cm = st.columns(2)
    with k_acc:
        st.metric("Test Accuracy", f"{acc*100:.2f}%")
    with k_cm:
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Mint', 
                           x=['Từ chối','Đồng ý'], y=['Từ chối','Đồng ý'], title="Ma trận nhầm lẫn")
        st.plotly_chart(fig_cm, use_container_width=True)

if btn:
    @st.dialog("KẾT LUẬN")
    def show():
        st.header(txt)
        if is_ok:
            st.success("Duyệt thành công.")
            if i_tot > 0:
                st.write(f"Hệ số thế chấp CIBIL của bạn là: **x{sig_factor:.2f}**")
        else:
            st.error("Từ chối.")
            for m in reasons: st.write(f"- {m}")
        st.divider()
        st.write(f"Điểm tin cậy: **{final_score*100:.1f}**")
    show()
