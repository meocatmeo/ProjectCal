import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler 


def Sigmoid(z):
    return 1 / (1 + np.exp(-z))

def Loss(Y, Y_hat):
    n = Y.shape[0]
    eps = 1e-9

    if len(Y.shape) > 1: Y = Y.flatten()
    if len(Y_hat.shape) > 1: Y_hat = Y_hat.flatten()
    
    loss = -1 / n * np.sum(Y * np.log(Y_hat + eps) + (1 - Y) * np.log(1 - Y_hat + eps))
    return loss

def GD(X, Y, Y_hat):
    n = X.shape[0]
  
    dW = 1 / n * np.dot(X.T, (Y_hat.reshape(-1, 1) - Y.reshape(-1, 1)))
    dc = 1 / n * np.sum(Y_hat - Y)
    return dW.flatten(), dc

def Train_LR(X, Y, lr=0.001, ep=10000):
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float).reshape(-1, 1)

    W = np.zeros((X.shape[1], 1))
    c = 0.0
    losses_history = []
    
    for i in range(ep):
        Z = np.dot(X, W) + c
        Y_hat = Sigmoid(Z)
        
        if (i % (ep // 100) == 0) or (i == ep - 1):
             losses_history.append(Loss(Y, Y_hat))
        
        dW, dc = GD(X, Y.flatten(), Y_hat.flatten())
        
        W -= lr * dW.reshape(-1, 1) 
        c -= lr * dc

    return W.flatten(), c, np.array(losses_history)

def Hybrid_Predict_Proba(X_input, rf_model, W, c):
    X = np.array(X_input, dtype=float)

    Z_lr = np.dot(X, W.reshape(-1, 1)) + c
    Y_hat_lr = Sigmoid(Z_lr).flatten()

    Y_hat_rf = rf_model.predict_proba(X)[:, 1]

    Y_pred_proba = 0.6 * Y_hat_lr + 0.4 * Y_hat_rf

    return Y_pred_proba

def Final_Predict(Y_prob, threshold=0.5):

    Y_class = np.where(Y_prob > threshold, 1, 0) 
    return Y_class

@st.cache_resource
def get_trained_model_results():
    try:
        fl = pd.read_csv("./Loan - Loan.csv") 
    except FileNotFoundError:
        st.error("Lỗi: Không tìm thấy file Loan - Loan.csv. Vui lòng đảm bảo file nằm cùng thư mục.")
        return None, None, None, None, None, None, None, None, None, None 
        
    fl['education'] = fl['education'].map({'Graduate': 1, 'Not Graduate': 0})
    fl['self_employed'] = fl['self_employed'].map({'Yes': 1, 'No': 0})
    fl['loan_status'] = fl['loan_status'].map({'Approved': 1, 'Rejected': 0})
    
    FEATURE_NAMES = list(fl.columns[1:-1])
    X_data = fl[FEATURE_NAMES].values.astype(float)
    Y_data = fl['loan_status'].values.astype(int)

    scaler = StandardScaler()
    X_data_scaled = scaler.fit_transform(X_data) 

    X_train, _, Y_train, _ = train_test_split(
        X_data_scaled, Y_data, test_size=0.4, random_state=11, stratify=Y_data
    )

    rf_uncalibrated = RandomForestClassifier(n_estimators=1000, random_state=11, n_jobs=-1)
    rf_model = CalibratedClassifierCV(rf_uncalibrated, method="sigmoid", cv=5)
    rf_model.fit(X_train, Y_train)

    W_lr, c_lr, losses_history = Train_LR(X_data_scaled, Y_data, ep=10000)
 
    Y_prob_full = Hybrid_Predict_Proba(X_data_scaled, rf_model, W_lr, c_lr)
    Y_pred_full = Final_Predict(Y_prob_full)
    
    return FEATURE_NAMES, X_data_scaled, Y_data, W_lr, c_lr, losses_history, Y_pred_full, Y_prob_full, rf_model, scaler

FEATURE_NAMES, x, y, w, b, losses, y_pred, y_prob, rf_model, scaler = get_trained_model_results()

if FEATURE_NAMES is None:
    st.stop()

FEATURE_VIETMAP = {
    "no_of_dependents": "Số người phụ thuộc", "education": "Trình độ học vấn", 
    "self_employed": "Tự kinh doanh", "income_annum": "Thu nhập hằng năm", 
    "loan_amount": "Số tiền vay", "loan_term": "Thời hạn vay", 
    "cibil_score": "Điểm tín dụng (CIBIL)", "residential_assets_value": "Giá trị tài sản nhà ở", 
    "commercial_assets_value": "Giá trị tài sản kinh doanh", 
    "luxury_assets_value": "Giá trị tài sản cao cấp", "bank_asset_value": "Tài sản tại ngân hàng"
}


def predict_labels(X_input_raw, W, c, rf_model_used, scaler_used):
    """Sử dụng mô hình hybrid để dự đoán cho điểm dữ liệu mới."""

    X_scaled_new = scaler_used.transform(X_input_raw) 
    
    P = Hybrid_Predict_Proba(X_scaled_new, rf_model_used, W, c)
    return Final_Predict(P), P

st.set_page_config(layout="wide", page_title="Dashboard Tín Dụng", page_icon="🏦")


with st.sidebar:
    st.title("🏦 Hệ Thống Duyệt Hồ Sơ Vay")
    st.markdown("Nhóm 2 - Giải tích 1")

    st.header("Tham số mô hình")
    with st.expander("Xem trọng số và bias"):
        st.metric(label="Bias (c)", value=f"{b:.4f}")
        for feature, weight in zip(FEATURE_NAMES, w):
            st.markdown(f"**{FEATURE_VIETMAP[feature]}**: {weight:.4f}")

    st.header("Nhập hồ sơ cần kiểm tra")
    input_data_raw = {}
 
    X_data_unscaled = scaler.inverse_transform(x) 
    mean_x_unscaled = np.mean(X_data_unscaled, axis=0)

    for i, feature in enumerate(FEATURE_NAMES):
        label_vi = FEATURE_VIETMAP[feature]
        default_value_unscaled = mean_x_unscaled[i]

        if feature == "no_of_dependents":
            input_data_raw[feature] = st.number_input(label=label_vi, min_value=0, step=1, value=int(round(default_value_unscaled)))
        elif feature == "education":
            current_choice = "Tốt nghiệp" if default_value_unscaled > 0.5 else "Chưa tốt nghiệp"
            input_data_raw[feature] = st.selectbox(label=label_vi, options=["Tốt nghiệp", "Chưa tốt nghiệp"], index=["Tốt nghiệp", "Chưa tốt nghiệp"].index(current_choice))
        elif feature == "self_employed":
            current_choice = "Có" if default_value_unscaled > 0.5 else "Không"
            input_data_raw[feature] = st.selectbox(label=label_vi, options=["Có", "Không"], index=["Có", "Không"].index(current_choice))
        else:
            value_to_display = int(default_value_unscaled) if default_value_unscaled > 10000 or feature in ['cibil_score', 'loan_term'] else float(default_value_unscaled)
            input_data_raw[feature] = st.number_input(
                label=label_vi,
                value=value_to_display,
                format="%d" if default_value_unscaled > 10000 or feature in ['cibil_score', 'loan_term'] else "%.2f"
            )

    if st.button("Dự đoán hồ sơ", use_container_width=True, type="primary"):

        final_input_values_raw = []
        for name in FEATURE_NAMES:
            if name == 'education':
                final_input_values_raw.append(1.0 if input_data_raw[name] == "Tốt nghiệp" else 0.0)
            elif name == 'self_employed':
                final_input_values_raw.append(1.0 if input_data_raw[name] == "Có" else 0.0)
            else:
                final_input_values_raw.append(float(input_data_raw[name]))
        
        new_point_raw = np.array(final_input_values_raw).reshape(1, -1)

        prediction, prob = predict_labels(new_point_raw, w, b, rf_model, scaler)

        if prediction[0] == 1:
            st.success(f"✅ Kết quả: **DUYỆT HỒ SƠ** — Xác suất: {prob[0]:.2%}")
            st.balloons()
        else:
            st.error(f"❌ Kết quả: **TỪ CHỐI** — Xác suất bị từ chối: {(1 - prob[0]):.2%}")


tab1, tab2, tab3 = st.tabs(["Quá trình huấn luyện", "Đánh giá mô hình", "Trực quan hóa"])

EP_TRAIN = 10000 
loss_divisor = 100 
loss_indices = np.arange(len(losses)) * (EP_TRAIN // loss_divisor) 


with tab1:
    st.header("Diễn biến Loss")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dữ liệu đầu vào mô hình")
        df_scaled = pd.DataFrame(x, columns=[FEATURE_VIETMAP[f] for f in FEATURE_NAMES])
        df_scaled['Trạng thái vay (y)'] = y
        st.dataframe(df_scaled.head())

    with col2:
        st.subheader("Loss giảm theo thời gian (Logistic Regression)")
        fig, ax = plt.subplots()
        ax.plot(loss_indices, losses, color='blue')
        ax.set_title("Loss giảm theo Epoch ")
        ax.set_xlabel("Epoch (x100 iterations)")
        ax.set_ylabel("Loss (Binary Cross-Entropy)")
        st.pyplot(fig)

with tab2:
    st.header("Hiệu năng mô hình")
    col1, col2 = st.columns([1, 1.4])
    
    if len(y_pred) > 0:
        with col1:
            acc = accuracy_score(y, y_pred)
            st.metric("Độ chính xác", f"{acc:.2%}")

            report = classification_report(
                y, y_pred,
                target_names=['Từ chối', 'Duyệt'],
                output_dict=True
            )
            st.subheader("Báo cáo phân loại")
            st.dataframe(pd.DataFrame(report).transpose())

        with col2:
            st.subheader("Ma trận Nhầm lẫn")
            cm = confusion_matrix(y, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Dự đoán: Từ chối', 'Dự đoán: Duyệt'],
                        yticklabels=['Thực tế: Từ chối', 'Thực tế: Duyệt'],
                        ax=ax_cm)
            ax_cm.set_title('Confusion Matrix')
            st.pyplot(fig_cm)
    else:
        st.warning("Không đủ dữ liệu để đánh giá hiệu năng.")


with tab3:
    st.header("Trực quan hóa 2 chiều")
    st.warning("Do dữ liệu có 11 chiều, chỉ có thể vẽ 'lát cắt' 2D bằng cách chọn 2 đặc trưng.", icon="⚠️")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Độ quan trọng của đặc trưng (LR Weights)")
        importance = pd.DataFrame({
            "Đặc trưng": [FEATURE_VIETMAP[f] for f in FEATURE_NAMES],
            "Độ quan trọng": np.abs(w)
        }).sort_values("Độ quan trọng", ascending=False)

        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Độ quan trọng", y="Đặc trưng", data=importance, palette="viridis", ax=ax_imp)
        ax_imp.set_title("Trọng số (W) tuyệt đối của Logistic Regression")
        st.pyplot(fig_imp)

    with col2:
        st.subheader("Chọn hai đặc trưng để vẽ ranh giới")
        

        format_func_vi = lambda f: FEATURE_VIETMAP[f]

        safe_index_cibil = FEATURE_NAMES.index('cibil_score') if 'cibil_score' in FEATURE_NAMES else 0
        safe_index_income = FEATURE_NAMES.index('income_annum') if 'income_annum' in FEATURE_NAMES else 1

        feat_x = st.selectbox("Trục X ", FEATURE_NAMES, index=safe_index_cibil, format_func=format_func_vi)
        feat_y = st.selectbox("Trục Y", FEATURE_NAMES, index=safe_index_income, format_func=format_func_vi)

        if feat_x == feat_y:
            st.error("Hai đặc trưng phải khác nhau.")
        else:
            ix = FEATURE_NAMES.index(feat_x)
            iy = FEATURE_NAMES.index(feat_y)

            mean_values_scaled = np.mean(x, axis=0)
            grid = np.ones((100*100, len(FEATURE_NAMES))) * mean_values_scaled

            xr = np.linspace(x[:, ix].min(), x[:, ix].max(), 100)
            yr = np.linspace(x[:, iy].min(), x[:, iy].max(), 100)
            xx, yy = np.meshgrid(xr, yr)

            grid[:, ix] = xx.ravel()
            grid[:, iy] = yy.ravel()

            y_grid_prob = Hybrid_Predict_Proba(grid, rf_model, w, b)
            Z = Final_Predict(y_grid_prob)
            Z = Z.reshape(xx.shape)

            fig2, ax2 = plt.subplots(figsize=(10, 8))
            ax2.contourf(xx, yy, Z, alpha=0.25, cmap=plt.cm.coolwarm)

            sns.scatterplot(
                x=x[:, ix], y=x[:, iy], hue=y,
                palette=['#FF5733', '#1F77FF'], 
                s=110, ax=ax2, style=y,
                legend='full'
            )
            ax2.set_xlabel(f"{FEATURE_VIETMAP[feat_x]} (Chuẩn hóa)")
            ax2.set_ylabel(f"{FEATURE_VIETMAP[feat_y]} (Chuẩn hóa)")

            st.pyplot(fig2)