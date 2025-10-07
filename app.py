import streamlit as st
import pandas as pd
import numpy as np
import io

# ==========================
# HÀM TÍNH EXPECTED SUPPORT
# ==========================
def expected_support(df, itemset):
    """Tính ESup của một tập mục trong CSDL không chắc chắn."""
    esup = 0
    for _, row in df.iterrows():
        prob = np.prod([row[item] for item in itemset])
        esup += prob
    return esup

# ==========================
# 3 GIẢI THUẬT HEURISTIC
# ==========================
def u_aggregate(df, itemset, minsup):
    df_mod = df.copy()
    while expected_support(df_mod, itemset) >= minsup and len(df_mod) > 0:
        df_mod["contrib"] = df_mod.apply(lambda r: np.prod([r[i] for i in itemset]), axis=1)
        max_idx = df_mod["contrib"].idxmax()
        df_mod = df_mod.drop(index=max_idx)
    return df_mod.drop(columns="contrib", errors='ignore')

def u_disaggregate(df, itemset, minsup, remove_all=False):
    """
    U-Disaggregate: Xóa item nhạy cảm trong các giao dịch có đóng góp lớn nhất
    để làm giảm Expected Support của tập mục nhạy cảm.
    
    - remove_all=False: chỉ xóa item đầu tiên trong itemset
    - remove_all=True: xóa toàn bộ item trong itemset
    """
    df_mod = df.copy()

    while expected_support(df_mod, itemset) >= minsup:
        # Tính độ đóng góp của từng giao dịch
        contrib = df_mod.apply(lambda r: np.prod([r[i] for i in itemset]), axis=1)
        max_idx = contrib.idxmax()  # chọn dòng có contrib cao nhất

        # Xóa item nhạy cảm trong giao dịch đó
        if remove_all:
            for i in itemset:
                df_mod.loc[max_idx, i] = 0
        else:
            df_mod.loc[max_idx, itemset[0]] = 0  # chỉ xóa item đầu tiên

    return df_mod


def u_hybrid(df, itemset, minsup):
    df_mod = u_disaggregate(df, itemset, minsup, reduce_factor=0.7)
    df_mod = u_aggregate(df_mod, itemset, minsup)
    return df_mod

# ==========================
# DIFFERENTIAL PRIVACY (DP)
# ==========================
def add_dp_noise(esup, epsilon=1.0, delta_f=1.0):
    noise = np.random.laplace(0, delta_f/epsilon)
    return esup + noise

# ==========================
# GIAO DIỆN STREAMLIT
# ==========================
st.set_page_config(page_title="Heuristic + Differential Privacy", layout="wide")
st.title("🔒 Ẩn các tập mục nhạy cảm trong CSDL không chắc chắn")
st.markdown("### Gồm 2 giai đoạn: Heuristic Hiding → Differential Privacy")

# ==========================
# NHẬP DỮ LIỆU
# ==========================
st.sidebar.header("📥 Nhập dữ liệu")
uploaded_file = st.sidebar.file_uploader("Chọn file TXT hoặc CSV", type=["txt", "csv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            content = uploaded_file.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(content), delim_whitespace=True)
            if not all(isinstance(x, str) for x in df.columns):
                num_cols = df.shape[1]
                df.columns = [f"I{i+1}" for i in range(num_cols)]
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
        st.stop()
else:
    st.sidebar.info("Hoặc dùng dữ liệu ví dụ:")
    df = pd.DataFrame({
        'A': [0.9, 0.7, 0.2],
        'B': [0.6, 0.5, 0.9],
        'C': [0.1, 0.4, 0.3]
    })

st.write("### 🧮 Cơ sở dữ liệu ban đầu:")
st.dataframe(df)

# ==========================
# CHỌN THAM SỐ
# ==========================
items = list(df.columns)
itemset = st.multiselect("Chọn tập mục nhạy cảm:", items)
if len(itemset) == 0:
    st.warning("⚠️ Vui lòng chọn ít nhất một mục trong tập nhạy cảm.")
    st.stop()

minsup = st.slider("Ngưỡng hỗ trợ kỳ vọng (min_esup)", 0.1, float(len(df)), 1.0, 0.1)
epsilon = st.slider("Giá trị ε (Differential Privacy)", 0.1, 5.0, 1.0, 0.1)
algo = st.selectbox("Chọn giải thuật Heuristic:", ["U-Aggregate", "U-Disaggregate", "U-Hybrid"])

# ==========================
# GIAI ĐOẠN 1: HEURISTIC HIDING
# ==========================
st.markdown("---")
st.subheader("🧱 Giai đoạn 1 – Ẩn tập mục nhạy cảm bằng Heuristic")

if st.button("▶️ Thực thi Giai đoạn 1"):
    esup_before = expected_support(df, itemset)
    st.info(f"🔹 Expected Support ban đầu của tập {itemset}: **{esup_before:.4f}**")

    if esup_before < minsup:
        st.warning("⚠️ Tập mục này KHÔNG phải frequent (ESup < min_sup) → không cần ẩn.")
        st.stop()

    if algo == "U-Aggregate":
        df_hidden = u_aggregate(df, itemset, minsup)
        method = "U-Aggregate"
    elif algo == "U-Disaggregate":
        df_hidden = u_disaggregate(df, itemset, minsup)
        method = "U-Disaggregate"
    else:
        df_hidden = u_hybrid(df, itemset, minsup)
        method = "U-Hybrid"

    esup_after = expected_support(df_hidden, itemset)
    st.success(f"✅ Đã áp dụng {method}! ESup sau khi ẩn = {esup_after:.4f}")

    st.write("### ✅ CSDL sau khi ẩn:")
    st.dataframe(df_hidden)

    # Lưu kết quả giai đoạn 1 vào session_state
    st.session_state["df_hidden"] = df_hidden
    st.session_state["esup_after"] = esup_after

# ==========================
# GIAI ĐOẠN 2: DIFFERENTIAL PRIVACY
# ==========================
st.markdown("---")
st.subheader("🔐 Giai đoạn 2 – Thêm Differential Privacy (DP)")

if "df_hidden" in st.session_state:
    df_hidden = st.session_state["df_hidden"]
    esup_after = st.session_state["esup_after"]

    if st.button("🧮 Áp dụng Giai đoạn 2 (DP)"):
        esup_noisy = add_dp_noise(esup_after, epsilon)
        st.info(f"Expected Support gốc sau ẩn: **{esup_after:.4f}**")
        st.success(f"Expected Support sau khi thêm nhiễu (ε={epsilon}): **{esup_noisy:.4f}**")
        st.write("✅ Dữ liệu đã được bảo vệ thêm bằng Differential Privacy.")
else:
    st.warning("⚠️ Hãy chạy Giai đoạn 1 trước khi áp dụng Giai đoạn 2 (DP).")
