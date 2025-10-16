# file: esup_auto_hide.py
import streamlit as st
import pandas as pd
import itertools

st.set_page_config(page_title="Auto Hide Frequent Itemsets", layout="wide")

# ==========================
# Chuẩn hoá dữ liệu
# ==========================
def prepare_df(df):
    possible_id_cols = ['TID', 'Id', 'id', 'tid', 'transaction']
    for c in possible_id_cols:
        if c in df.columns:
            df = df.drop(columns=[c])
    return df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

# ==========================
# eSup của 1 transaction
# ==========================
def esup_transaction_safe(row, itemset):
    prob = 1.0
    for item in itemset:
        try:
            prob *= float(row.get(item, 0.0))
        except Exception:
            prob *= 0.0
    return prob

# ==========================
# Tính eSup toàn bộ (1,2,3)
# ==========================
def compute_global_esup(df):
    items = df.columns.tolist()
    summary = []
    N = len(df)
    if N == 0:
        return pd.DataFrame(columns=["Itemset", "eSup"])
    for k in (1, 2, 3):
        for subset in itertools.combinations(items, k):
            total = 0.0
            for _, row in df.iterrows():
                total += esup_transaction_safe(row, subset)
            summary.append({"Itemset": tuple(subset), "eSup": total / N})
    return pd.DataFrame(summary).sort_values(by="eSup", ascending=False).reset_index(drop=True)

# ==========================
# Aggregate
# ==========================
def aggregate(df, frequent_sets, min_sup=0.5):
    if df.empty:
        return df.copy()
    sanitized = df.copy().reset_index(drop=True)

    def compute_esup(data, itemset):
        total = 0.0
        for _, row in data.iterrows():
            prob = 1.0
            for i in itemset:
                prob *= float(row.get(i, 0))
            total += prob
        return total / len(data) if len(data) > 0 else 0.0

    popular_items = set()
    for s in frequent_sets:
        if isinstance(s, str):
            try: s = eval(s)
            except: s = (s,)
        for i in s: popular_items.add(i)

    while True:
        current_esup = {x: compute_esup(sanitized, (x,)) for x in popular_items}
        if all(v < min_sup for v in current_esup.values()):
            break

        scores = []
        for idx, row in sanitized.iterrows():
            count_popular = sum(1 for x in popular_items if row.get(x, 0) > 0)
            if count_popular == 0: continue
            total_influence = sum(row.get(x, 0) for x in popular_items)
            scores.append((idx, count_popular, total_influence))

        if not scores: break
        scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        to_remove = scores[0][0]
        sanitized = sanitized.drop(index=to_remove).reset_index(drop=True)

    return sanitized

# ==========================
# Disaggregate
# ==========================
def disaggregate(df, frequent_sets):
    sanitized = df.copy()
    for idx, row in sanitized.iterrows():
        f_values = {}
        for subset in frequent_sets:
            if all(row.get(i, 0) > 0 for i in subset):
                for x in subset:
                    a_kx = row[x]; pr_x_in_k = row[x]; b_kx = 1.0
                    for y in subset:
                        if y != x: b_kx *= row[y]
                    f_kx = b_kx / (a_kx * pr_x_in_k) if a_kx * pr_x_in_k != 0 else 0.0
                    f_values[x] = max(f_values.get(x, 0.0), f_kx)
        if f_values:
            max_item = max(f_values, key=f_values.get)
            sanitized.at[idx, max_item] = 0.0
    return sanitized

# ==========================
# Hybrid 50–50 (rút gọn log)
# ==========================
def hybrid(df, frequent_sets, min_sup=0.5):
    if df.empty:
        return df.copy()
    sanitized = df.copy().reset_index(drop=True)

    popular_items = set()
    for s in frequent_sets:
        if isinstance(s, str):
            try: s = eval(s)
            except: s = (s,)
        for i in s: popular_items.add(i)

    def compute_esup(data, item):
        total = 0.0
        for _, row in data.iterrows():
            total += float(row.get(item, 0))
        return total / len(data) if len(data) > 0 else 0.0

    def transaction_influence(row):
        return sum(float(row.get(i, 0)) for i in popular_items)

    while True:
        esup_values = {i: compute_esup(sanitized, i) for i in popular_items}
        if all(v < min_sup for v in esup_values.values()):
            break
        N = len(sanitized)
        if N == 0: break

        half = max(1, N // 2)
        sanitized["influence"] = sanitized.apply(transaction_influence, axis=1)
        sanitized = sanitized.sort_values(by="influence", ascending=False).reset_index(drop=True)
        sanitized = sanitized.drop(index=list(range(min(half, len(sanitized))))).reset_index(drop=True)

        if len(sanitized) == 0: break

        num_to_hide = max(1, len(sanitized) // 2)
        sanitized["influence"] = sanitized.apply(transaction_influence, axis=1)
        top_rows = sanitized.sort_values(by="influence", ascending=False).head(num_to_hide)
        for idx in top_rows.index:
            row = sanitized.loc[idx]
            item_values = {i: row.get(i, 0) for i in popular_items}
            target_item = max(item_values, key=item_values.get)
            if row.get(target_item, 0) > 0:
                sanitized.at[idx, target_item] = 0.0
        sanitized = sanitized.drop(columns=["influence"])
        if all(v < min_sup for v in {i: compute_esup(sanitized, i) for i in popular_items}.values()):
            break

    return sanitized.reset_index(drop=True)

# ==========================
# Giao diện Streamlit
# ==========================
st.title("🔒 Ẩn Tập Mục Phổ Biến Tự Động (Heuristic)")

uploaded_file = st.file_uploader("📂 Tải file dữ liệu CSV/TXT", type=["csv", "txt"])

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception:
        df_raw = pd.read_csv(uploaded_file, engine="python")

    st.subheader("📋 Dữ liệu gốc")
    st.dataframe(df_raw)

    df = prepare_df(df_raw)
    global_esup = compute_global_esup(df)

    st.subheader("📈 Expected Support (eSup)")
    st.dataframe(global_esup)

    min_esup = st.slider("Chọn ngưỡng min_eSup", 0.0, 1.0, 0.5, 0.05)
    frequent_sets = [tuple(x) for x in global_esup[global_esup["eSup"] >= min_esup]["Itemset"].tolist()]
    st.subheader("🔥 Tập mục phổ biến được phát hiện")
    st.write(frequent_sets if frequent_sets else "Không có tập nào vượt ngưỡng.")

    method = st.selectbox("🧩 Thuật toán ẩn:", ["Aggregate", "Disaggregate", "Hybrid"])

    if frequent_sets:
        if method == "Aggregate":
            sanitized = aggregate(df, frequent_sets)
        elif method == "Disaggregate":
            sanitized = disaggregate(df, frequent_sets)
        else:
            sanitized = hybrid(df, frequent_sets, min_sup=min_esup)

        st.subheader("📉 Dữ liệu sau khi ẩn")
        st.dataframe(sanitized)
        st.success("✅ Hoàn tất quá trình ẩn dữ liệu.")
    else:
        st.info("Không có tập phổ biến nào vượt ngưỡng để ẩn.")
else:
    st.info("📥 Hãy tải file dữ liệu để bắt đầu.")
