import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="EPDSLL 预测模型(SVM TopK=9)", layout="centered")

# ✅ MODEL_PATH 永远相对 app.py 所在目录
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "deploy_resources" / "svm_topk9_deploy_res.joblib"

@st.cache_resource
def load_deploy_resources(path: Path):
    res = joblib.load(path)
    required = ["best_model", "youden_threshold", "final_top9_vars"]
    missing = [k for k in required if k not in res]
    if missing:
        raise ValueError(f"Deploy resource missing key(s): {missing}")
    return res

# -----------------------------
# Label mappings (dropdowns)
# -----------------------------
EDU_MAP = {1: "高中/中专及以下", 2: "大专", 3: "本科", 4: "硕士及以上"}
PG_MAP = {0: "计划内", 1: "计划外"}
REACTIONS_MAP = {
    1: "无反应",
    2: "正常妊娠反应（恶心呕吐）",
    3: "不良妊娠反应（感冒、出血、严重恶心呕吐就医）",
}
HMI_MAP = {1: "10000以下", 2: "10001-20000", 3: "20000以上"}

# -----------------------------
# UI
# -----------------------------
st.title("孕晚期抑郁症状预测模型 (SVM)")
st.write("填写下方信息，点击 **Predict** 输出预测概率。")

# 🔧 调试信息
with st.expander("🔧 部署调试信息"):
    st.write("APP_DIR:", str(APP_DIR))
    st.write("MODEL_PATH:", str(MODEL_PATH))
    st.write("模型文件是否存在:", MODEL_PATH.exists())
    st.write("当前工作目录 CWD:", os.getcwd())

# 模型文件不存在 → 直接终止
if not MODEL_PATH.exists():
    st.error(f"找不到模型文件：{MODEL_PATH}")
    st.stop()

# 加载模型
try:
    res = load_deploy_resources(MODEL_PATH)
except Exception as e:
    st.error(f"模型文件加载失败：{e}")
    st.stop()

best_model = res["best_model"]
thr_star   = float(res["youden_threshold"])
TOP9_VARS  = res["final_top9_vars"]

with st.expander("模型信息（部署）"):
    st.write("模型文件：", str(MODEL_PATH))
    st.write("TopK=9 特征顺序：", TOP9_VARS)
    st.write(f"训练集 Youden 阈值：{thr_star:.6f}")

# -----------------------------
# Input
# -----------------------------
st.subheader("输入变量")
col1, col2 = st.columns(2)

with col1:
    EPDSA = st.number_input("孕早期 EPDS 分数 (EPDSA)", value=10.0, step=1.0)
    Insomnia = st.number_input("睡眠情况分数 (Insomnia)", value=1.0, step=1.0)
    Anxiety = st.number_input("妊娠焦虑分数 (Anxiety)", value=1.0, step=1.0)
    GA = st.number_input("孕周 (GA)", value=38.0, step=0.1, format="%.1f")
    Capital = st.number_input("社会资本分数 (Capital)", value=1.0, step=1.0)

with col2:
    edu_code = st.selectbox(
        "学历 (Educational)",
        options=list(EDU_MAP.keys()),
        format_func=lambda x: f"{x} - {EDU_MAP[x]}",
        index=1,
    )
    pg_code = st.selectbox(
        "怀孕计划 (PG)",
        options=list(PG_MAP.keys()),
        format_func=lambda x: f"{x} - {PG_MAP[x]}",
        index=0,
    )
    react_code = st.selectbox(
        "孕期反应 (reactions)",
        options=list(REACTIONS_MAP.keys()),
        format_func=lambda x: f"{x} - {REACTIONS_MAP[x]}",
        index=0,
    )
    hmi_code = st.selectbox(
        "家庭月总收入 (HMI)",
        options=list(HMI_MAP.keys()),
        format_func=lambda x: f"{x} - {HMI_MAP[x]}",
        index=1,
    )

# -----------------------------
# DataFrame (顺序严格一致)
# -----------------------------
x = pd.DataFrame(
    [{
        "EPDSA": EPDSA,
        "Insomnia": Insomnia,
        "Anxiety": Anxiety,
        "GA": GA,
        "reactions": react_code,
        "Educational": edu_code,
        "Capital": Capital,
        "PG": pg_code,
        "HMI": hmi_code,
    }],
    columns=TOP9_VARS,
)

# 与训练保持一致：PG / reactions → str
x["PG"] = x["PG"].astype(int).astype(str)
x["reactions"] = x["reactions"].astype(int).astype(str)

st.divider()

# -----------------------------
# Predict
# -----------------------------
predict_btn = st.button("Predict", type="primary")

if predict_btn:
    try:
        proba = float(best_model.predict_proba(x)[0, 1])
    except Exception as e:
        st.error(f"预测失败：{e}")
        st.stop()

    st.subheader(f"预测孕晚期抑郁阳性 (EPDS>9) 概率：{proba*100:.2f}%")

    if proba >= thr_star:
        st.error(f"高风险：概率 ≥ Youden 阈值 ({thr_star:.6f})")
    else:
        st.success(f"低风险：概率 < Youden 阈值 ({thr_star:.6f})")

    with st.expander("查看传入模型的编码值"):
        st.dataframe(x)

st.caption("运行方式：pip install -r requirements.txt  |  streamlit run app.py")
