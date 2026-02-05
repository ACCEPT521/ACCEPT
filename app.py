import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
import streamlit as st
import shap
import streamlit.components.v1 as components

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="EPDSLL 预测模型(SVM TopK=9)", layout="centered")

# ✅ 关键：MODEL_PATH 永远相对 app.py 所在目录，不受当前终端目录影响
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "deploy_resources" / "svm_topk9_deploy_res.joblib"

@st.cache_resource
def load_deploy_resources(path: Path):
    res = joblib.load(path)
    required = ["best_model", "youden_threshold", "shap_background", "final_top9_vars"]
    missing = [k for k in required if k not in res]
    if missing:
        raise ValueError(f"Deploy resource missing key(s): {missing}")
    return res

def render_force_plot(exp: shap.Explanation):
    """最稳的 SHAP force plot: HTML 渲染"""
    force = shap.plots.force(exp, matplotlib=False)
    html = f"<head>{shap.getjs()}</head><body>{force.html()}</body>"
    components.html(html, height=260, scrolling=True)

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
st.title("孕晚期抑郁症状预测模型(Final SVM TopK=9)")
st.write("按下方输入信息，点击 **Predict** 输出 EPDSLL=1 的预测概率，并可选生成 SHAP 个体解释(force plot)。")

# 🔧 调试信息（部署时非常有用）
with st.expander("🔧 部署调试信息"):
    st.write("app.py 所在目录 APP_DIR:", str(APP_DIR))
    st.write("模型文件 MODEL_PATH:", str(MODEL_PATH))
    st.write("模型文件是否存在:", MODEL_PATH.exists())
    st.write("当前工作目录 CWD:", os.getcwd())

# ✅ 文件不存在时，直接给出明确错误，不继续运行
if not MODEL_PATH.exists():
    st.error(f"找不到模型文件：{MODEL_PATH}\n请确认 deploy_resources 文件夹与 app.py 同目录。")
    st.stop()

# ✅ 防崩：res 先定义，再加载
res = None
try:
    res = load_deploy_resources(MODEL_PATH)
except Exception as e:
    st.error(f"模型文件加载失败：{e}")
    st.stop()

best_model = res["best_model"]
thr_star   = float(res["youden_threshold"])
background = res["shap_background"]
TOP9_VARS  = res["final_top9_vars"]

with st.expander("模型信息（部署）"):
    st.write("模型文件：", str(MODEL_PATH))
    st.write("TopK=9 特征顺序：", TOP9_VARS)
    st.write(f"训练集 Youden 阈值：{thr_star:.6f}")

st.subheader("输入变量")
col1, col2 = st.columns(2)

with col1:
    EPDSA = st.number_input("孕早期 EPDS 分数(EPDSA)", value=10.0, step=1.0)
    Insomnia = st.number_input("睡眠情况分数(Insomnia)", value=1.0, step=1.0)
    Anxiety = st.number_input("妊娠焦虑分数(Anxiety)", value=1.0, step=1.0)
    GA = st.number_input("孕周(GA)", value=38.0, step=0.1, format="%.1f")
    Capital = st.number_input("社会资本分数(Capital)", value=1.0, step=1.0)

with col2:
    edu_code = st.selectbox("学历(Educational)", options=list(EDU_MAP.keys()),
                            format_func=lambda x: f"{x} - {EDU_MAP[x]}", index=1)
    pg_code = st.selectbox("怀孕计划(PG)", options=list(PG_MAP.keys()),
                           format_func=lambda x: f"{x} - {PG_MAP[x]}", index=0)
    react_code = st.selectbox("孕期反应(reactions)", options=list(REACTIONS_MAP.keys()),
                              format_func=lambda x: f"{x} - {REACTIONS_MAP[x]}", index=0)
    hmi_code = st.selectbox("家庭月总收入(HMI)", options=list(HMI_MAP.keys()),
                            format_func=lambda x: f"{x} - {HMI_MAP[x]}", index=1)

x = pd.DataFrame([{
    "EPDSA": EPDSA,
    "Insomnia": Insomnia,
    "Anxiety": Anxiety,
    "GA": GA,
    "reactions": react_code,
    "Educational": edu_code,
    "Capital": Capital,
    "PG": pg_code,
    "HMI": hmi_code,
}], columns=TOP9_VARS)

# 与训练一致：PG/reactions 用 str 进入 OneHot
x["PG"] = x["PG"].astype(int).astype(str)
x["reactions"] = x["reactions"].astype(int).astype(str)

st.divider()

colA, colB = st.columns([1, 1])
with colA:
    predict_btn = st.button("Predict", type="primary")
with colB:
    do_shap = st.checkbox("生成 SHAP 解释（可能较慢）", value=False)

if predict_btn:
    # ---- Predict ----
    try:
        proba = float(best_model.predict_proba(x)[0, 1])
    except Exception as e:
        st.error(f"预测失败：{e}")
        st.stop()

    st.subheader(f"预测 孕晚期抑郁阳性(EPDS>9分)概率：{proba*100:.2f}%")
    if proba >= thr_star:
        st.error(f"高风险：概率 ≥ 训练集 Youden 阈值（{thr_star:.6f})")
    else:
        st.success(f"低风险：概率 < 训练集 Youden 阈值（{thr_star:.6f})")

    with st.expander("查看传入模型的编码值（用于复现）"):
        st.dataframe(x)

    # ---- SHAP (optional) ----
    if do_shap:
        st.markdown("### SHAP 个体解释(force plot)")

        try:
            prep = best_model.named_steps["prep"]
            clf  = best_model.named_steps["clf"]

            x_trans = prep.transform(x)
            x_dense = x_trans.toarray() if sp.issparse(x_trans) else np.asarray(x_trans)

            f = lambda X: clf.predict_proba(X)[:, 1]

            @st.cache_resource
            def get_explainer(_background):
                _bg = np.asarray(_background)
                # 防止 background 太大导致云端卡死
                if _bg.ndim == 2 and _bg.shape[0] > 80:
                    _bg = _bg[:80, :]
                return shap.KernelExplainer(f, _bg)

            explainer = get_explainer(background)
            nsamples = 300
            shap_values = explainer.shap_values(x_dense, nsamples=nsamples)
            shap_pos = shap_values[0] if isinstance(shap_values, list) else shap_values

            ev = explainer.expected_value
            ev = ev[0] if isinstance(ev, (list, np.ndarray)) else ev

            exp = shap.Explanation(values=shap_pos[0], base_values=ev, data=x_dense[0])
            render_force_plot(exp)

        except Exception as e:
            st.warning(f"SHAP 解释生成失败（不影响预测结果）：{e}")

st.caption("运行：pip install -r requirements.txt  |  streamlit run app.py")
