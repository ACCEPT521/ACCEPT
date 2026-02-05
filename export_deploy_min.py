# export_deploy_min.py
# Export only what Streamlit needs: (Pipeline model) + (Youden threshold) + (SHAP background) + (Top9 list)

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import roc_curve

random_seed = 42
np.random.seed(random_seed)

train_path = r"D:\4.毕业论文相关\数据重整-12-31\TRAIN\一些前期结果\train_data.xlsx"
out_dir    = r"D:\4.毕业论文相关\数据重整-12-31\Python"
deploy_dir = os.path.join(out_dir, "deploy_resources")
os.makedirs(deploy_dir, exist_ok=True)

TARGET_COL = "EPDSLL"
ID_COL     = "id_仅标识"

TOP9_VARS = ["EPDSA", "Insomnia", "Anxiety", "GA", "reactions", "Educational", "Capital", "PG", "HMI"]

# ✅ 与你原始训练脚本一致：PG、reactions 为 cat；Educational、HMI 为 ord
sel_num = ["EPDSA", "Insomnia", "Anxiety", "GA", "Capital"]
sel_cat = ["PG", "reactions"]
sel_ord = ["Educational", "HMI"]

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    for c in (sel_num + sel_ord):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in sel_cat:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df

def build_preprocess():
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    ord_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, sel_num),
            ("cat", cat_pipe, sel_cat),
            ("ord", ord_pipe, sel_ord),
        ],
        remainder="drop"
    )

def youden_threshold(y_true, y_score) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    youden = tpr - fpr
    best_idx = int(np.nanargmax(youden))
    return float(thr[best_idx])

train_df = basic_clean(pd.read_excel(train_path))
X_train = train_df[TOP9_VARS].copy()
y_train = train_df[TARGET_COL].astype(int)

preprocess = build_preprocess()

pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", SVC(probability=True, random_state=random_seed))
])

param_grid = {
    "clf__C": [0.1, 1, 10],
    "clf__kernel": ["linear", "rbf"]
}

gs = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=1)
gs.fit(X_train, y_train)

best_model = gs.best_estimator_

p_tr = best_model.predict_proba(X_train)[:, 1]
thr_star = youden_threshold(y_train, p_tr)

prep = best_model.named_steps["prep"]
X_tr_trans = prep.transform(X_train)
X_tr_dense = X_tr_trans.toarray() if sp.issparse(X_tr_trans) else np.asarray(X_tr_trans)

bg_size = min(100, X_tr_dense.shape[0])
rng = np.random.RandomState(random_seed)
bg_idx = rng.choice(X_tr_dense.shape[0], size=bg_size, replace=False)
background = X_tr_dense[bg_idx]

deploy_resources = {
    "best_model": best_model,
    "youden_threshold": thr_star,
    "shap_background": background,
    "final_top9_vars": TOP9_VARS
}

out_path = os.path.join(deploy_dir, "svm_topk9_deploy_res.joblib")
joblib.dump(deploy_resources, out_path, compress=3)

print("\n✅ Export done!")
print("Saved:", out_path)
print("Youden threshold:", thr_star)
print("Best params:", gs.best_params_)
