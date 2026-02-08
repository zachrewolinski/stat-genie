import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

_df = pd.read_csv("amtl.csv")

_df["is_human"] = (_df["genus"] == "Homo sapiens").astype(int)
_df["tooth_class"] = _df["tooth_class"].astype("category")

_df = _df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "is_human"])
_df = _df[(_df["sockets"] > 0) & (_df["num_amtl"] >= 0) & (_df["num_amtl"] <= _df["sockets"])].copy()

X = _df[["is_human", "age", "prob_male", "tooth_class"]].copy()
X = pd.get_dummies(X, columns=["tooth_class"], drop_first=True)
X = sm.add_constant(X, has_constant="add")

successes = _df["num_amtl"].astype(float).values
failures = (_df["sockets"] - _df["num_amtl"]).astype(float).values
endog = np.column_stack([successes, failures])

model = sm.GLM(endog, X, family=sm.families.Binomial())
result = model.fit()

coef = result.params.get("is_human", np.nan)
se = result.bse.get("is_human", np.nan)
z = coef / se if np.isfinite(coef) and np.isfinite(se) and se != 0 else np.nan
p_value = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
odds_ratio = float(np.exp(coef)) if np.isfinite(coef) else np.nan

median_age = float(_df["age"].median())
median_prob_male = float(_df["prob_male"].median())
mode_tooth = _df["tooth_class"].mode().iloc[0]

base = {
    "const": 1.0,
    "is_human": 0,
    "age": median_age,
    "prob_male": median_prob_male,
}

for col in X.columns:
    if col.startswith("tooth_class_"):
        base[col] = 1.0 if col == f"tooth_class_{mode_tooth}" else 0.0

row_nonhuman = base.copy()
row_human = base.copy()
row_human["is_human"] = 1

pred_df = pd.DataFrame([row_nonhuman, row_human])[X.columns]

pred_probs = result.predict(pred_df)

pred_diff = float(pred_probs.iloc[1] - pred_probs.iloc[0])

summary = {
    "n_rows": int(_df.shape[0]),
    "coef_is_human": float(coef),
    "se_is_human": float(se),
    "z_is_human": float(z),
    "p_is_human": float(p_value),
    "odds_ratio_is_human": float(odds_ratio),
    "pred_prob_nonhuman": float(pred_probs.iloc[0]),
    "pred_prob_human": float(pred_probs.iloc[1]),
    "pred_prob_diff": float(pred_diff),
    "mode_tooth_class": str(mode_tooth),
    "median_age": float(median_age),
    "median_prob_male": float(median_prob_male),
}

print(summary)
