import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy

# Load data
_df = pd.read_csv("amtl.csv")

# Basic checks and derived fields
_df = _df.copy()
_df["is_human"] = (_df["genus"] == "Homo sapiens").astype(int)
_df["failures"] = _df["sockets"] - _df["num_amtl"]

# Remove any rows with nonpositive socket counts or negative failures
_df = _df[(
    _df["sockets"].notna()
    & _df["num_amtl"].notna()
    & _df["age"].notna()
    & _df["prob_male"].notna()
    & _df["tooth_class"].notna()
)]
_df = _df[_df["sockets"] > 0]
_df = _df[_df["failures"] >= 0]

# Build design matrices
formula = "is_human + age + prob_male + C(tooth_class)"
X = patsy.dmatrix(formula, _df, return_type="dataframe")
design_info = X.design_info
Y = _df[["num_amtl", "failures"]]

model = sm.GLM(Y, X, family=sm.families.Binomial())
result = model.fit()

coef = result.params["is_human"]
se = result.bse["is_human"]
z = coef / se
p = result.pvalues["is_human"]
odds_ratio = float(np.exp(coef))

# Marginal predicted probability difference at mean covariates
mean_age = float(_df["age"].mean())
mean_male = float(_df["prob_male"].mean())

# Use tooth_class distribution to average predictions
class_probs = _df["tooth_class"].value_counts(normalize=True)

preds = {}
for human in [0, 1]:
    pr_list = []
    for cls, w in class_probs.items():
        row = {
            "is_human": human,
            "age": mean_age,
            "prob_male": mean_male,
            "tooth_class": cls,
        }
        X_row = patsy.build_design_matrices([design_info], pd.DataFrame([row]))[0]
        pr = float(result.predict(X_row)[0])
        pr_list.append(pr * w)
    preds[human] = float(np.sum(pr_list))

delta = preds[1] - preds[0]

# Heuristic mapping to Likert scale
# Use z-stat strength and odds ratio to set magnitude
abs_z = abs(float(z))
base = min(100, int(round((abs_z / 4.0) * 100)))  # z=4 -> 100

# Adjust based on effect size in probability space
if abs(delta) >= 0.10:
    base = min(100, base + 10)
elif abs(delta) >= 0.05:
    base = min(100, base + 5)

score = base if coef > 0 else -base
score = int(max(-100, min(100, score)))

summary = {
    "n_rows_used": int(_df.shape[0]),
    "coef_is_human": float(coef),
    "se_is_human": float(se),
    "z_is_human": float(z),
    "p_is_human": float(p),
    "odds_ratio_is_human": float(odds_ratio),
    "pred_prob_nonhuman": float(preds[0]),
    "pred_prob_human": float(preds[1]),
    "pred_prob_diff": float(delta),
    "likert_score": int(score),
}

print(summary)

# Write conclusion
with open("conclusion.txt", "w", encoding="utf-8") as f:
    f.write(str(score))
