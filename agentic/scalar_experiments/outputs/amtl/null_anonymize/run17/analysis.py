import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy

# Load data
_df = pd.read_csv("amtl.csv")

# Basic cleaning and feature setup
_df = _df.copy()
_df["missing"] = _df["feature3"].astype(float)
_df["total"] = _df["feature4"].astype(float)
_df = _df[(_df["total"] > 0) & (_df["missing"] >= 0) & (_df["missing"] <= _df["total"])].copy()

_df["prop"] = _df["missing"] / _df["total"]
_df["is_human"] = (_df["feature8"] == "Homo sapiens").astype(int)

# Build design matrices
formula = "prop ~ is_human + feature5 + feature7 + C(feature1)"

y, X = patsy.dmatrices(formula, data=_df, return_type="dataframe")

# Binomial GLM with weights equal to number of trials
model = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=_df["total"])
res = model.fit()

coef = float(res.params.get("is_human", np.nan))
p_value = float(res.pvalues.get("is_human", np.nan))

# Marginal average predicted probability difference
X_human = X.copy()
X_human["is_human"] = 1
X_non = X.copy()
X_non["is_human"] = 0
pred_h = res.predict(X_human)
pred_n = res.predict(X_non)
avg_diff = float(np.mean(pred_h - pred_n))

# Map to Likert scale [-100, 100]
if np.isnan(coef) or np.isnan(p_value) or np.isnan(avg_diff):
    score = 0
else:
    sign = 1 if coef > 0 else -1 if coef < 0 else 0
    # magnitude from probability difference (cap at 0.2)
    mag = min(abs(avg_diff) / 0.2, 1.0)
    # confidence from p-value
    if p_value < 0.001:
        conf = 1.0
    elif p_value < 0.01:
        conf = 0.85
    elif p_value < 0.05:
        conf = 0.7
    elif p_value < 0.1:
        conf = 0.5
    else:
        conf = 0.3
    score = int(round(sign * (100.0 * mag * conf)))
    if score == 0 and sign != 0:
        # ensure a minimal directional signal when effect exists
        score = sign * 1

# Write conclusion
with open("conclusion.txt", "w", encoding="utf-8") as f:
    f.write(str(int(score)))

# Print key stats for traceability
print({
    "coef_is_human": coef,
    "p_value": p_value,
    "avg_prob_diff": avg_diff,
    "score": score,
})
