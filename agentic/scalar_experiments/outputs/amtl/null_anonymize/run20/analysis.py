import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix

# Load data
_df = pd.read_csv("amtl.csv")

# Rename for readability
cols = {
    "feature1": "tooth_class",
    "feature2": "specimen_id",
    "feature3": "missing_teeth",
    "feature4": "observable_sockets",
    "feature5": "age",
    "feature6": "age_uncertainty",
    "feature7": "sex",
    "feature8": "genus",
    "feature9": "region",
}

df = _df.rename(columns=cols).copy()

# Basic cleaning
for col in ["missing_teeth", "observable_sockets", "age", "sex"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Keep rows with valid counts
mask = (
    df["observable_sockets"].notna()
    & df["missing_teeth"].notna()
    & (df["observable_sockets"] > 0)
    & (df["missing_teeth"] >= 0)
    & (df["missing_teeth"] <= df["observable_sockets"])
    & df["age"].notna()
    & df["sex"].notna()
    & df["tooth_class"].notna()
    & df["genus"].notna()
)

df = df.loc[mask].copy()

# Human indicator
human_label = "Homo sapiens"
df["is_human"] = (df["genus"] == human_label).astype(int)

# Response as proportion
# statsmodels GLM can take proportion with var_weights as n

# Model with human indicator using successes/failures
X = dmatrix(
    "is_human + age + sex + C(tooth_class)",
    data=df,
    return_type="dataframe",
)

endog = np.column_stack(
    [df["missing_teeth"].to_numpy(), (df["observable_sockets"] - df["missing_teeth"]).to_numpy()]
)

model = sm.GLM(endog, X, family=sm.families.Binomial()).fit()

# Predicted marginal mean for human vs non-human
# Keep covariates as observed; toggle is_human
base = df[["is_human", "age", "sex", "tooth_class", "observable_sockets"]].copy()

preds = {}
for val, label in [(0, "nonhuman"), (1, "human")]:
    tmp = base.copy()
    tmp["is_human"] = val
    X_tmp = dmatrix(
        "is_human + age + sex + C(tooth_class)",
        data=tmp,
        return_type="dataframe",
    )
    p = model.predict(X_tmp)
    # Weighted average by observable sockets
    w = tmp["observable_sockets"].to_numpy()
    preds[label] = np.average(p, weights=w)

# Effect size: difference in predicted probability
pred_diff = preds["human"] - preds["nonhuman"]

# Wald test for is_human coefficient
coef = model.params.get("is_human", np.nan)
se = model.bse.get("is_human", np.nan)
z = coef / se if se and not np.isnan(se) else np.nan
p_value = model.pvalues.get("is_human", np.nan)

# Map to Likert scale
# Direction by pred_diff
# Strength based on p_value and magnitude
abs_diff = abs(pred_diff)
score = 0

if pred_diff > 0:
    if p_value < 0.001 and abs_diff >= 0.10:
        score = 90
    elif p_value < 0.01 and abs_diff >= 0.05:
        score = 75
    elif p_value < 0.05 and abs_diff >= 0.02:
        score = 55
    elif p_value < 0.1 and abs_diff >= 0.01:
        score = 35
    elif abs_diff >= 0.01:
        score = 20
    else:
        score = 5
elif pred_diff < 0:
    if p_value < 0.001 and abs_diff >= 0.10:
        score = -90
    elif p_value < 0.01 and abs_diff >= 0.05:
        score = -75
    elif p_value < 0.05 and abs_diff >= 0.02:
        score = -55
    elif p_value < 0.1 and abs_diff >= 0.01:
        score = -35
    elif abs_diff >= 0.01:
        score = -20
    else:
        score = -5
else:
    score = 0

score = int(np.clip(round(score), -100, 100))

# Save scalar conclusion
with open("conclusion.txt", "w", encoding="utf-8") as f:
    f.write(str(score))

# Also save a brief JSON report for traceability (not requested, but useful for debugging)
report = {
    "n_rows": int(df.shape[0]),
    "predicted_human": float(preds["human"]),
    "predicted_nonhuman": float(preds["nonhuman"]),
    "pred_diff": float(pred_diff),
    "coef_is_human": float(coef),
    "se_is_human": float(se),
    "p_value": float(p_value),
    "score": score,
}
with open("analysis_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
