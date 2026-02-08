import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv("amtl.csv")

# Basic cleaning
# Ensure expected columns
required_cols = [
    "num_amtl",
    "sockets",
    "age",
    "prob_male",
    "tooth_class",
    "genus",
]
missing = [c for c in required_cols if c not in _df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Drop rows with missing key values or zero sockets
_df = _df.dropna(subset=required_cols)
_df = _df[_df["sockets"] > 0].copy()

# Create outcome proportion
_df["amtl_rate"] = _df["num_amtl"] / _df["sockets"]

# Create Homo sapiens indicator
_df["is_human"] = (_df["genus"] == "Homo sapiens").astype(int)

# Encode tooth_class as categorical with stable baseline
_df["tooth_class"] = _df["tooth_class"].astype("category")
# Set explicit order to keep baseline consistent
classes = ["Anterior", "Premolar", "Posterior"]
existing = [c for c in classes if c in _df["tooth_class"].cat.categories]
_df["tooth_class"] = _df["tooth_class"].cat.set_categories(existing)

# Design matrix
X = pd.get_dummies(
    _df[["is_human", "age", "prob_male", "tooth_class"]],
    columns=["tooth_class"],
    drop_first=True,
)
X = sm.add_constant(X, has_constant="add")

# Binomial GLM with logit link and socket counts as variance weights
model = sm.GLM(
    _df["amtl_rate"],
    X,
    family=sm.families.Binomial(),
    var_weights=_df["sockets"],
)
result = model.fit()

# Extract effect for Homo sapiens
coef = result.params.get("is_human", np.nan)
se = result.bse.get("is_human", np.nan)

# Odds ratio and 95% CI
or_est = np.exp(coef) if np.isfinite(coef) else np.nan
ci_low = np.exp(coef - 1.96 * se) if np.isfinite(coef) and np.isfinite(se) else np.nan
ci_high = np.exp(coef + 1.96 * se) if np.isfinite(coef) and np.isfinite(se) else np.nan

# Compute average marginal effect at mean for is_human
# Predict probability for human vs non-human at mean covariates
means = X.mean()
means_h = means.copy()
means_h["is_human"] = 1.0
means_n = means.copy()
means_n["is_human"] = 0.0

pred_h = result.predict(means_h)
pred_n = result.predict(means_n)

# Summaries for decision
print("N rows:", len(_df))
print("Human rows:", int(_df["is_human"].sum()))
print("Non-human rows:", int((1 - _df["is_human"]).sum()))
print("Coef is_human:", float(coef))
print("SE is_human:", float(se))
print("OR is_human:", float(or_est))
print("OR 95% CI:", float(ci_low), float(ci_high))
print("Predicted AMTL rate (human):", float(pred_h))
print("Predicted AMTL rate (non-human):", float(pred_n))
print("Difference (human - non-human):", float(pred_h - pred_n))

# Save a small JSON-like summary for reference
summary = {
    "n": int(len(_df)),
    "n_human": int(_df["is_human"].sum()),
    "n_nonhuman": int((1 - _df["is_human"]).sum()),
    "coef_is_human": float(coef),
    "se_is_human": float(se),
    "or_is_human": float(or_est),
    "or_ci_low": float(ci_low),
    "or_ci_high": float(ci_high),
    "pred_human": float(pred_h),
    "pred_nonhuman": float(pred_n),
    "pred_diff": float(pred_h - pred_n),
}

with open("analysis_summary.json", "w") as f:
    f.write(str(summary))
