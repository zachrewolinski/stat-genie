import json
import math
import pandas as pd
import statsmodels.api as sm

# Load data
path = "affairs.csv"
df = pd.read_csv(path)

# Basic cleaning
# Ensure expected columns
if "children" not in df.columns or "affairs" not in df.columns:
    raise ValueError("Required columns missing")

# Binary indicator for children
children_yes = df["children"].astype(str).str.lower().map({"yes": 1, "no": 0})
if children_yes.isna().any():
    raise ValueError("Unexpected values in children column")

df = df.copy()
df["children_yes"] = children_yes

df["affair_any"] = (df["affairs"] > 0).astype(int)

# Proportions
prop_yes = df.loc[df["children_yes"] == 1, "affair_any"].mean()
prop_no = df.loc[df["children_yes"] == 0, "affair_any"].mean()
prop_diff = prop_no - prop_yes  # positive => children reduces affairs

# Mean affairs
mean_yes = df.loc[df["children_yes"] == 1, "affairs"].mean()
mean_no = df.loc[df["children_yes"] == 0, "affairs"].mean()
mean_diff = mean_no - mean_yes

# Simple logistic regression with controls
# Use common covariates from Fair model if present
control_cols = [
    "gender",
    "age",
    "yearsmarried",
    "religiousness",
    "education",
    "occupation",
    "rating",
]

X = pd.DataFrame(index=df.index)
X["children_yes"] = df["children_yes"]

# Encode gender if present
if "gender" in df.columns:
    X["gender_male"] = (df["gender"].astype(str).str.lower() == "male").astype(int)

for col in control_cols:
    if col in df.columns and col not in X.columns:
        # Coerce to numeric when possible
        X[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing in X or y
model_df = pd.concat([df["affair_any"], X], axis=1).dropna()

y = model_df["affair_any"]
X_model = model_df.drop(columns=["affair_any"])
X_model = sm.add_constant(X_model, has_constant="add")

logit_result = None
try:
    logit_model = sm.Logit(y, X_model)
    logit_result = logit_model.fit(disp=False)
except Exception:
    logit_result = None

# Fallback: unadjusted logit
if logit_result is None:
    X_simple = sm.add_constant(df["children_yes"])
    logit_result = sm.Logit(df["affair_any"], X_simple).fit(disp=False)
    coef = logit_result.params["children_yes"]
    pval = logit_result.pvalues["children_yes"]
else:
    coef = logit_result.params.get("children_yes", float("nan"))
    pval = logit_result.pvalues.get("children_yes", float("nan"))

odds_ratio = float(math.exp(coef)) if math.isfinite(coef) else float("nan")

# Score construction
# Direction: children reduces affairs if prop_diff > 0 and OR < 1
# Base magnitude from proportion difference
base = prop_diff * 200.0  # 0.2 diff -> 40

# Incorporate odds ratio magnitude if available
if math.isfinite(odds_ratio):
    log_or = abs(math.log(odds_ratio))
    or_component = min(1.0, log_or / 0.7) * 40.0  # up to 40 points
else:
    or_component = 0.0

score = base

# Align direction using odds ratio if contradiction
if math.isfinite(odds_ratio):
    if odds_ratio < 1:
        score += or_component
    elif odds_ratio > 1:
        score -= or_component

# Adjust by p-value for children coefficient
if math.isfinite(pval):
    if pval < 0.01:
        score *= 1.4
    elif pval < 0.05:
        score *= 1.2
    elif pval > 0.2:
        score *= 0.7

# Clamp and round
score = max(-100, min(100, score))
score_int = int(round(score))

# Save conclusion
with open("conclusion.txt", "w", encoding="utf-8") as f:
    f.write(str(score_int))

# Also write a small JSON for debugging (not required)
summary = {
    "prop_yes": prop_yes,
    "prop_no": prop_no,
    "prop_diff": prop_diff,
    "mean_yes": mean_yes,
    "mean_no": mean_no,
    "mean_diff": mean_diff,
    "odds_ratio": odds_ratio,
    "pval": pval,
    "score": score,
    "score_int": score_int,
}
with open("analysis_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
