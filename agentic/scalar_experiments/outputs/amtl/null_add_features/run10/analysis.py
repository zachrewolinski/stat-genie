import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
csv_path = "amtl.csv"
info_path = "info.json"

with open(info_path, "r") as f:
    info = json.load(f)

# Relevant columns per research question
cols = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]

df = pd.read_csv(csv_path)

# Keep only relevant columns and drop missing
missing_cols = [c for c in cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

df = df[cols].copy()

# Basic cleaning
for c in ["num_amtl", "sockets", "age", "prob_male"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Filter valid rows
mask = (
    df["sockets"].notna()
    & (df["sockets"] > 0)
    & df["num_amtl"].notna()
    & (df["num_amtl"] >= 0)
    & df["age"].notna()
    & df["prob_male"].notna()
    & df["tooth_class"].notna()
    & df["genus"].notna()
)

df = df.loc[mask].copy()

# Create binary indicator for Homo sapiens
# Use explicit string match from metadata
homo_label = "Homo sapiens"
df["is_homo"] = (df["genus"] == homo_label).astype(int)

# One-hot encode tooth_class
# Use Anterior as baseline if present
tooth_dummies = pd.get_dummies(df["tooth_class"], prefix="tooth", drop_first=True)

X = pd.concat(
    [
        df[["is_homo", "age", "prob_male"]].reset_index(drop=True),
        tooth_dummies.reset_index(drop=True),
    ],
    axis=1,
)
X = sm.add_constant(X, has_constant="add")

y = df["num_amtl"].values
weights = df["sockets"].values

# Fit binomial GLM with logit link
model = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=weights)
result = model.fit()

coef = result.params.get("is_homo", np.nan)
se = result.bse.get("is_homo", np.nan)

# If something went wrong, fall back to neutral
if not np.isfinite(coef) or not np.isfinite(se) or se == 0:
    conclusion = 0
else:
    z = coef / se
    log_or = coef
    # Convert to a bounded strength score based on effect size and certainty
    strength = int(round((min(abs(z), 4) / 4) * 50 + (min(abs(log_or), 1.0) / 1.0) * 50))
    strength = min(100, max(0, strength))

    # If effect is tiny and not significant, treat as neutral
    p = result.pvalues.get("is_homo", 1.0)
    if (p > 0.2) and (abs(log_or) < 0.1):
        strength = 0

    conclusion = int(np.sign(coef) * strength)

# Ensure integer in [-100, 100]
conclusion = int(max(-100, min(100, conclusion)))

with open("conclusion.txt", "w") as f:
    f.write(str(conclusion))

