import json
import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
DATA_PATH = "amtl.csv"

df = pd.read_csv(DATA_PATH)

# Keep relevant columns
cols = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
df = df[cols].copy()

# Basic cleaning
# Drop rows with missing or invalid values
for c in cols:
    df = df[df[c].notna()]

# Ensure numeric columns are numeric
for c in ["num_amtl", "sockets", "age", "prob_male"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df[df[["num_amtl", "sockets", "age", "prob_male"]].notna().all(axis=1)]

# Filter invalid counts
# sockets must be positive and num_amtl between 0 and sockets
mask = (df["sockets"] > 0) & (df["num_amtl"] >= 0) & (df["num_amtl"] <= df["sockets"])
df = df[mask].copy()

# Ensure categorical columns are strings
for c in ["tooth_class", "genus"]:
    df[c] = df[c].astype(str)

# Define formula for binomial GLM (logit)
# Use Pan as reference genus
formula = "amtl_rate ~ C(genus, Treatment(reference='Pan')) + age + prob_male + C(tooth_class)"

# Response as proportion with binomial weights
# Avoid division by zero (already filtered)
df["amtl_rate"] = df["num_amtl"] / df["sockets"]

model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df["sockets"],
)
result = model.fit()

# Build standardized predictions
base = df[["age", "prob_male", "tooth_class", "genus"]].copy()

genera = ["Homo sapiens", "Pan", "Pongo", "Papio"]

# Ensure all genera present
present = set(df["genus"].unique())
missing = [g for g in genera if g not in present]
if missing:
    raise RuntimeError(f"Missing genera in data: {missing}")

# Use model design info to build matrices
info = result.model.data.design_info
link_inv = result.family.link.inverse

# Precompute design matrices for each genus
exog_by_genus = {}
for g in genera:
    tmp = base.copy()
    tmp["genus"] = g
    exog = patsy.build_design_matrices([info], tmp, return_type="dataframe")[0]
    exog_by_genus[g] = np.asarray(exog)


def avg_pred_for_genus(params, genus_value):
    exog = exog_by_genus[genus_value]
    linpred = exog @ params
    p = link_inv(linpred)
    return float(np.mean(p))

# Point estimates
params_vec = result.params.values
avg_pred = {g: avg_pred_for_genus(params_vec, g) for g in genera}

# Non-human mean (equal weight among three non-human genera)
nonhuman_mean = np.mean([avg_pred[g] for g in genera if g != "Homo sapiens"])
point_diff = avg_pred["Homo sapiens"] - nonhuman_mean

# Monte Carlo for uncertainty
rng = np.random.default_rng(0)
params = result.params.values
cov = result.cov_params().values

n_draws = 2000
param_draws = rng.multivariate_normal(params, cov, size=n_draws)

# Vectorized predictions for each genus
def avg_preds_for_genus_draws(genus_value):
    exog = exog_by_genus[genus_value]  # (n, p)
    linpred = exog @ param_draws.T     # (n, draws)
    p = link_inv(linpred)
    return p.mean(axis=0)              # (draws,)

homo_preds = avg_preds_for_genus_draws("Homo sapiens")
nonhuman_preds = (
    avg_preds_for_genus_draws("Pan")
    + avg_preds_for_genus_draws("Pongo")
    + avg_preds_for_genus_draws("Papio")
) / 3.0

diff = homo_preds - nonhuman_preds

prob_gt0 = float(np.mean(diff > 0))

# Likert mapping: probability-based
score = int(round((prob_gt0 - 0.5) * 200))
score = max(-100, min(100, score))

# Save detailed results for inspection
summary = {
    "avg_pred": avg_pred,
    "nonhuman_mean": nonhuman_mean,
    "point_diff": point_diff,
    "prob_gt0": prob_gt0,
    "score": score,
}

with open("analysis_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

# Write conclusion
with open("conclusion.txt", "w", encoding="utf-8") as f:
    f.write(str(score))
