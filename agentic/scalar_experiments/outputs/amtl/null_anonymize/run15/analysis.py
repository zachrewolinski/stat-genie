import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy

DATA_PATH = "amtl.csv"

# Load data
_df = pd.read_csv(DATA_PATH)

# Required columns
cols = ["feature1", "feature3", "feature4", "feature5", "feature7", "feature8"]
_df = _df[cols].copy()

# Basic cleaning
_df = _df.dropna(subset=cols)
_df = _df[_df["feature4"] > 0]
_df = _df[_df["feature3"].between(0, _df["feature4"])]

# Prepare model data
_df["missing_rate"] = _df["feature3"] / _df["feature4"]

# Ensure categorical types
_df["feature1"] = _df["feature1"].astype("category")
_df["feature8"] = _df["feature8"].astype("category")

# Fit binomial GLM with counts (proportions + var_weights)
formula = "missing_rate ~ C(feature8) + C(feature1) + feature5 + feature7"
model = sm.GLM.from_formula(
    formula,
    data=_df,
    family=sm.families.Binomial(),
    var_weights=_df["feature4"].values,
)
res = model.fit()

# Build design matrices for counterfactual predictions
# Keep observed covariates, swap genus
nonhuman_genera = [g for g in _df["feature8"].cat.categories if g != "Homo sapiens"]
if "Homo sapiens" not in _df["feature8"].cat.categories:
    raise RuntimeError("Homo sapiens not found in feature8 categories")
if len(nonhuman_genera) == 0:
    raise RuntimeError("No non-human genera found")

base_df = _df[["feature1", "feature5", "feature7", "feature8"]].copy()

# Design info from fitted model
design_info = res.model.data.design_info

# Build exog for Homo sapiens
homo_df = base_df.copy()
homo_df["feature8"] = "Homo sapiens"
X_homo = patsy.build_design_matrices([design_info], homo_df)[0]

# Build exog for each non-human genus
X_nonhumans = []
for g in nonhuman_genera:
    tmp = base_df.copy()
    tmp["feature8"] = g
    X_g = patsy.build_design_matrices([design_info], tmp)[0]
    X_nonhumans.append(X_g)

# Helper
sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

# Point estimate difference
beta = res.params.values
p_homo = sigmoid(X_homo @ beta)

p_nonhuman_avg = np.zeros_like(p_homo)
for Xg in X_nonhumans:
    p_nonhuman_avg += sigmoid(Xg @ beta)
p_nonhuman_avg /= len(X_nonhumans)

diff = p_homo - p_nonhuman_avg
mean_diff = float(np.mean(diff))

# Uncertainty via coefficient simulation
cov = res.cov_params().values
rng = np.random.default_rng(0)
num_draws = 5000
betas = rng.multivariate_normal(beta, cov, size=num_draws)

# Compute mean diff for each draw efficiently
mean_diffs = []
for b in betas:
    p_h = sigmoid(X_homo @ b)
    p_n = np.zeros_like(p_h)
    for Xg in X_nonhumans:
        p_n += sigmoid(Xg @ b)
    p_n /= len(X_nonhumans)
    mean_diffs.append(float(np.mean(p_h - p_n)))

mean_diffs = np.array(mean_diffs)
std_diff = float(np.std(mean_diffs, ddof=1))

prob_diff_gt0 = float(np.mean(mean_diffs > 0))

# z-score-like signal
z = 0.0
if std_diff > 0:
    z = mean_diff / std_diff

# Map to Likert scalar [-100, 100]
sign = 1.0 if mean_diff > 0 else (-1.0 if mean_diff < 0 else 0.0)
confidence = 2.0 * abs(prob_diff_gt0 - 0.5)  # 0..1
strength = min(1.0, (abs(z) / 3.0 + abs(mean_diff) / 0.10) / 2.0)  # combine stat + effect
scalar = int(round(100.0 * sign * confidence * strength))

# Clamp
scalar = max(-100, min(100, scalar))

# Write conclusion
with open("conclusion.txt", "w", encoding="utf-8") as f:
    f.write(str(scalar))

# Optional: save a small analysis summary for debugging
summary = {
    "mean_diff": mean_diff,
    "std_diff": std_diff,
    "prob_diff_gt0": prob_diff_gt0,
    "z": z,
    "scalar": scalar,
}
with open("analysis_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
