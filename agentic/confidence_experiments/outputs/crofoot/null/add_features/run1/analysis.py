import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "crofoot.csv"
df = pd.read_csv(path)

# Focus on relevant columns
needed = ["win", "n_focal", "n_other", "dist_focal", "dist_other"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Construct predictors
# Relative group size: difference in group size (focal - other)
df["rel_size"] = df["n_focal"] - df["n_other"]
# Relative location: positive means contest is closer to focal home range center
# (other is farther from its center than focal is from its center)
df["rel_location"] = df["dist_other"] - df["dist_focal"]

# Drop rows with missing values
model_df = df[["win", "rel_size", "rel_location"]].dropna()

# Standardize predictors for interpretability
model_df["rel_size_z"] = (model_df["rel_size"] - model_df["rel_size"].mean()) / model_df["rel_size"].std(ddof=0)
model_df["rel_location_z"] = (model_df["rel_location"] - model_df["rel_location"].mean()) / model_df["rel_location"].std(ddof=0)

X = model_df[["rel_size_z", "rel_location_z"]]
X = sm.add_constant(X)
y = model_df["win"]

# Fit logistic regression
logit = sm.Logit(y, X)
res = logit.fit(disp=False)

# Extract key stats
params = res.params.to_dict()
conf_int = res.conf_int().rename(columns={0: "ci_low", 1: "ci_high"}).to_dict(orient="index")
pvalues = res.pvalues.to_dict()

# Compute odds ratios for one SD increase
odds_ratios = {k: float(np.exp(v)) for k, v in params.items() if k != "const"}

# Simple effect size: change in predicted probability from -1 SD to +1 SD for each predictor
base = X.copy()

# Function to compute predicted prob for given z values

def pred_prob(size_z, loc_z):
    return float(res.predict([1.0, size_z, loc_z])[0])

prob_size_minus = pred_prob(-1.0, 0.0)
prob_size_plus = pred_prob(1.0, 0.0)
prob_loc_minus = pred_prob(0.0, -1.0)
prob_loc_plus = pred_prob(0.0, 1.0)

results = {
    "n": int(len(model_df)),
    "params": params,
    "pvalues": pvalues,
    "conf_int": conf_int,
    "odds_ratios": odds_ratios,
    "prob_change_rel_size": float(prob_size_plus - prob_size_minus),
    "prob_change_rel_location": float(prob_loc_plus - prob_loc_minus),
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)
