import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv("crofoot.csv")

# Outcome: focal win (1) vs other win (0)
y = df["feature4"].astype(int)

# Predictors
rel_group_size = df["feature7"] - df["feature8"]
location_index = df["feature6"] - df["feature5"]  # positive => contest closer to focal center

# Standardize predictors for comparable effect sizes
X = pd.DataFrame({
    "rel_group_size": rel_group_size,
    "location_index": location_index,
})

X_std = (X - X.mean()) / X.std(ddof=0)
X_std = sm.add_constant(X_std, has_constant="add")

# Logistic regression
model = sm.Logit(y, X_std)
result = model.fit(disp=False)

# Also compute unstandardized model for interpretability
X_raw = sm.add_constant(X, has_constant="add")
model_raw = sm.Logit(y, X_raw)
result_raw = model_raw.fit(disp=False)

# Predicted probability difference across +/-1 SD of each predictor
# Use standardized model: coefficient corresponds to log-odds change per 1 SD.

# Helper to compute odds ratio and CI for standardized coefficients
params = result.params
conf = result.conf_int()

summary = {
    "n": int(len(df)),
    "rel_group_size_coef_std": float(params["rel_group_size"]),
    "rel_group_size_p": float(result.pvalues["rel_group_size"]),
    "rel_group_size_or_std": float(np.exp(params["rel_group_size"])),
    "rel_group_size_or_ci_std": [float(np.exp(conf.loc["rel_group_size", 0])), float(np.exp(conf.loc["rel_group_size", 1]))],
    "location_index_coef_std": float(params["location_index"]),
    "location_index_p": float(result.pvalues["location_index"]),
    "location_index_or_std": float(np.exp(params["location_index"])),
    "location_index_or_ci_std": [float(np.exp(conf.loc["location_index", 0])), float(np.exp(conf.loc["location_index", 1]))],
    "model_pseudo_r2": float(result.prsquared),
}

# Compute predicted probabilities for illustrative contrasts (raw model)
# baseline at mean predictors
mean_pred = X.mean()

# For relative group size: compare -1 SD vs +1 SD while holding location at mean
rel_sd = X["rel_group_size"].std(ddof=0)
loc_mean = mean_pred["location_index"]

X_low = sm.add_constant(
    pd.DataFrame({"rel_group_size": [mean_pred["rel_group_size"] - rel_sd], "location_index": [loc_mean]}),
    has_constant="add",
)
X_high = sm.add_constant(
    pd.DataFrame({"rel_group_size": [mean_pred["rel_group_size"] + rel_sd], "location_index": [loc_mean]}),
    has_constant="add",
)

p_low = float(result_raw.predict(X_low)[0])
p_high = float(result_raw.predict(X_high)[0])

# For location index: compare -1 SD vs +1 SD while holding rel size at mean
loc_sd = X["location_index"].std(ddof=0)
rel_mean = mean_pred["rel_group_size"]

X_loc_low = sm.add_constant(
    pd.DataFrame({"rel_group_size": [rel_mean], "location_index": [mean_pred["location_index"] - loc_sd]}),
    has_constant="add",
)
X_loc_high = sm.add_constant(
    pd.DataFrame({"rel_group_size": [rel_mean], "location_index": [mean_pred["location_index"] + loc_sd]}),
    has_constant="add",
)

p_loc_low = float(result_raw.predict(X_loc_low)[0])
p_loc_high = float(result_raw.predict(X_loc_high)[0])

summary["prob_rel_group_size_minus1sd"] = p_low
summary["prob_rel_group_size_plus1sd"] = p_high
summary["prob_location_minus1sd"] = p_loc_low
summary["prob_location_plus1sd"] = p_loc_high

print(json.dumps(summary, indent=2))
