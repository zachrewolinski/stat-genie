import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "crofoot.csv"
df = pd.read_csv(path)

# Core variables
win = df["feature4"].astype(int)
size_focal = df["feature7"].astype(float)
size_other = df["feature8"].astype(float)

# Relative group size metrics
size_diff = size_focal - size_other
size_ratio = size_focal / size_other
log_size_ratio = np.log(size_ratio)

# Contest location (relative distance to home range centers)
# Positive values mean the other group is farther from its center than the focal group.
loc_diff = df["feature6"].astype(float) - df["feature5"].astype(float)

# Standardize predictors for effect size interpretation
X = pd.DataFrame({
    "size_diff": size_diff,
    "loc_diff": loc_diff,
})
X_std = (X - X.mean()) / X.std(ddof=0)
X_std = sm.add_constant(X_std)

# Fit logistic regression (standardized predictors)
model = sm.Logit(win, X_std)
res = model.fit(disp=False)

# Odds ratios per 1 SD increase
params = res.params
conf = res.conf_int()

def odds_ratio_and_ci(param_name):
    beta = params[param_name]
    lo, hi = conf.loc[param_name]
    return float(np.exp(beta)), float(np.exp(lo)), float(np.exp(hi))

or_size = odds_ratio_and_ci("size_diff")
or_loc = odds_ratio_and_ci("loc_diff")

# P-values
pvals = res.pvalues

# Also fit alternative with log size ratio to ensure robustness
X_alt = pd.DataFrame({
    "log_size_ratio": log_size_ratio,
    "loc_diff": loc_diff,
})
X_alt_std = (X_alt - X_alt.mean()) / X_alt.std(ddof=0)
X_alt_std = sm.add_constant(X_alt_std)
res_alt = sm.Logit(win, X_alt_std).fit(disp=False)

summary = {
    "n": int(len(df)),
    "pvalues_std": {k: float(v) for k, v in pvals.items()},
    "odds_ratio_std": {
        "size_diff": {"or": or_size[0], "ci_low": or_size[1], "ci_high": or_size[2]},
        "loc_diff": {"or": or_loc[0], "ci_low": or_loc[1], "ci_high": or_loc[2]},
    },
    "alt_pvalues_std": {k: float(v) for k, v in res_alt.pvalues.items()},
}

with open("analysis_results.json", "w") as f:
    json.dump(summary, f, indent=2)
