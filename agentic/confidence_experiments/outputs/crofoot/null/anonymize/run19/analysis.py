import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Load data
path = "crofoot.csv"
df = pd.read_csv(path)

# Rename columns for clarity
cols = {
    "feature4": "focal_win",
    "feature5": "focal_dist",
    "feature6": "other_dist",
    "feature7": "focal_size",
    "feature8": "other_size",
}

df = df.rename(columns=cols)

# Construct predictors
# Relative group size: focal size minus other size (positive means focal larger)
df["rel_size_diff"] = df["focal_size"] - df["other_size"]
# Relative group size ratio
df["rel_size_ratio"] = df["focal_size"] / df["other_size"]
# Contest location advantage: other distance minus focal distance (positive means focal closer to its home range)
df["loc_adv"] = df["other_dist"] - df["focal_dist"]

y = df["focal_win"].astype(int)

def fit_logit(predictors):
    X = df[predictors].astype(float)
    X = sm.add_constant(X)
    model = sm.Logit(y, X)
    result = model.fit(disp=False)
    return result

# Main model: relative size (difference) + location advantage
result_diff = fit_logit(["rel_size_diff", "loc_adv"])

# Alternative model: relative size (ratio) + location advantage
result_ratio = fit_logit(["rel_size_ratio", "loc_adv"])

# Bivariate correlations (point-biserial)
corr_rel_diff = np.corrcoef(df["rel_size_diff"], y)[0, 1]
corr_rel_ratio = np.corrcoef(df["rel_size_ratio"], y)[0, 1]
corr_loc = np.corrcoef(df["loc_adv"], y)[0, 1]

# Simple group comparisons
win = df[df["focal_win"] == 1]
lose = df[df["focal_win"] == 0]

tt_rel_diff = stats.ttest_ind(win["rel_size_diff"], lose["rel_size_diff"], equal_var=False)
tt_loc = stats.ttest_ind(win["loc_adv"], lose["loc_adv"], equal_var=False)

summary = {
    "n": int(len(df)),
    "logit_diff": {
        "rel_size_coef": float(result_diff.params["rel_size_diff"]),
        "rel_size_p": float(result_diff.pvalues["rel_size_diff"]),
        "loc_adv_coef": float(result_diff.params["loc_adv"]),
        "loc_adv_p": float(result_diff.pvalues["loc_adv"]),
        "rel_size_or": float(np.exp(result_diff.params["rel_size_diff"])),
        "loc_adv_or": float(np.exp(result_diff.params["loc_adv"])),
        "pseudo_r2": float(result_diff.prsquared),
    },
    "logit_ratio": {
        "rel_size_coef": float(result_ratio.params["rel_size_ratio"]),
        "rel_size_p": float(result_ratio.pvalues["rel_size_ratio"]),
        "loc_adv_coef": float(result_ratio.params["loc_adv"]),
        "loc_adv_p": float(result_ratio.pvalues["loc_adv"]),
        "rel_size_or": float(np.exp(result_ratio.params["rel_size_ratio"])),
        "loc_adv_or": float(np.exp(result_ratio.params["loc_adv"])),
        "pseudo_r2": float(result_ratio.prsquared),
    },
    "corr_rel_diff": float(corr_rel_diff),
    "corr_rel_ratio": float(corr_rel_ratio),
    "corr_loc": float(corr_loc),
    "tt_rel_diff_p": float(tt_rel_diff.pvalue),
    "tt_loc_p": float(tt_loc.pvalue),
    "means": {
        "rel_size_diff_win": float(win["rel_size_diff"].mean()),
        "rel_size_diff_lose": float(lose["rel_size_diff"].mean()),
        "loc_adv_win": float(win["loc_adv"].mean()),
        "loc_adv_lose": float(lose["loc_adv"].mean()),
    },
}

print(json.dumps(summary, indent=2))
