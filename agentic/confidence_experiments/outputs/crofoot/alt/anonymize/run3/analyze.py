import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = "crofoot.csv"

# Load data
_df = pd.read_csv(DATA_PATH)

# Outcome: focal win (1/0)
_df["win"] = _df["feature4"].astype(int)

# Relative group size: focal - other
_df["rel_size"] = _df["feature7"] - _df["feature8"]

# Contest location advantage: other distance - focal distance (positive => closer to focal home range center)
_df["loc_adv"] = _df["feature6"] - _df["feature5"]

# Standardize predictors for comparability
for col in ["rel_size", "loc_adv"]:
    _df[col + "_z"] = (_df[col] - _df[col].mean()) / _df[col].std(ddof=0)

# Fit logistic regression with both predictors
X = _df[["rel_size_z", "loc_adv_z"]]
X = sm.add_constant(X)
model = sm.Logit(_df["win"], X)

result = model.fit(disp=False)

# Also fit single-predictor models for robustness
X_size = sm.add_constant(_df[["rel_size_z"]])
res_size = sm.Logit(_df["win"], X_size).fit(disp=False)

X_loc = sm.add_constant(_df[["loc_adv_z"]])
res_loc = sm.Logit(_df["win"], X_loc).fit(disp=False)

# Collect summaries
summary = {
    "n": int(_df.shape[0]),
    "rel_size_mean": float(_df["rel_size"].mean()),
    "rel_size_std": float(_df["rel_size"].std(ddof=0)),
    "loc_adv_mean": float(_df["loc_adv"].mean()),
    "loc_adv_std": float(_df["loc_adv"].std(ddof=0)),
    "model_both": {
        "params": result.params.to_dict(),
        "pvalues": result.pvalues.to_dict(),
        "llf": float(result.llf),
        "aic": float(result.aic),
        "pseudo_r2": float(1 - result.llf / result.llnull),
    },
    "model_size": {
        "params": res_size.params.to_dict(),
        "pvalues": res_size.pvalues.to_dict(),
        "llf": float(res_size.llf),
        "aic": float(res_size.aic),
        "pseudo_r2": float(1 - res_size.llf / res_size.llnull),
    },
    "model_loc": {
        "params": res_loc.params.to_dict(),
        "pvalues": res_loc.pvalues.to_dict(),
        "llf": float(res_loc.llf),
        "aic": float(res_loc.aic),
        "pseudo_r2": float(1 - res_loc.llf / res_loc.llnull),
    },
}

with open("analysis_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
