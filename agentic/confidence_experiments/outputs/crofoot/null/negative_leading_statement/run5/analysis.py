import json
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

DATA_PATH = Path("crofoot.csv")

df = pd.read_csv(DATA_PATH)

# Derived variables
# relative group size advantage (focal minus other)
df["size_diff"] = df["n_focal"] - df["n_other"]
# relative location advantage: positive means contest closer to focal home range
# (other group's distance minus focal group's distance)
df["location_adv"] = df["dist_other"] - df["dist_focal"]

# Logistic regression: win ~ size_diff + location_adv
X = df[["size_diff", "location_adv"]]
X = sm.add_constant(X)
model = sm.Logit(df["win"], X).fit(disp=False)

# Also run alternative model using size ratio for robustness
# Avoid divide by zero (none expected), but keep as float

df["size_ratio"] = df["n_focal"] / df["n_other"]
X_ratio = df[["size_ratio", "location_adv"]]
X_ratio = sm.add_constant(X_ratio)
model_ratio = sm.Logit(df["win"], X_ratio).fit(disp=False)

# Simple correlations (point-biserial) using logistic regression coefficients and Pearson
corr_size = df["win"].corr(df["size_diff"])
corr_loc = df["win"].corr(df["location_adv"])

results = {
    "n": len(df),
    "model": {
        "params": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "llf": float(model.llf),
        "pseudo_r2": float(model.prsquared),
    },
    "model_ratio": {
        "params": model_ratio.params.to_dict(),
        "pvalues": model_ratio.pvalues.to_dict(),
        "llf": float(model_ratio.llf),
        "pseudo_r2": float(model_ratio.prsquared),
    },
    "correlations": {
        "win_size_diff": float(corr_size),
        "win_location_adv": float(corr_loc),
    },
    "summary_stats": {
        "size_diff_mean": float(df["size_diff"].mean()),
        "location_adv_mean": float(df["location_adv"].mean()),
    },
}

Path("analysis_results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
