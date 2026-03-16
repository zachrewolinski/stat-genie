import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

path = "crofoot.csv"
df = pd.read_csv(path)

df["rel_size"] = df["n_focal"] - df["n_other"]
df["rel_dist"] = df["dist_other"] - df["dist_focal"]

results = {}

# Single-predictor models
for col in ["rel_size", "rel_dist"]:
    X = sm.add_constant(df[[col]])
    y = df["win"]
    res = sm.Logit(y, X).fit(disp=False)
    results[col] = {
        "coef": float(res.params[col]),
        "p": float(res.pvalues[col]),
        "or": float(np.exp(res.params[col])),
        "ci_or": [float(np.exp(res.conf_int().loc[col, 0])), float(np.exp(res.conf_int().loc[col, 1]))],
    }

# Descriptive win rates by advantage
results["win_rate_rel_size_positive"] = float(df.loc[df["rel_size"] > 0, "win"].mean())
results["win_rate_rel_size_zero"] = float(df.loc[df["rel_size"] == 0, "win"].mean())
results["win_rate_rel_size_negative"] = float(df.loc[df["rel_size"] < 0, "win"].mean())

results["win_rate_rel_dist_positive"] = float(df.loc[df["rel_dist"] > 0, "win"].mean())
results["win_rate_rel_dist_zero"] = float(df.loc[df["rel_dist"] == 0, "win"].mean())
results["win_rate_rel_dist_negative"] = float(df.loc[df["rel_dist"] < 0, "win"].mean())

results["counts"] = {
    "rel_size_positive": int((df["rel_size"] > 0).sum()),
    "rel_size_zero": int((df["rel_size"] == 0).sum()),
    "rel_size_negative": int((df["rel_size"] < 0).sum()),
    "rel_dist_positive": int((df["rel_dist"] > 0).sum()),
    "rel_dist_zero": int((df["rel_dist"] == 0).sum()),
    "rel_dist_negative": int((df["rel_dist"] < 0).sum()),
}

print(json.dumps(results, indent=2))
