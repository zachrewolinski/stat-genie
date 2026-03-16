import pandas as pd
import numpy as np
import statsmodels.api as sm
import json

# Load data
path = "/home/chenwang/stat-genie/agentic/scalar_experiments/outputs/crofoot/null/negative_leading_statement/run20/crofoot.csv"
df = pd.read_csv(path)

# Derived variables
# Relative group size: focal minus other
# Relative location: positive means focal is closer to its home range center than the other group
# (since smaller distance to center suggests the contest is closer to that group's core area)
df["rel_size"] = df["n_focal"] - df["n_other"]
df["rel_loc"] = df["dist_other"] - df["dist_focal"]

# Basic check
n = len(df)

# Logistic regression
X = df[["rel_size", "rel_loc"]].copy()
X = sm.add_constant(X)
model = sm.Logit(df["win"], X)
res = model.fit(disp=False)

# Also compute odds ratios with 95% CI
params = res.params
conf = res.conf_int()
conf.columns = ["2.5%", "97.5%"]
odds_ratios = np.exp(params)
conf_or = np.exp(conf)

# Collect key stats
summary = {
    "n": n,
    "coefficients": params.to_dict(),
    "pvalues": res.pvalues.to_dict(),
    "odds_ratios": odds_ratios.to_dict(),
    "odds_ratio_ci": conf_or.to_dict(),
    "llf": res.llf,
    "aic": res.aic,
}

# Save summary for later inspection
with open("/home/chenwang/stat-genie/agentic/scalar_experiments/outputs/crofoot/null/negative_leading_statement/run20/model_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(res.summary())
print("\nOdds ratios (with 95% CI):")
for k in ["const", "rel_size", "rel_loc"]:
    print(k, odds_ratios[k], conf_or.loc[k, "2.5%"], conf_or.loc[k, "97.5%"], "p=", res.pvalues[k])
