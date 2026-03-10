import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "crofoot.csv"
df = pd.read_csv(path)

# Create relative measures
# Relative group size: difference and proportion

df["n_diff"] = df["n_focal"] - df["n_other"]
df["n_prop"] = df["n_focal"] / (df["n_focal"] + df["n_other"])

# Contest location: relative distance to home range centers
# Positive dist_diff means focal farther from its home range center than other group
# Negative means focal closer to its center (i.e., more on its own turf)

df["dist_diff"] = df["dist_focal"] - df["dist_other"]

# Standardize continuous predictors for interpretability
for col in ["n_diff", "n_prop", "dist_diff", "dist_focal", "dist_other"]:
    df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Primary model: win ~ n_diff + dist_diff
model1 = smf.glm("win ~ n_diff_z + dist_diff_z", data=df, family=sm.families.Binomial()).fit()

# Alternative model: win ~ n_prop + dist_diff
model2 = smf.glm("win ~ n_prop_z + dist_diff_z", data=df, family=sm.families.Binomial()).fit()

# Separate distances model
model3 = smf.glm("win ~ n_diff_z + dist_focal_z + dist_other_z", data=df, family=sm.families.Binomial()).fit()

# Null model for comparison
null_model = smf.glm("win ~ 1", data=df, family=sm.families.Binomial()).fit()

# Likelihood ratio tests
lr1 = 2 * (model1.llf - null_model.llf)
lr2 = 2 * (model2.llf - null_model.llf)

# Degrees of freedom
from scipy import stats
p_lr1 = stats.chi2.sf(lr1, df=2)
p_lr2 = stats.chi2.sf(lr2, df=2)

# Extract key results

def coef_table(model):
    res = model.summary2().tables[1]
    return res[["Coef.", "Std.Err.", "z", "P>|z|"]]

results = {
    "n": int(len(df)),
    "win_rate": float(df["win"].mean()),
    "model1": {
        "params": coef_table(model1).to_dict(),
        "llf": float(model1.llf),
        "aic": float(model1.aic),
        "p_lr": float(p_lr1),
    },
    "model2": {
        "params": coef_table(model2).to_dict(),
        "llf": float(model2.llf),
        "aic": float(model2.aic),
        "p_lr": float(p_lr2),
    },
    "model3": {
        "params": coef_table(model3).to_dict(),
        "llf": float(model3.llf),
        "aic": float(model3.aic),
    },
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
