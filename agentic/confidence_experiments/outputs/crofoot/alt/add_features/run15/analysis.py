import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = Path(__file__).with_name("crofoot.csv")

df = pd.read_csv(DATA_PATH)

# Derived predictors: relative group size and relative contest location
# rel_size > 0 means focal group larger than other group
# rel_dist > 0 means focal group is farther from its home-range center than the other group is from theirs
# (i.e., contest is relatively closer to the other group)

df["rel_size"] = df["n_focal"] - df["n_other"]
df["rel_dist"] = df["dist_focal"] - df["dist_other"]

# Logistic regression
model = smf.glm("win ~ rel_size + rel_dist", data=df, family=sm.families.Binomial()).fit()

# Standardize for effect-size interpretation

df["rel_size_z"] = (df["rel_size"] - df["rel_size"].mean()) / df["rel_size"].std()
df["rel_dist_z"] = (df["rel_dist"] - df["rel_dist"].mean()) / df["rel_dist"].std()
model_z = smf.glm("win ~ rel_size_z + rel_dist_z", data=df, family=sm.families.Binomial()).fit()

# Simple point-biserial correlations
corr_size = stats.pointbiserialr(df["win"], df["rel_size"])
corr_dist = stats.pointbiserialr(df["win"], df["rel_dist"])

# Predicted probabilities at +/-1 SD
params = model_z.params

def prob(z_size, z_dist):
    lin = params["Intercept"] + params["rel_size_z"] * z_size + params["rel_dist_z"] * z_dist
    return 1 / (1 + np.exp(-lin))

summary = {
    "n": int(df.shape[0]),
    "rel_size_coef": float(model.params["rel_size"]),
    "rel_size_se": float(model.bse["rel_size"]),
    "rel_size_p": float(model.pvalues["rel_size"]),
    "rel_dist_coef": float(model.params["rel_dist"]),
    "rel_dist_se": float(model.bse["rel_dist"]),
    "rel_dist_p": float(model.pvalues["rel_dist"]),
    "prob_size_minus1sd": float(prob(-1, 0)),
    "prob_size_plus1sd": float(prob(1, 0)),
    "prob_dist_minus1sd": float(prob(0, -1)),
    "prob_dist_plus1sd": float(prob(0, 1)),
    "corr_size": float(corr_size.statistic),
    "corr_size_p": float(corr_size.pvalue),
    "corr_dist": float(corr_dist.statistic),
    "corr_dist_p": float(corr_dist.pvalue),
}

print(json.dumps(summary, indent=2))
