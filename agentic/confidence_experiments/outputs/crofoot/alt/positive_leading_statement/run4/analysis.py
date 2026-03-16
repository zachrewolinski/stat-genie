import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
path = "crofoot.csv"
df = pd.read_csv(path)

# Construct relative group size and relative location metrics
# Relative group size: focal size - other size
# Relative location: other distance - focal distance (positive means focal is closer to its home-range center)
df = df.copy()
df["rel_size"] = df["n_focal"] - df["n_other"]
df["rel_dist"] = df["dist_other"] - df["dist_focal"]

# Fit logistic regression: win ~ rel_size + rel_dist
X = df[["rel_size", "rel_dist"]]
X = sm.add_constant(X)
y = df["win"]

model = sm.Logit(y, X)
result = model.fit(disp=False)

# Compute odds ratios and confidence intervals
params = result.params
conf = result.conf_int()
odds_ratios = params.apply(lambda v: float(np.exp(v)))
conf_or = conf.apply(lambda s: s.apply(lambda v: float(np.exp(v))))

output = {
    "n": int(len(df)),
    "coefficients": params.to_dict(),
    "pvalues": result.pvalues.to_dict(),
    "odds_ratios": odds_ratios.to_dict(),
    "or_conf_int": {
        k: [float(conf_or.loc[k, 0]), float(conf_or.loc[k, 1])] for k in conf_or.index
    },
}

print(json.dumps(output, indent=2))
