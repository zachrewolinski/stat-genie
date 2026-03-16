import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "crofoot.csv"
df = pd.read_csv(path)

win = df["m_focal"].astype(int)
size_focal = df["f_other"].astype(float)
size_other = df["win"].astype(float)
dist_focal = df["m_other"].astype(float)
dist_other = df["n_focal"].astype(float)

rel_size = size_focal - size_other
rel_size_ratio = np.log((size_focal + 0.5) / (size_other + 0.5))
loc_adv = dist_other - dist_focal
loc_ratio = np.log((dist_other + 1) / (dist_focal + 1))

results = {}

def fit_logit(x, name):
    X = sm.add_constant(pd.DataFrame({name: x}))
    model = sm.Logit(win, X, missing="drop")
    res = model.fit(disp=False)
    results[name] = {
        "coef": res.params[name],
        "pvalue": res.pvalues[name],
        "odds_ratio": float(np.exp(res.params[name])),
    }

fit_logit(rel_size, "rel_size")
fit_logit(rel_size_ratio, "rel_size_ratio")
fit_logit(loc_adv, "loc_adv")
fit_logit(loc_ratio, "loc_ratio")

# combined model using difference measures
X = sm.add_constant(pd.DataFrame({"rel_size": rel_size, "loc_adv": loc_adv}))
res = sm.Logit(win, X, missing="drop").fit(disp=False)
results["combined"] = {
    "coef_rel_size": res.params["rel_size"],
    "pvalue_rel_size": res.pvalues["rel_size"],
    "coef_loc_adv": res.params["loc_adv"],
    "pvalue_loc_adv": res.pvalues["loc_adv"],
}

import json
with open("analysis2_output.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
