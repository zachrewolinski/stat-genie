import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv("crofoot.csv")

# Define variables
_df["rel_group_size"] = _df["feature7"] - _df["feature8"]
_df["rel_location"] = _df["feature6"] - _df["feature5"]  # positive => contest closer to focal

# Outcome
_y = _df["feature4"]

# Standardize predictors for effect comparability
_df["rel_group_size_z"] = (_df["rel_group_size"] - _df["rel_group_size"].mean()) / _df["rel_group_size"].std(ddof=0)
_df["rel_location_z"] = (_df["rel_location"] - _df["rel_location"].mean()) / _df["rel_location"].std(ddof=0)

X = _df[["rel_group_size", "rel_location"]]
X = sm.add_constant(X)
model = sm.Logit(_y, X).fit(disp=False)

Xz = _df[["rel_group_size_z", "rel_location_z"]]
Xz = sm.add_constant(Xz)
model_z = sm.Logit(_y, Xz).fit(disp=False)

# Individual models
X_g = sm.add_constant(_df[["rel_group_size"]])
model_g = sm.Logit(_y, X_g).fit(disp=False)

X_l = sm.add_constant(_df[["rel_location"]])
model_l = sm.Logit(_y, X_l).fit(disp=False)


def summarize(m, label):
    params = m.params
    pvals = m.pvalues
    conf = m.conf_int()
    out = []
    for k in params.index:
        out.append({
            "term": k,
            "coef": float(params[k]),
            "p": float(pvals[k]),
            "odds_ratio": float(np.exp(params[k])),
            "ci_low": float(np.exp(conf.loc[k, 0])),
            "ci_high": float(np.exp(conf.loc[k, 1])),
        })
    return {"label": label, "n": int(m.nobs), "aic": float(m.aic), "params": out}

results = {
    "combined": summarize(model, "combined"),
    "combined_z": summarize(model_z, "combined_z"),
    "group_only": summarize(model_g, "group_only"),
    "location_only": summarize(model_l, "location_only"),
    "summary": {
        "rel_group_size_mean": float(_df["rel_group_size"].mean()),
        "rel_group_size_sd": float(_df["rel_group_size"].std(ddof=0)),
        "rel_location_mean": float(_df["rel_location"].mean()),
        "rel_location_sd": float(_df["rel_location"].std(ddof=0)),
        "n": int(len(_df))
    }
}

print(pd.DataFrame(results["combined"]["params"]))
print(pd.DataFrame(results["combined_z"]["params"]))
print(pd.DataFrame(results["group_only"]["params"]))
print(pd.DataFrame(results["location_only"]["params"]))
print(results["summary"])
