import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv("hurricane.csv")

# Basic vars

df = df.copy()

# Derived outcome

df["log_deaths"] = np.log1p(df["alldeaths"])

# Simple OLS
m1 = smf.ols("log_deaths ~ masfem", data=df).fit()

# Severity controls
m2 = smf.ols("log_deaths ~ masfem + wind + min + category", data=df).fit()

# Alternative femininity measure
m3 = smf.ols("log_deaths ~ masfem_mturk + wind + min + category", data=df).fit()

# Binary gender for reference
m4 = smf.ols("log_deaths ~ gender_mf + wind + min + category", data=df).fit()

# Negative binomial on counts
# Use GLM NB with log link
nb = smf.glm(
    "alldeaths ~ masfem + wind + min + category",
    data=df,
    family=sm.families.NegativeBinomial(),
).fit()

results = {
    "n": int(df.shape[0]),
    "zeros_deaths": int((df["alldeaths"] == 0).sum()),
    "m1": {
        "coef": float(m1.params["masfem"]),
        "p": float(m1.pvalues["masfem"]),
        "r2": float(m1.rsquared),
    },
    "m2": {
        "coef": float(m2.params["masfem"]),
        "p": float(m2.pvalues["masfem"]),
        "r2": float(m2.rsquared),
    },
    "m3": {
        "coef": float(m3.params["masfem_mturk"]),
        "p": float(m3.pvalues["masfem_mturk"]),
        "r2": float(m3.rsquared),
    },
    "m4": {
        "coef": float(m4.params["gender_mf"]),
        "p": float(m4.pvalues["gender_mf"]),
        "r2": float(m4.rsquared),
    },
    "nb": {
        "coef": float(nb.params["masfem"]),
        "p": float(nb.pvalues["masfem"]),
    },
}

print(json.dumps(results, indent=2, sort_keys=True))
