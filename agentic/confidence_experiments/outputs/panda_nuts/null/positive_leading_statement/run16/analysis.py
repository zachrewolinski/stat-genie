import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import NegativeBinomial

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Prepare categorical variables
for col in ["sex", "help", "hammer"]:
    if col in df.columns:
        df[col] = df[col].astype("category")

# Efficiency: nuts opened per second
# (seconds should be > 0 in this dataset)
df["efficiency"] = df["nuts_opened"] / df["seconds"]

# Poisson GLM with exposure time as offset
poisson = smf.glm(
    "nuts_opened ~ age + C(sex) + C(help)",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["seconds"])
).fit()

# Overdispersion check
pearson_chi2 = ((df["nuts_opened"] - poisson.mu) ** 2 / poisson.mu).sum()
dispersion = pearson_chi2 / poisson.df_resid

# Negative binomial (NB2) with exposure to handle overdispersion
nb2 = NegativeBinomial.from_formula(
    "nuts_opened ~ age + C(sex) + C(help)",
    data=df,
    exposure=df["seconds"]
).fit(disp=False)

# OLS on efficiency with cluster-robust SEs by chimpanzee
ols = smf.ols(
    "efficiency ~ age + C(sex) + C(help)",
    data=df
).fit(cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]})

# Extract key results

def extract_pvalues(model, terms):
    out = {}
    for term in terms:
        if term in model.params.index:
            out[term] = {
                "coef": float(model.params[term]),
                "p": float(model.pvalues[term])
            }
    return out

terms = ["age", "C(sex)[T.m]", "C(help)[T.y]"]

results = {
    "n": int(df.shape[0]),
    "n_chimpanzees": int(df["chimpanzee"].nunique()),
    "dispersion_poisson": float(dispersion),
    "poisson": {
        "aic": float(poisson.aic)
    },
    "nb2": {
        "aic": float(nb2.aic),
        "alpha": float(nb2.params.get("alpha", np.nan))
    },
    "nb2_results": extract_pvalues(nb2, terms),
    "ols_cluster": extract_pvalues(ols, terms),
    "ols_r2": float(ols.rsquared)
}

print(json.dumps(results, indent=2))
