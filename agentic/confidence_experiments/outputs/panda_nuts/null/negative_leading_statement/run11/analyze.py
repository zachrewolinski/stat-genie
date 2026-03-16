import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm


df = pd.read_csv("panda_nuts.csv")

# Clean categorical variables
for col in ["sex", "help", "hammer"]:
    if col in df.columns:
        df[col] = df[col].astype(str)

# Efficiency: nuts opened per second
# Avoid division by zero; seconds min per metadata is 2.5

df["efficiency"] = df["nuts_opened"] / df["seconds"]

# Basic summaries
summary = {
    "n": int(df.shape[0]),
    "efficiency_mean": float(df["efficiency"].mean()),
    "efficiency_sd": float(df["efficiency"].std()),
    "efficiency_min": float(df["efficiency"].min()),
    "efficiency_max": float(df["efficiency"].max()),
}

# OLS regression on efficiency
ols_model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit(cov_type="HC3")

# Poisson GLM for nuts_opened with offset log(seconds)
# This models rate directly and is robust for count data
poisson_model = smf.glm(
    "nuts_opened ~ age + C(sex) + C(help)",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["seconds"]),
).fit(cov_type="HC3")

# Negative binomial GLM to account for overdispersion
nb_model = smf.glm(
    "nuts_opened ~ age + C(sex) + C(help)",
    data=df,
    family=sm.families.NegativeBinomial(),
    offset=np.log(df["seconds"]),
).fit(cov_type="HC3")

results = {
    "summary": summary,
    "ols_params": ols_model.params.to_dict(),
    "ols_pvalues": ols_model.pvalues.to_dict(),
    "ols_rsquared": float(ols_model.rsquared),
    "poisson_params": poisson_model.params.to_dict(),
    "poisson_pvalues": poisson_model.pvalues.to_dict(),
    "poisson_deviance": float(poisson_model.deviance),
    "poisson_df_resid": float(poisson_model.df_resid),
    "poisson_dispersion": float(poisson_model.deviance / poisson_model.df_resid),
    "nb_params": nb_model.params.to_dict(),
    "nb_pvalues": nb_model.pvalues.to_dict(),
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
