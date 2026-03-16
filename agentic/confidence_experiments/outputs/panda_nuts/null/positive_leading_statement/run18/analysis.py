import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv("panda_nuts.csv")

# Basic cleaning
_df = _df.copy()
_df["sex"] = _df["sex"].astype(str)
_df["help"] = _df["help"].astype(str)
_df["seconds"] = pd.to_numeric(_df["seconds"], errors="coerce")
_df["nuts_opened"] = pd.to_numeric(_df["nuts_opened"], errors="coerce")
_df = _df.dropna(subset=["seconds", "nuts_opened", "age", "sex", "help"])

# Poisson GLM with offset for exposure time
_df["log_seconds"] = np.log(_df["seconds"])

model = smf.glm(
    formula="nuts_opened ~ age + C(sex) + C(help)",
    data=_df,
    family=sm.families.Poisson(),
    offset=_df["log_seconds"],
)

result = model.fit(cov_type="HC0")

# Rate ratios and p-values
params = result.params
pvalues = result.pvalues
rate_ratios = np.exp(params)

# Overdispersion check
pearson_chi2 = sum(result.resid_pearson**2)
ratio = pearson_chi2 / result.df_resid if result.df_resid > 0 else np.nan

summary = {
    "n": int(result.nobs),
    "pearson_chi2": float(pearson_chi2),
    "overdispersion_ratio": float(ratio),
    "coef": params.to_dict(),
    "pvalues": pvalues.to_dict(),
    "rate_ratios": rate_ratios.to_dict(),
}

print(json.dumps(summary, indent=2, sort_keys=True))
