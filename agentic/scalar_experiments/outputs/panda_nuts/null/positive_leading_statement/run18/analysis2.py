import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

_df = pd.read_csv("panda_nuts.csv")
_df = _df.copy()
_df["sex"] = _df["sex"].astype(str)
_df["help"] = _df["help"].astype(str)
_df["seconds"] = pd.to_numeric(_df["seconds"], errors="coerce")
_df["nuts_opened"] = pd.to_numeric(_df["nuts_opened"], errors="coerce")
_df = _df.dropna(subset=["seconds", "nuts_opened", "age", "sex", "help"])
_df["log_seconds"] = np.log(_df["seconds"])

# Poisson GLM with robust SE
poisson_model = smf.glm(
    formula="nuts_opened ~ age + C(sex) + C(help)",
    data=_df,
    family=sm.families.Poisson(),
    offset=_df["log_seconds"],
)
poisson_result = poisson_model.fit(cov_type="HC0")

# Negative Binomial (discrete) with robust SE
nb_model = smf.negativebinomial(
    formula="nuts_opened ~ age + C(sex) + C(help)",
    data=_df,
    offset=_df["log_seconds"],
)
nb_result = nb_model.fit(disp=0)
try:
    nb_result_robust = nb_result.get_robustcov_results(cov_type="HC0")
except Exception:
    nb_result_robust = nb_result

# Overdispersion check for Poisson
pearson_chi2 = float(sum(poisson_result.resid_pearson**2))
ratio = pearson_chi2 / poisson_result.df_resid if poisson_result.df_resid > 0 else np.nan


def pack(res):
    params = res.params
    pvalues = res.pvalues
    rate_ratios = np.exp(params)
    return {
        "coef": params.to_dict(),
        "pvalues": pvalues.to_dict(),
        "rate_ratios": rate_ratios.to_dict(),
    }

summary = {
    "n": int(poisson_result.nobs),
    "poisson_overdispersion_ratio": float(ratio),
    "poisson": pack(poisson_result),
    "neg_binom": pack(nb_result_robust),
    "neg_binom_alpha": float(nb_result.params.get("alpha", np.nan)),
}

print(json.dumps(summary, indent=2, sort_keys=True))
