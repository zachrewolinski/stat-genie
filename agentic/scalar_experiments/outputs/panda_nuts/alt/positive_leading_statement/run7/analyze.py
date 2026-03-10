import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices

DATA_PATH = "panda_nuts.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleaning
# Ensure categorical variables are treated as such
for col in ["sex", "help", "hammer"]:
    if col in df.columns:
        df[col] = df[col].astype("category")

# Efficiency as nuts opened per second
# (avoid division by zero; seconds min is > 0 per metadata)
df["rate"] = df["nuts_opened"] / df["seconds"]

def summarize_result(result):
    params = result.params
    conf = result.conf_int()
    conf.columns = ["ci_low", "ci_high"]
    rows = []
    for name in params.index:
        if name == "Intercept":
            continue
        coef = params[name]
        p = result.pvalues[name]
        ci_low = conf.loc[name, "ci_low"]
        ci_high = conf.loc[name, "ci_high"]
        irr = np.exp(coef)
        irr_low = np.exp(ci_low)
        irr_high = np.exp(ci_high)
        rows.append(
            {
                "term": name,
                "coef": float(coef),
                "p_value": float(p),
                "irr": float(irr),
                "irr_low": float(irr_low),
                "irr_high": float(irr_high),
            }
        )
    return rows


# Poisson GLM with log(seconds) offset (standard for rate modeling)
poisson_model = smf.glm(
    formula="nuts_opened ~ age + C(sex) + C(help)",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["seconds"]),
)
poisson_result = poisson_model.fit(cov_type="HC0")  # robust SE for overdispersion

# Compute dispersion (deviance/df)
dispersion = (
    poisson_result.deviance / poisson_result.df_resid
    if poisson_result.df_resid > 0
    else np.nan
)

# Negative Binomial model (nb2) with exposure (seconds) for rate modeling
y, X = dmatrices("nuts_opened ~ age + C(sex) + C(help)", data=df, return_type="dataframe")
nb_model = sm.NegativeBinomial(y, X, loglike_method="nb2", exposure=df["seconds"])
nb_result = nb_model.fit(disp=0)
# Statsmodels' NegativeBinomialResults may not expose get_robustcov_results
# in some versions, so fall back to the private method when needed.
if hasattr(nb_result, "get_robustcov_results"):
    nb_result_robust = nb_result.get_robustcov_results(cov_type="HC0")
else:
    nb_result_robust = nb_result._get_robustcov_results(cov_type="HC0")
    if nb_result_robust is None:
        # Some versions mutate in-place and return None
        nb_result_robust = nb_result

# Save a JSON-friendly summary to stdout
nb_alpha = np.nan
if hasattr(nb_result, "params_alpha"):
    nb_alpha = float(nb_result.params_alpha)
elif hasattr(nb_result, "params") and "alpha" in nb_result.params.index:
    nb_alpha = float(nb_result.params["alpha"])

out = {
    "n_rows": int(df.shape[0]),
    "dispersion": float(dispersion),
    "poisson_terms": summarize_result(poisson_result),
    "nb_terms": summarize_result(nb_result_robust),
    "nb_alpha": float(nb_alpha),
}

print(json.dumps(out, indent=2))
