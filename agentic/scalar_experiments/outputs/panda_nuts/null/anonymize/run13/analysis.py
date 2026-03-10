import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
from statsmodels.discrete.discrete_model import NegativeBinomial

# Load data
_df = pd.read_csv("panda_nuts.csv")

# Rename for clarity
_df = _df.rename(columns={
    "feature1": "id",
    "feature2": "age",
    "feature3": "sex",
    "feature4": "hammer",
    "feature5": "nuts_opened",
    "feature6": "duration_s",
    "feature7": "help"
})

# Compute efficiency (rate per second)
_df["rate_per_s"] = _df["nuts_opened"] / _df["duration_s"]
_df["log_duration"] = np.log(_df["duration_s"])

formula = "nuts_opened ~ age + C(sex) + C(help)"

# Poisson regression for counts with log-duration offset (models rate)
poisson_model = smf.glm(
    formula,
    data=_df,
    family=sm.families.Poisson(),
    offset=_df["log_duration"],
)

try:
    poisson = poisson_model.fit(cov_type="HC0")
except TypeError:
    poisson = poisson_model.fit()

# Overdispersion check
poisson_dispersion = poisson.deviance / poisson.df_resid if poisson.df_resid > 0 else np.nan

# Negative Binomial (NB2) with offset, estimating alpha
# Use patsy to build design matrices
_y, _X = dmatrices(formula, data=_df, return_type="dataframe")
nb_model = NegativeBinomial(
    _y,
    _X,
    offset=_df["log_duration"],
)
nb = nb_model.fit(disp=False)

results = {
    "n": int(_df.shape[0]),
    "mean_rate_per_s": float(_df["rate_per_s"].mean()),
    "median_rate_per_s": float(_df["rate_per_s"].median()),
    "poisson_dispersion": float(poisson_dispersion),
    "poisson": {
        "params": {k: float(v) for k, v in poisson.params.items()},
        "pvalues": {k: float(v) for k, v in poisson.pvalues.items()},
        "conf_int": {k: [float(v) for v in poisson.conf_int().loc[k].tolist()] for k in poisson.params.index},
        "cov_type": getattr(poisson, "cov_type", "nonrobust"),
    },
    "negative_binomial": {
        "params": {k: float(v) for k, v in nb.params.items()},
        "pvalues": {k: float(v) for k, v in nb.pvalues.items()},
        "conf_int": {k: [float(v) for v in nb.conf_int().loc[k].tolist()] for k in nb.params.index},
        "alpha": float(nb.model._dispersion if hasattr(nb.model, "_dispersion") else nb.params.get("alpha", np.nan)),
    },
}

print(json.dumps(results, indent=2))
