import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "panda_nuts.csv"

df = pd.read_csv(DATA_PATH)

use_cols = ["nuts_opened", "seconds", "age", "sex", "help"]
missing_cols = [c for c in use_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

sub = df[use_cols].copy().dropna()

sub["sex"] = sub["sex"].astype(str).str.strip()
sub["help"] = sub["help"].astype(str).str.strip()

sub = sub[sub["seconds"] > 0].copy()
sub["rate"] = sub["nuts_opened"] / sub["seconds"]

formula = "nuts_opened ~ age + C(sex) + C(help)"

# Poisson GLM with offset
poisson_model = smf.glm(
    formula=formula,
    data=sub,
    family=sm.families.Poisson(),
    offset=np.log(sub["seconds"]),
).fit()

# Overdispersion check
pearson_chi2 = float(np.sum(poisson_model.resid_pearson ** 2))
df_resid = float(poisson_model.df_resid)
overdispersion = pearson_chi2 / df_resid if df_resid > 0 else float("nan")

# Quasi-Poisson style adjustment for standard errors if overdispersed
scale = float(overdispersion) if overdispersion > 1 else 1.0
params_pois = poisson_model.params
bse_pois = poisson_model.bse * np.sqrt(scale)
z_pois = params_pois / bse_pois
pvalues_pois = 2 * (1 - stats.norm.cdf(np.abs(z_pois)))
rate_ratios_pois = np.exp(params_pois)

# Negative binomial (discrete) with offset if possible
nb_params = {}
nb_pvalues = {}
nb_rate_ratios = {}
nb_alpha = None
nb_success = False
try:
    nb_model = smf.negativebinomial(
        formula=formula,
        data=sub,
        offset=np.log(sub["seconds"]),
    ).fit(disp=0)
    nb_params = {k: float(v) for k, v in nb_model.params.items()}
    nb_pvalues = {k: float(v) for k, v in nb_model.pvalues.items()}
    nb_rate_ratios = {k: float(np.exp(v)) for k, v in nb_model.params.items()}
    nb_alpha = float(nb_model.params.get("alpha", np.nan))
    nb_success = True
except Exception:
    nb_success = False

n = int(len(sub))
rate_mean = float(sub["rate"].mean())
rate_median = float(sub["rate"].median())

rate_by_help = sub.groupby("help")["rate"].mean().to_dict()
rate_by_sex = sub.groupby("sex")["rate"].mean().to_dict()

results = {
    "n": n,
    "rate_mean": rate_mean,
    "rate_median": rate_median,
    "overdispersion": overdispersion,
    "scale_used": scale,
    "poisson_params": {k: float(v) for k, v in params_pois.items()},
    "poisson_pvalues": {k: float(v) for k, v in zip(params_pois.index, pvalues_pois)},
    "poisson_rate_ratios": {k: float(v) for k, v in rate_ratios_pois.items()},
    "rate_by_help": {str(k): float(v) for k, v in rate_by_help.items()},
    "rate_by_sex": {str(k): float(v) for k, v in rate_by_sex.items()},
    "nb_success": nb_success,
    "nb_alpha": nb_alpha,
    "nb_params": nb_params,
    "nb_pvalues": nb_pvalues,
    "nb_rate_ratios": nb_rate_ratios,
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

