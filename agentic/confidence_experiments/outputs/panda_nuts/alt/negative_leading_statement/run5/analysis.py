import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "panda_nuts.csv"

# Load data
_df = pd.read_csv(DATA_PATH)

df = _df.copy()

# Normalize categorical values
for col in ["sex", "help", "hammer"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Efficiency metric: nuts opened per second
EPS = 1e-9

df["rate"] = df["nuts_opened"] / (df["seconds"] + EPS)

# Ensure categorical
for col in ["sex", "help"]:
    if col in df.columns:
        df[col] = df[col].astype("category")

# Exclude missing
model_df = df.dropna(subset=["nuts_opened", "seconds", "age", "sex", "help"]).copy()

# Formula for count model with offset
formula = "nuts_opened ~ age + C(sex) + C(help)"

# Poisson GLM with offset and robust SE
poisson = smf.glm(
    formula=formula,
    data=model_df,
    family=sm.families.Poisson(),
    offset=np.log(model_df["seconds"] + EPS),
).fit(cov_type="HC3")

# Overdispersion check
od = poisson.deviance / poisson.df_resid if poisson.df_resid > 0 else np.nan

# Negative Binomial (discrete) with exposure (estimates alpha)
# Use exposure to model rates (log exposure offset)
nb = smf.negativebinomial(
    formula=formula,
    data=model_df,
    exposure=model_df["seconds"],
).fit(disp=False)

# Robust covariance for NB
try:
    nb_robust = nb.get_robustcov_results(cov_type="HC3")
except Exception:
    nb_robust = nb

# Extract stats

def _conf_int(res):
    ci = res.conf_int()
    return ci.rename(columns={0: "low", 1: "high"}).to_dict(orient="index")

results = {
    "n": int(model_df.shape[0]),
    "n_chimpanzees": int(model_df["chimpanzee"].nunique()) if "chimpanzee" in model_df.columns else None,
    "rate_mean": float(model_df["rate"].mean()),
    "rate_median": float(model_df["rate"].median()),
    "overdispersion": float(od),
    "poisson": {
        "params": poisson.params.to_dict(),
        "pvalues": poisson.pvalues.to_dict(),
        "conf_int": _conf_int(poisson),
    },
    "negbin": {
        "params": nb_robust.params.to_dict(),
        "pvalues": nb_robust.pvalues.to_dict(),
        "conf_int": _conf_int(nb_robust),
        "alpha": float(getattr(nb, "params")["alpha"]) if "alpha" in nb.params else None,
    },
}

# Add rate ratios (exp of coefficients) for NB
nb_rr = {k: float(np.exp(v)) for k, v in nb_robust.params.items()}
results["negbin"]["rate_ratios"] = nb_rr

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)
