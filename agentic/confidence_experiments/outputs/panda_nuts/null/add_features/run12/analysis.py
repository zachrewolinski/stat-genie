import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

DATA_PATH = "panda_nuts.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleaning
# Ensure expected columns exist
required_cols = ["nuts_opened", "seconds", "age", "sex", "help", "chimpanzee"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Drop rows with missing core variables
core = df[required_cols].dropna().copy()

# Ensure positive seconds
core = core[core["seconds"] > 0].copy()

# Efficiency rate (nuts per second)
core["rate"] = core["nuts_opened"] / core["seconds"]

# Poisson GLM with offset for time
core["log_seconds"] = np.log(core["seconds"])

formula = "nuts_opened ~ age + C(sex) + C(help)"
model = smf.glm(
    formula=formula,
    data=core,
    family=sm.families.Poisson(),
    offset=core["log_seconds"],
)
result = model.fit()

# Cluster-robust SEs by individual to account for repeated sessions
try:
    rob = result.get_robustcov_results(cov_type="cluster", groups=core["chimpanzee"])
except Exception:
    rob = result

# Negative binomial (discrete) to address overdispersion, if supported
nb_error = None
nb_params = None
nb_pvalues = None
try:
    nb_model = smf.negativebinomial(formula, data=core, offset=core["log_seconds"])
    nb_result = nb_model.fit(disp=False)
    try:
        nb_rob = nb_result.get_robustcov_results(cov_type="cluster", groups=core["chimpanzee"])
    except Exception:
        nb_rob = nb_result
    nb_params = nb_rob.params
    nb_pvalues = nb_rob.pvalues
except Exception as e:
    nb_error = str(e)

# OLS on log(1 + rate) as a robustness check
core["log_rate"] = np.log1p(core["rate"])
ols = smf.ols("log_rate ~ age + C(sex) + C(help)", data=core)
ols_result = ols.fit(cov_type="cluster", cov_kwds={"groups": core["chimpanzee"]})

# Simple descriptive stats
rate_summary = core["rate"].describe()
rate_by_sex = core.groupby("sex")["rate"].mean()
rate_by_help = core.groupby("help")["rate"].mean()
rate_age_corr = core[["age", "rate"]].corr().iloc[0, 1]

# Overdispersion check (Pearson chi2 / df)
pearson_chi2 = result.pearson_chi2
od_ratio = pearson_chi2 / result.df_resid if result.df_resid > 0 else np.nan

# Collect coefficients and p-values
params = rob.params
pvalues = rob.pvalues

coef_table = []
for name in params.index:
    coef_table.append(
        {
            "term": name,
            "coef": float(params[name]),
            "rate_ratio": float(np.exp(params[name])),
            "pvalue": float(pvalues[name]),
        }
    )

nb_table = None
if nb_params is not None:
    nb_table = []
    for name in nb_params.index:
        nb_table.append(
            {
                "term": name,
                "coef": float(nb_params[name]),
                "rate_ratio": float(np.exp(nb_params[name])),
                "pvalue": float(nb_pvalues[name]),
            }
        )

ols_table = []
for name in ols_result.params.index:
    ols_table.append(
        {
            "term": name,
            "coef": float(ols_result.params[name]),
            "pvalue": float(ols_result.pvalues[name]),
        }
    )

output = {
    "n_rows": int(core.shape[0]),
    "n_chimpanzee": int(core["chimpanzee"].nunique()),
    "rate_summary": rate_summary.to_dict(),
    "rate_by_sex": rate_by_sex.to_dict(),
    "rate_by_help": rate_by_help.to_dict(),
    "rate_age_corr": float(rate_age_corr),
    "overdispersion_ratio": float(od_ratio),
    "coef_table": coef_table,
    "nb_coef_table": nb_table,
    "nb_error": nb_error,
    "ols_coef_table": ols_table,
}

with open("analysis_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
