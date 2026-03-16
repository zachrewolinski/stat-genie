import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "panda_nuts.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleaning
for col in ["sex", "help", "hammer"]:
    if col in df.columns:
        df[col] = df[col].astype("category")

# Efficiency: nuts opened per second
if (df["seconds"] <= 0).any():
    df = df[df["seconds"] > 0].copy()

df["efficiency"] = df["nuts_opened"] / df["seconds"]

# OLS on efficiency
ols_formula = "efficiency ~ age + C(sex) + C(help)"
ols_model = smf.ols(ols_formula, data=df).fit()

# Cluster-robust SE by chimpanzee
cluster_groups = df["chimpanzee"] if "chimpanzee" in df.columns else None
if cluster_groups is not None:
    ols_cluster = ols_model.get_robustcov_results(cov_type="cluster", groups=cluster_groups)
else:
    ols_cluster = ols_model

# Poisson GLM with offset for seconds
poisson_formula = "nuts_opened ~ age + C(sex) + C(help)"
if cluster_groups is not None:
    poisson_model = smf.glm(
        poisson_formula,
        data=df,
        family=sm.families.Poisson(),
        offset=np.log(df["seconds"])  # exposure offset
    ).fit(cov_type="cluster", cov_kwds={"groups": cluster_groups})
else:
    poisson_model = smf.glm(
        poisson_formula,
        data=df,
        family=sm.families.Poisson(),
        offset=np.log(df["seconds"])  # exposure offset
    ).fit()
poisson_cluster = poisson_model

# Overdispersion check
pearson_chi2 = (poisson_model.resid_pearson ** 2).sum()
dispersion = pearson_chi2 / poisson_model.df_resid

# Try Negative Binomial if overdispersed
nb_result = None
try:
    nb_model = sm.NegativeBinomial.from_formula(
        poisson_formula,
        data=df,
        offset=np.log(df["seconds"])
    )
    if cluster_groups is not None:
        nb_result = nb_model.fit(disp=0, cov_type="cluster", cov_kwds={"groups": cluster_groups})
    else:
        nb_result = nb_model.fit(disp=0)
except Exception:
    nb_result = None

# Joint test for all predictors in OLS (age, sex, help)
param_names = list(ols_model.params.index)
ols_params = param_names
constraints = []
for term in ols_params:
    if term == "Intercept":
        continue
    if term == "age" or term.startswith("C(sex)") or term.startswith("C(help)"):
        constraints.append(f"{term} = 0")

joint_test = None
if constraints:
    try:
        joint_test = ols_cluster.f_test(", ".join(constraints))
    except Exception:
        joint_test = None

def to_series(values, names):
    if hasattr(values, "index"):
        return values
    return pd.Series(values, index=names)

ols_params_series = to_series(ols_cluster.params, param_names)
ols_pvalues_series = to_series(ols_cluster.pvalues, param_names)

summary = {
    "n_rows": int(df.shape[0]),
    "n_chimpanzees": int(df["chimpanzee"].nunique()) if "chimpanzee" in df.columns else None,
    "efficiency_mean": float(df["efficiency"].mean()),
    "efficiency_std": float(df["efficiency"].std()),
    "ols_params": ols_params_series.to_dict(),
    "ols_pvalues": ols_pvalues_series.to_dict(),
    "poisson_params": poisson_cluster.params.to_dict(),
    "poisson_pvalues": poisson_cluster.pvalues.to_dict(),
    "poisson_dispersion": float(dispersion),
    "joint_test": {
        "fvalue": float(joint_test.fvalue) if joint_test is not None else None,
        "pvalue": float(joint_test.pvalue) if joint_test is not None else None,
        "df_denom": float(joint_test.df_denom) if joint_test is not None else None,
        "df_num": float(joint_test.df_num) if joint_test is not None else None,
    },
}

if nb_result is not None:
    summary["nb_params"] = nb_result.params.to_dict()
    summary["nb_pvalues"] = nb_result.pvalues.to_dict()

with open("analysis_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
