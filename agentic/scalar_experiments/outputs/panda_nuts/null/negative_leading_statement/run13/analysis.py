import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

DATA_PATH = "panda_nuts.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleaning
# Ensure categorical types
for col in ["sex", "help", "hammer"]:
    if col in df.columns:
        df[col] = df[col].astype("category")

# Efficiency as nuts per second
# seconds > 0 per metadata

# Poisson regression with offset (log seconds) to model rate
# nuts_opened ~ age + sex + help

# Avoid log(0) by ensuring seconds positive
if (df["seconds"] <= 0).any():
    raise ValueError("Non-positive seconds found.")

df["log_seconds"] = np.log(df["seconds"])

formula = "nuts_opened ~ age + C(sex) + C(help)"

poisson_model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Poisson(),
    offset=df["log_seconds"],
).fit(cov_type="HC3")

# Check overdispersion and fit Negative Binomial (NB2) with estimated alpha
nb2_model = None
nb2_params = None
nb2_pvalues = None
nb2_alpha = None
try:
    exog = pd.get_dummies(df[["age", "sex", "help"]], drop_first=True)
    exog = sm.add_constant(exog, has_constant="add")
    nb2_model = sm.NegativeBinomial(
        df["nuts_opened"],
        exog,
        loglike_method="nb2",
        offset=df["log_seconds"],
    ).fit(disp=0)
    nb2_params = nb2_model.params.to_dict()
    nb2_pvalues = nb2_model.pvalues.to_dict()
    if "alpha" in nb2_model.params.index:
        nb2_alpha = float(nb2_model.params["alpha"])
except Exception:
    nb2_model = None

# Linear model on efficiency for sensitivity
# Add small epsilon to avoid zero efficiency? Use raw efficiency


df["efficiency"] = df["nuts_opened"] / df["seconds"]
ols_model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit(cov_type="HC3")

# Joint Wald test for age, sex, help in poisson
# Build hypothesis matrix by terms
terms = ["age", "C(sex)[T.m]", "C(help)[T.y]"]
# Some categories might be different base; capture names
coef_names = poisson_model.params.index.tolist()

# Identify term names dynamically
age_term = "age" if "age" in coef_names else None
sex_term = None
help_term = None
for name in coef_names:
    if name.startswith("C(sex)"):
        sex_term = name
    if name.startswith("C(help)"):
        help_term = name

hypotheses = []
if age_term:
    hypotheses.append(f"{age_term} = 0")
if sex_term:
    hypotheses.append(f"{sex_term} = 0")
if help_term:
    hypotheses.append(f"{help_term} = 0")

wald_test = poisson_model.wald_test(hypotheses) if hypotheses else None

output = {
    "n": int(df.shape[0]),
    "poisson_params": poisson_model.params.to_dict(),
    "poisson_pvalues": poisson_model.pvalues.to_dict(),
    "poisson_aic": float(poisson_model.aic),
    "poisson_deviance": float(poisson_model.deviance),
    "poisson_df_resid": float(poisson_model.df_resid),
    "poisson_pearson_chi2": float(poisson_model.pearson_chi2),
    "poisson_overdispersion_ratio": float(poisson_model.pearson_chi2 / poisson_model.df_resid),
    "poisson_wald_test": {
        "stat": float(wald_test.statistic) if wald_test is not None else None,
        "pvalue": float(wald_test.pvalue) if wald_test is not None else None,
        "df": int(wald_test.df_denom) if wald_test is not None else None,
    },
    "nb2_available": nb2_model is not None,
    "nb2_params": nb2_params,
    "nb2_pvalues": nb2_pvalues,
    "nb2_alpha": nb2_alpha,
    "ols_params": ols_model.params.to_dict(),
    "ols_pvalues": ols_model.pvalues.to_dict(),
    "efficiency_summary": df["efficiency"].describe().to_dict(),
}

print(json.dumps(output, indent=2, sort_keys=True))
