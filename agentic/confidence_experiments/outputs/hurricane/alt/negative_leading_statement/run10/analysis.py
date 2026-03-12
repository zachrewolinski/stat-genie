import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import NegativeBinomial as NB2

# Load data
path = "hurricane.csv"
df = pd.read_csv(path)

# Basic cleaning
# Ensure numeric columns are numeric
num_cols = [
    "masfem",
    "masfem_mturk",
    "gender_mf",
    "alldeaths",
    "wind",
    "min",
    "category",
    "ndam",
    "ndam15",
    "year",
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop rows with missing key variables
key_cols = ["alldeaths", "masfem", "wind", "min", "category"]
analysis_df = df.dropna(subset=key_cols).copy()

# Create transformed outcome
analysis_df["log_deaths"] = np.log1p(analysis_df["alldeaths"])

# 1) Correlations
corr_pearson = analysis_df["masfem"].corr(analysis_df["alldeaths"], method="pearson")
corr_spearman = analysis_df["masfem"].corr(analysis_df["alldeaths"], method="spearman")

# 2) OLS on log deaths with severity controls
ols_formula = "log_deaths ~ masfem + wind + min + category"
ols_model = smf.ols(ols_formula, data=analysis_df).fit(cov_type="HC3")

# 2b) OLS with severity controls + year trend
ols_formula_year = "log_deaths ~ masfem + wind + min + category + year"
ols_model_year = smf.ols(ols_formula_year, data=analysis_df).fit(cov_type="HC3")

# 3) Alternative femininity measure
ols_formula_mturk = "log_deaths ~ masfem_mturk + wind + min + category"
ols_model_mturk = smf.ols(ols_formula_mturk, data=analysis_df).fit(cov_type="HC3")

# 4) Binary gender indicator
ols_formula_gender = "log_deaths ~ gender_mf + wind + min + category"
ols_model_gender = smf.ols(ols_formula_gender, data=analysis_df).fit(cov_type="HC3")

# 5) Negative binomial (count) with same controls
# Use statsmodels GLM with NegativeBinomial family
nb_model = smf.glm(
    "alldeaths ~ masfem + wind + min + category",
    data=analysis_df,
    family=sm.families.NegativeBinomial(),
).fit(cov_type="HC3")

# 6) Poisson GLM with robust SEs
pois_model = smf.glm(
    "alldeaths ~ masfem + wind + min + category",
    data=analysis_df,
    family=sm.families.Poisson(),
).fit(cov_type="HC3")

# 6b) Poisson with year trend
pois_model_year = smf.glm(
    "alldeaths ~ masfem + wind + min + category + year",
    data=analysis_df,
    family=sm.families.Poisson(),
).fit(cov_type="HC3")

# 7) NB2 (discrete) with estimated dispersion
nb2_model = NB2.from_formula(
    "alldeaths ~ masfem + wind + min + category", data=analysis_df
).fit(disp=False)

# 7b) NB2 with year trend
nb2_model_year = NB2.from_formula(
    "alldeaths ~ masfem + wind + min + category + year", data=analysis_df
).fit(disp=False)

# Collect results
results = {
    "n": int(len(analysis_df)),
    "corr_pearson": float(corr_pearson),
    "corr_spearman": float(corr_spearman),
    "ols_masfem_coef": float(ols_model.params["masfem"]),
    "ols_masfem_p": float(ols_model.pvalues["masfem"]),
    "ols_year_masfem_coef": float(ols_model_year.params["masfem"]),
    "ols_year_masfem_p": float(ols_model_year.pvalues["masfem"]),
    "ols_mturk_coef": float(ols_model_mturk.params["masfem_mturk"]),
    "ols_mturk_p": float(ols_model_mturk.pvalues["masfem_mturk"]),
    "ols_gender_coef": float(ols_model_gender.params["gender_mf"]),
    "ols_gender_p": float(ols_model_gender.pvalues["gender_mf"]),
    "nb_masfem_coef": float(nb_model.params["masfem"]),
    "nb_masfem_p": float(nb_model.pvalues["masfem"]),
    "pois_masfem_coef": float(pois_model.params["masfem"]),
    "pois_masfem_p": float(pois_model.pvalues["masfem"]),
    "pois_year_masfem_coef": float(pois_model_year.params["masfem"]),
    "pois_year_masfem_p": float(pois_model_year.pvalues["masfem"]),
    "nb2_masfem_coef": float(nb2_model.params["masfem"]),
    "nb2_masfem_p": float(nb2_model.pvalues["masfem"]),
    "nb2_year_masfem_coef": float(nb2_model_year.params["masfem"]),
    "nb2_year_masfem_p": float(nb2_model_year.pvalues["masfem"]),
    "nb2_alpha": float(nb2_model.params.get("alpha", np.nan)),
}

print(json.dumps(results, indent=2))
