import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv("hurricane.csv")

# Focus on relevant hurricane variables
cols = [
    "masfem",
    "gender_mf",
    "alldeaths",
    "wind",
    "min",
    "category",
    "year",
    "ndam15",
    "elapsedyrs",
]

work = df[cols].copy()

# Basic cleaning
work = work.replace([np.inf, -np.inf], np.nan).dropna()

# Outcomes
work["log_deaths"] = np.log1p(work["alldeaths"])

# Descriptive: mean deaths by binary gender
mean_by_gender = work.groupby("gender_mf")["alldeaths"].mean()
median_by_gender = work.groupby("gender_mf")["alldeaths"].median()

print("Rows used:", len(work))
print("Mean deaths by gender (0=male,1=female):")
print(mean_by_gender)
print("Median deaths by gender (0=male,1=female):")
print(median_by_gender)

# OLS on log deaths with controls for intensity and time
formula_ols = "log_deaths ~ masfem + wind + min + category + year"
ols_model = smf.ols(formula_ols, data=work).fit(cov_type="HC3")
print("\nOLS (log deaths) summary (HC3):")
print(ols_model.summary())

# Alternate OLS using binary gender
formula_ols_bin = "log_deaths ~ gender_mf + wind + min + category + year"
ols_model_bin = smf.ols(formula_ols_bin, data=work).fit(cov_type="HC3")
print("\nOLS (log deaths) with binary gender (HC3):")
print(ols_model_bin.summary())

# Poisson regression on deaths (counts)
# Use the same controls; add small check for zeros (Poisson handles zeros)
poisson_formula = "alldeaths ~ masfem + wind + min + category + year"
poisson_model = smf.glm(poisson_formula, data=work, family=sm.families.Poisson()).fit()
print("\nPoisson (deaths) summary:")
print(poisson_model.summary())

# Extract key coefficients for quick reference
results = {
    "ols_masfem_coef": ols_model.params.get("masfem", np.nan),
    "ols_masfem_p": ols_model.pvalues.get("masfem", np.nan),
    "ols_gender_coef": ols_model_bin.params.get("gender_mf", np.nan),
    "ols_gender_p": ols_model_bin.pvalues.get("gender_mf", np.nan),
    "poisson_masfem_coef": poisson_model.params.get("masfem", np.nan),
    "poisson_masfem_p": poisson_model.pvalues.get("masfem", np.nan),
}

print("\nKey coefficients:")
for k, v in results.items():
    print(f"{k}: {v}")
