import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv("hurricane.csv")

# Outcomes
_df["log_deaths"] = np.log1p(_df["alldeaths"])

print("rows", len(_df))

# Correlations (descriptive only)
print("corr masfem vs log_deaths", _df["masfem"].corr(_df["log_deaths"]))
print("corr masfem_mturk vs log_deaths", _df["masfem_mturk"].corr(_df["log_deaths"]))

# OLS models with robust SEs
ols_simple = smf.ols("log_deaths ~ masfem", data=_df).fit(cov_type="HC3")
ols_controls = smf.ols("log_deaths ~ masfem + wind + min + category", data=_df).fit(cov_type="HC3")
ols_controls_mturk = smf.ols("log_deaths ~ masfem_mturk + wind + min + category", data=_df).fit(cov_type="HC3")
ols_gender = smf.ols("log_deaths ~ gender_mf + wind + min + category", data=_df).fit(cov_type="HC3")

print("\nOLS (log deaths) simple:")
print(ols_simple.summary().tables[1])
print("\nOLS (log deaths) controls:")
print(ols_controls.summary().tables[1])
print("\nOLS (log deaths) controls mturk:")
print(ols_controls_mturk.summary().tables[1])
print("\nOLS (log deaths) controls gender:")
print(ols_gender.summary().tables[1])

# Count models
# Poisson with robust SEs (misspecification-robust)
pois_controls = smf.glm(
    "alldeaths ~ masfem + wind + min + category",
    data=_df,
    family=sm.families.Poisson()
).fit(cov_type="HC3")

pois_controls_mturk = smf.glm(
    "alldeaths ~ masfem_mturk + wind + min + category",
    data=_df,
    family=sm.families.Poisson()
).fit(cov_type="HC3")

pois_gender = smf.glm(
    "alldeaths ~ gender_mf + wind + min + category",
    data=_df,
    family=sm.families.Poisson()
).fit(cov_type="HC3")

print("\nPoisson (robust) controls:")
print(pois_controls.summary().tables[1])
print("\nPoisson (robust) controls mturk:")
print(pois_controls_mturk.summary().tables[1])
print("\nPoisson (robust) controls gender:")
print(pois_gender.summary().tables[1])

# Negative Binomial (discrete) to estimate overdispersion
try:
    nb_controls = smf.negativebinomial(
        "alldeaths ~ masfem + wind + min + category",
        data=_df
    ).fit(disp=0)
    nb_controls_mturk = smf.negativebinomial(
        "alldeaths ~ masfem_mturk + wind + min + category",
        data=_df
    ).fit(disp=0)
    nb_gender = smf.negativebinomial(
        "alldeaths ~ gender_mf + wind + min + category",
        data=_df
    ).fit(disp=0)

    print("\nNegative Binomial (discrete) controls:")
    print(nb_controls.summary().tables[1])
    print("\nNegative Binomial (discrete) controls mturk:")
    print(nb_controls_mturk.summary().tables[1])
    print("\nNegative Binomial (discrete) controls gender:")
    print(nb_gender.summary().tables[1])
except Exception as e:
    print("NB discrete failed:", e)

# Descriptives
print("\nDeaths summary:")
print(_df["alldeaths"].describe())

print("\nMean log_deaths by gender_mf:")
print(_df.groupby("gender_mf")["log_deaths"].mean())

