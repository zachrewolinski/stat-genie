import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = "teachingratings.csv"
df = pd.read_csv(path)

# Basic subset for analysis
cols = [
    "eval",
    "beauty",
    "age",
    "gender",
    "minority",
    "native",
    "tenure",
    "division",
    "credits",
    "students",
]

df_model = df[cols].dropna().copy()

# Pearson and Spearman correlations
pearson_r, pearson_p = stats.pearsonr(df_model["beauty"], df_model["eval"])
spearman_r, spearman_p = stats.spearmanr(df_model["beauty"], df_model["eval"])

# OLS with categorical controls and robust SEs
formula = (
    "eval ~ beauty + age + C(gender) + C(minority) + C(native) + "
    "C(tenure) + C(division) + C(credits) + students"
)
model = smf.ols(formula, data=df_model).fit(cov_type="HC3")

coef_beauty = model.params["beauty"]
se_beauty = model.bse["beauty"]
p_beauty = model.pvalues["beauty"]

# Effect size in eval points per 1 SD of beauty
beauty_sd = df_model["beauty"].std()
effect_1sd = coef_beauty * beauty_sd

# R-squared
r2 = model.rsquared
n = int(model.nobs)

# Prepare results for printing
print({
    "n": n,
    "pearson_r": pearson_r,
    "pearson_p": pearson_p,
    "spearman_r": spearman_r,
    "spearman_p": spearman_p,
    "coef_beauty": coef_beauty,
    "se_beauty": se_beauty,
    "p_beauty": p_beauty,
    "effect_1sd": effect_1sd,
    "r2": r2,
})
