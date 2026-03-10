import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.formula.api as smf

# Load data
path = "teachingratings.csv"
df = pd.read_csv(path)

# Map columns by description
beauty = "feature6"
rating = "feature7"

# Basic clean: drop missing
sub = df[[beauty, rating]].dropna()

n = len(sub)

# Pearson correlation
corr, corr_p = stats.pearsonr(sub[beauty], sub[rating])

# Simple regression
model_simple = smf.ols(f"{rating} ~ {beauty}", data=df).fit()

# Multiple regression with controls
# Controls based on metadata
controls = [
    "feature2",  # minority
    "feature3",  # age
    "feature4",  # gender
    "feature5",  # single-credit elective
    "feature8",  # upper/lower division
    "feature9",  # native English
    "feature10", # tenure track
    "feature11", # students participated
    "feature12", # students enrolled
]

formula = f"{rating} ~ {beauty} + " + " + ".join(controls)
model_ctrl = smf.ols(formula, data=df).fit()

# Standardized effect for beauty (beta)
# Standardize beauty and rating to compute standardized coefficient
z_df = df[[beauty, rating]].dropna().copy()
z_df[beauty] = (z_df[beauty] - z_df[beauty].mean()) / z_df[beauty].std(ddof=0)
z_df[rating] = (z_df[rating] - z_df[rating].mean()) / z_df[rating].std(ddof=0)
model_std = smf.ols(f"{rating} ~ {beauty}", data=z_df).fit()

print("n", n)
print("corr", corr)
print("corr_p", corr_p)
print("simple_coef", model_simple.params[beauty])
print("simple_p", model_simple.pvalues[beauty])
print("simple_r2", model_simple.rsquared)
print("ctrl_coef", model_ctrl.params[beauty])
print("ctrl_p", model_ctrl.pvalues[beauty])
print("ctrl_r2", model_ctrl.rsquared)
print("std_beta", model_std.params[beauty])
