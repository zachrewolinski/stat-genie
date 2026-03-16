import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Core variables
rating_col = 'allstudents'
beauty_col = 'beauty'

# Basic correlation
r, p = stats.pearsonr(df[beauty_col], df[rating_col])

# Simple OLS
model_simple = smf.ols(f"{rating_col} ~ {beauty_col}", data=df).fit()

# Controls: choose plausible covariates that are not identifiers
# Treat categorical columns as C()
cat_cols = ['eval', 'tenure', 'prof', 'native', 'gender', 'credits']
num_cols = ['age', 'rownames', 'minority']

formula = f"{rating_col} ~ {beauty_col}"
for col in cat_cols:
    if col in df.columns:
        formula += f" + C({col})"
for col in num_cols:
    if col in df.columns:
        formula += f" + {col}"

model_controls = smf.ols(formula, data=df).fit()

# Effect size: change in rating per 1 SD of beauty
beauty_sd = df[beauty_col].std(ddof=1)
coef = model_controls.params.get(beauty_col, np.nan)
coef_simple = model_simple.params.get(beauty_col, np.nan)

# Output summary stats
print("N", len(df))
print("Pearson r", r, "p", p)
print("Simple OLS coef", coef_simple, "p", model_simple.pvalues.get(beauty_col))
print("Controls OLS coef", coef, "p", model_controls.pvalues.get(beauty_col))
print("Beauty SD", beauty_sd)
print("Effect (1 SD) simple", coef_simple * beauty_sd)
print("Effect (1 SD) controls", coef * beauty_sd)
print("R2 simple", model_simple.rsquared, "R2 controls", model_controls.rsquared)

