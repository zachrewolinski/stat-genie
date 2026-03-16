import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('teachingratings.csv')

# Basic checks
n_rows = len(df)
missing = df.isna().sum()

# Primary variables
beauty = df['beauty']
ratings = df['allstudents']

# Correlation
corr, corr_p = stats.pearsonr(beauty, ratings)

# Simple regression
model_simple = smf.ols('allstudents ~ beauty', data=df).fit()

# Multivariate regression with controls (exclude division as it is unique per row)
# Treat categorical columns as categories
cat_cols = ['eval', 'tenure', 'prof', 'native', 'gender', 'credits']
for c in cat_cols:
    df[c] = df[c].astype('category')

model_controls = smf.ols(
    'allstudents ~ beauty + age + C(eval) + C(tenure) + C(prof) + C(native) + C(gender) + C(credits) + rownames + minority + students',
    data=df
).fit()

# Standardized effect for beauty in simple model
beauty_sd = beauty.std(ddof=1)
ratings_sd = ratings.std(ddof=1)
std_beta_simple = model_simple.params['beauty'] * beauty_sd / ratings_sd

# Standardized effect for beauty in controls model
std_beta_controls = model_controls.params['beauty'] * beauty_sd / ratings_sd

results = {
    'n_rows': n_rows,
    'missing': missing.to_dict(),
    'corr': corr,
    'corr_p': corr_p,
    'simple_coef': model_simple.params['beauty'],
    'simple_p': model_simple.pvalues['beauty'],
    'simple_r2': model_simple.rsquared,
    'controls_coef': model_controls.params['beauty'],
    'controls_p': model_controls.pvalues['beauty'],
    'controls_r2': model_controls.rsquared,
    'std_beta_simple': std_beta_simple,
    'std_beta_controls': std_beta_controls,
}

print(json.dumps(results, indent=2))
