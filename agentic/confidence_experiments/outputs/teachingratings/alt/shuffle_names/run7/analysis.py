import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = 'teachingratings.csv'

df = pd.read_csv(DATA_PATH)

# Basic cleaning
# Drop rows with missing key variables
key_cols = ['beauty', 'allstudents']
df_clean = df.dropna(subset=key_cols).copy()

# Correlation
pearson_r, pearson_p = stats.pearsonr(df_clean['beauty'], df_clean['allstudents'])

# Simple OLS
model_simple = smf.ols('allstudents ~ beauty', data=df_clean).fit()

# Build a more controlled model using available covariates.
# Treat obvious categorical columns as categorical.
cat_cols = ['eval', 'tenure', 'prof', 'native', 'gender', 'credits']
# Some columns like division and students may be identifiers; still include numeric controls for robustness.
# Use available numeric columns other than beauty/allstudents.
num_cols = [
    col for col in df_clean.columns
    if col not in ['beauty', 'allstudents'] + cat_cols
]

# Construct formula
cat_terms = ' + '.join([f'C({c})' for c in cat_cols if c in df_clean.columns])
num_terms = ' + '.join([c for c in num_cols if c in df_clean.columns])
terms = ' + '.join([t for t in [cat_terms, num_terms] if t])
formula = 'allstudents ~ beauty'
if terms:
    formula += ' + ' + terms

model_control = smf.ols(formula, data=df_clean).fit()

results = {
    'n': int(df_clean.shape[0]),
    'pearson_r': float(pearson_r),
    'pearson_p': float(pearson_p),
    'simple_coef': float(model_simple.params['beauty']),
    'simple_p': float(model_simple.pvalues['beauty']),
    'simple_r2': float(model_simple.rsquared),
    'control_coef': float(model_control.params['beauty']),
    'control_p': float(model_control.pvalues['beauty']),
    'control_r2': float(model_control.rsquared),
    'formula': formula,
}

print(json.dumps(results, indent=2))
