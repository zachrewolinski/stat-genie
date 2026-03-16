import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = 'teachingratings.csv'

df = pd.read_csv(DATA_PATH)

# Basic sanity checks
n = len(df)

# Core variables
beauty = df['beauty']
ratings = df['allstudents']

# Correlation
corr = beauty.corr(ratings)

# Simple OLS
model_simple = smf.ols('allstudents ~ beauty', data=df).fit(cov_type='HC3')

# Build controlled model (use plausible covariates; exclude identifiers)
# Treat known categorical columns as categorical
categorical_cols = ['eval', 'tenure', 'prof', 'native', 'gender', 'credits']

# Some columns look like identifiers or counts (division, students) - avoid to reduce overfitting.
control_cols = ['age', 'rownames', 'minority'] + categorical_cols

# Build formula
control_terms = ' + '.join([
    'age',
    'rownames',
    'minority',
] + [f'C({c})' for c in categorical_cols])

formula = f'allstudents ~ beauty + {control_terms}'
model_controls = smf.ols(formula, data=df).fit(cov_type='HC3')

# Standardized effect (per 1 SD beauty)
beauty_sd = beauty.std(ddof=1)
ratings_sd = ratings.std(ddof=1)
std_beta_simple = model_simple.params['beauty'] * beauty_sd / ratings_sd
std_beta_controls = model_controls.params['beauty'] * beauty_sd / ratings_sd

results = {
    'n': n,
    'corr': corr,
    'simple_coef': model_simple.params['beauty'],
    'simple_p': model_simple.pvalues['beauty'],
    'simple_ci': model_simple.conf_int().loc['beauty'].tolist(),
    'controls_coef': model_controls.params['beauty'],
    'controls_p': model_controls.pvalues['beauty'],
    'controls_ci': model_controls.conf_int().loc['beauty'].tolist(),
    'std_beta_simple': std_beta_simple,
    'std_beta_controls': std_beta_controls,
}

print(json.dumps(results, indent=2))
