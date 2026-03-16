import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('teachingratings.csv')

# Map feature names to readable
col_map = {
    'feature1': 'id',
    'feature2': 'minority',
    'feature3': 'age',
    'feature4': 'gender',
    'feature5': 'single_credit',
    'feature6': 'beauty',
    'feature7': 'rating',
    'feature8': 'division',
    'feature9': 'native',
    'feature10': 'tenure',
    'feature11': 'students_rated',
    'feature12': 'students_enrolled',
    'feature13': 'instructor_id'
}
df = df.rename(columns=col_map)

# Basic stats
n = len(df)

# Simple correlation
corr = df['beauty'].corr(df['rating'])

# Simple OLS
model_simple = smf.ols('rating ~ beauty', data=df).fit()

# Multivariate OLS controlling for observed covariates
# Use categorical variables as categories
formula = (
    'rating ~ beauty + age + C(gender) + C(minority) + C(single_credit) + '
    'C(division) + C(native) + C(tenure) + students_rated + students_enrolled'
)
model_full = smf.ols(formula, data=df).fit()

# Extract coefficients
simple_coef = model_simple.params['beauty']
simple_p = model_simple.pvalues['beauty']
full_coef = model_full.params['beauty']
full_p = model_full.pvalues['beauty']

# R-squared for context
simple_r2 = model_simple.rsquared
full_r2 = model_full.rsquared

# Build a compact result for later narrative
results = {
    'n': n,
    'corr': corr,
    'simple_coef': simple_coef,
    'simple_p': simple_p,
    'simple_r2': simple_r2,
    'full_coef': full_coef,
    'full_p': full_p,
    'full_r2': full_r2,
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
