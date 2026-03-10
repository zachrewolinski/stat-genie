import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Basic clean: drop rows with missing
# Identify columns needed
cols = ['eval', 'beauty', 'gender', 'age', 'minority', 'native', 'tenure', 'division', 'credits', 'students', 'allstudents']
missing_cols = [c for c in cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

# Drop NA for analysis columns
use_df = df[cols].dropna()

# Simple correlation
pearson_r, pearson_p = stats.pearsonr(use_df['beauty'], use_df['eval'])

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=use_df).fit(cov_type='HC3')

# Multiple OLS with controls
formula = 'eval ~ beauty + age + students + allstudents + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits)'
model_ctrl = smf.ols(formula, data=use_df).fit(cov_type='HC3')

# Extract key stats
results = {
    'n': int(use_df.shape[0]),
    'pearson_r': pearson_r,
    'pearson_p': pearson_p,
    'simple_coef': model_simple.params['beauty'],
    'simple_p': model_simple.pvalues['beauty'],
    'simple_ci': model_simple.conf_int().loc['beauty'].tolist(),
    'ctrl_coef': model_ctrl.params['beauty'],
    'ctrl_p': model_ctrl.pvalues['beauty'],
    'ctrl_ci': model_ctrl.conf_int().loc['beauty'].tolist(),
    'simple_r2': model_simple.rsquared,
    'ctrl_r2': model_ctrl.rsquared,
}

# Save results for inspection
import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
