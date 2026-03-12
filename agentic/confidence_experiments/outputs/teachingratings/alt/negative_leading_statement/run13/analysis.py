import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic cleaning
# Ensure categorical columns are treated as category
categorical_cols = ['minority', 'gender', 'credits', 'division', 'native', 'tenure']
for col in categorical_cols:
    if col in _df.columns:
        _df[col] = _df[col].astype('category')

# Drop rows with missing values in relevant columns
cols_needed = ['eval', 'beauty', 'age', 'students', 'allstudents'] + categorical_cols
_df_clean = _df.dropna(subset=cols_needed).copy()

n = len(_df_clean)

# Pearson correlation
r, p_corr = stats.pearsonr(_df_clean['beauty'], _df_clean['eval'])

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=_df_clean).fit(cov_type='HC3')

# Multiple OLS with controls
formula_controls = (
    'eval ~ beauty + age + students + allstudents '
    '+ C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure)'
)
model_controls = smf.ols(formula_controls, data=_df_clean).fit(cov_type='HC3')

# Standardized effect (beauty in SD units) for simple model
beauty_sd = _df_clean['beauty'].std(ddof=1)
# coefficient per 1 SD of beauty
simple_slope = model_simple.params['beauty']
std_effect_simple = simple_slope * beauty_sd

# Extract key results
results = {
    'n': int(n),
    'pearson_r': float(r),
    'pearson_p': float(p_corr),
    'simple_slope': float(simple_slope),
    'simple_p': float(model_simple.pvalues['beauty']),
    'simple_r2': float(model_simple.rsquared),
    'simple_std_effect': float(std_effect_simple),
    'controls_slope': float(model_controls.params['beauty']),
    'controls_p': float(model_controls.pvalues['beauty']),
    'controls_r2': float(model_controls.rsquared),
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
