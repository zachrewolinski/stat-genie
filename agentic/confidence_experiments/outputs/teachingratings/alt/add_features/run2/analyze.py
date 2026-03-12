import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

path = 'teachingratings.csv'
df = pd.read_csv(path)

# Focus on key variables; drop rows with missing
key_cols = [
    'eval', 'beauty', 'age', 'gender', 'minority', 'native', 'tenure',
    'division', 'credits', 'students', 'allstudents'
]
# Some columns may have missing values
analysis_df = df[key_cols].copy()
analysis_df = analysis_df.dropna()

# Basic correlation
corr = analysis_df['beauty'].corr(analysis_df['eval'])

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=analysis_df).fit(cov_type='HC3')

# Multiple OLS with controls
# Use categorical encoding for relevant factors
model_controls = smf.ols(
    'eval ~ beauty + age + students + allstudents + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits)',
    data=analysis_df
).fit(cov_type='HC3')

# Standardized effect for beauty in controlled model
# Compute standardized coefficient: beta * (sd_x / sd_y)
beauty_sd = analysis_df['beauty'].std(ddof=0)
eval_sd = analysis_df['eval'].std(ddof=0)
std_beta = model_controls.params['beauty'] * (beauty_sd / eval_sd)

# Effect size in eval units for 1 SD change in beauty
sd_effect = model_controls.params['beauty'] * beauty_sd

# Two-sample t-test: high vs low beauty (top vs bottom tertile)
analysis_df['beauty_tertile'] = pd.qcut(analysis_df['beauty'], 3, labels=['low','mid','high'])
low = analysis_df.loc[analysis_df['beauty_tertile']=='low','eval']
high = analysis_df.loc[analysis_df['beauty_tertile']=='high','eval']

t_stat, p_val = stats.ttest_ind(high, low, equal_var=False)
mean_diff = high.mean() - low.mean()

output = {
    'n': int(len(analysis_df)),
    'corr': corr,
    'simple_coef': model_simple.params['beauty'],
    'simple_p': model_simple.pvalues['beauty'],
    'controls_coef': model_controls.params['beauty'],
    'controls_p': model_controls.pvalues['beauty'],
    'controls_ci': model_controls.conf_int().loc['beauty'].tolist(),
    'std_beta': std_beta,
    'sd_effect': sd_effect,
    'tertile_mean_diff': mean_diff,
    'tertile_p': p_val,
    'means': {
        'low': low.mean(),
        'mid': analysis_df.loc[analysis_df['beauty_tertile']=='mid','eval'].mean(),
        'high': high.mean()
    }
}

with open('analysis_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
