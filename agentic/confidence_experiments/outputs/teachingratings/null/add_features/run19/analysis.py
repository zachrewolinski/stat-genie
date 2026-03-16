import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Basic cleaning: drop rows with missing key variables
key_cols = ['eval', 'beauty']
df = df.dropna(subset=key_cols).copy()

# Ensure categorical columns are treated as such
cat_cols = ['minority', 'gender', 'credits', 'division', 'native', 'tenure']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Simple correlation
corr = df['beauty'].corr(df['eval'])
pearson_r, pearson_p = stats.pearsonr(df['beauty'], df['eval'])

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=df).fit(cov_type='HC3')

# OLS with controls
# Use students as numeric control (class size proxy)
formula_controls = 'eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students'
model_controls = smf.ols(formula_controls, data=df).fit(cov_type='HC3')

# Effect size: change in eval per 1 SD beauty
sd_beauty = df['beauty'].std()
coef_simple = model_simple.params['beauty']
coef_controls = model_controls.params['beauty']

# Predicted difference between top and bottom beauty quartile
q1 = df['beauty'].quantile(0.25)
q3 = df['beauty'].quantile(0.75)

pred_diff_simple = coef_simple * (q3 - q1)
pred_diff_controls = coef_controls * (q3 - q1)

# Observed mean difference by quartiles
low = df[df['beauty'] <= q1]['eval'].mean()
high = df[df['beauty'] >= q3]['eval'].mean()
obs_diff = high - low

results = {
    'n': int(df.shape[0]),
    'corr_r': float(corr),
    'corr_p': float(pearson_p),
    'simple_coef': float(coef_simple),
    'simple_p': float(model_simple.pvalues['beauty']),
    'controls_coef': float(coef_controls),
    'controls_p': float(model_controls.pvalues['beauty']),
    'sd_beauty': float(sd_beauty),
    'pred_diff_simple_q3_q1': float(pred_diff_simple),
    'pred_diff_controls_q3_q1': float(pred_diff_controls),
    'obs_diff_q3_q1': float(obs_diff),
    'model_simple_r2': float(model_simple.rsquared),
    'model_controls_r2': float(model_controls.rsquared)
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
