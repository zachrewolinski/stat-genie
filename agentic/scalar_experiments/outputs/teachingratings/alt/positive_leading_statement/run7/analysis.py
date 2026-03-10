import json
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
file_path = 'teachingratings.csv'
df = pd.read_csv(file_path)

# Basic cleaning: ensure expected columns
# Convert categorical columns to category type
cat_cols = [
    'minority','gender','credits','division','native','tenure'
]
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype('category')

# Drop rows with missing values in core variables
core_cols = ['eval','beauty']
base_df = df.dropna(subset=core_cols)

# 1) Simple correlation
corr = base_df['eval'].corr(base_df['beauty'])

# 2) Simple OLS: eval ~ beauty
model_simple = smf.ols('eval ~ beauty', data=base_df).fit()

# 3) OLS with controls (common in literature)
# Include age, gender, minority, native, tenure, division, credits, students, allstudents
# Some of these are categorical; statsmodels will handle via C().
control_cols = ['age','gender','minority','native','tenure','division','credits','students','allstudents']
control_df = base_df.dropna(subset=control_cols)

formula = 'eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students + allstudents'
model_controls = smf.ols(formula, data=control_df).fit()

# Extract effect sizes and p-values
simple_coef = model_simple.params['beauty']
simple_p = model_simple.pvalues['beauty']

controls_coef = model_controls.params['beauty']
controls_p = model_controls.pvalues['beauty']

# R-squared for models
simple_r2 = model_simple.rsquared
controls_r2 = model_controls.rsquared

# Build results dict for later use
results = {
    'n_total': len(df),
    'n_simple': int(model_simple.nobs),
    'n_controls': int(model_controls.nobs),
    'corr': corr,
    'simple_coef': simple_coef,
    'simple_p': simple_p,
    'simple_r2': simple_r2,
    'controls_coef': controls_coef,
    'controls_p': controls_p,
    'controls_r2': controls_r2,
}

with open('analysis_results.json','w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
