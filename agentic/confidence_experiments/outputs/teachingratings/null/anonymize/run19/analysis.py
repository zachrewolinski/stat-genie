import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('teachingratings.csv')

# Basic cleaning: ensure numeric for feature6/7
for col in ['feature6','feature7','feature3','feature11','feature12']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing in key vars
analysis_df = df.dropna(subset=['feature6','feature7'])

# Simple correlation
corr = analysis_df['feature6'].corr(analysis_df['feature7'])

# Simple OLS
model_simple = smf.ols('feature7 ~ feature6', data=analysis_df).fit(cov_type='HC3')

# OLS with controls
formula = (
    'feature7 ~ feature6 + feature3 + C(feature2) + C(feature4) + C(feature5) '
    '+ C(feature8) + C(feature9) + C(feature10) + feature11 + feature12'
)
model_controls = smf.ols(formula, data=analysis_df).fit(cov_type='HC3')

# Summaries
results = {
    'n': int(analysis_df.shape[0]),
    'corr': float(corr),
    'simple_coef': float(model_simple.params['feature6']),
    'simple_p': float(model_simple.pvalues['feature6']),
    'simple_ci_low': float(model_simple.conf_int().loc['feature6', 0]),
    'simple_ci_high': float(model_simple.conf_int().loc['feature6', 1]),
    'controls_coef': float(model_controls.params['feature6']),
    'controls_p': float(model_controls.pvalues['feature6']),
    'controls_ci_low': float(model_controls.conf_int().loc['feature6', 0]),
    'controls_ci_high': float(model_controls.conf_int().loc['feature6', 1]),
    'r2_simple': float(model_simple.rsquared),
    'r2_controls': float(model_controls.rsquared),
}

# Save results to inspect
import json
with open('analysis_results.json','w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
