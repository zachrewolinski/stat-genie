import pandas as pd
import statsmodels.formula.api as smf
import json

# Load data
df = pd.read_csv('teachingratings.csv')

# Basic summaries
summary = {
    'n': len(df),
    'beauty_mean': df['beauty'].mean(),
    'beauty_std': df['beauty'].std(),
    'eval_mean': df['eval'].mean(),
    'eval_std': df['eval'].std(),
    'beauty_eval_corr': df['beauty'].corr(df['eval'])
}

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=df).fit()

# Controlled OLS with covariates (categorical handled by C())
model_controls = smf.ols(
    'eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students + allstudents',
    data=df
).fit()

# Robust (HC3) for controls
model_controls_robust = model_controls.get_robustcov_results(cov_type='HC3')

# Collect coefficients and p-values
results = {
    'simple_coef': model_simple.params['beauty'],
    'simple_pvalue': model_simple.pvalues['beauty'],
    'simple_ci': model_simple.conf_int().loc['beauty'].tolist(),
    'simple_r2': model_simple.rsquared,
    'controls_coef': model_controls.params['beauty'],
    'controls_pvalue': model_controls.pvalues['beauty'],
    'controls_ci': model_controls.conf_int().loc['beauty'].tolist(),
    'controls_r2': model_controls.rsquared,
    'controls_robust_pvalue': model_controls_robust.pvalues[model_controls.model.exog_names.index('beauty')],
}

# Save outputs for inspection
with open('analysis_results.json', 'w') as f:
    json.dump({'summary': summary, 'results': results}, f, indent=2)

print(json.dumps({'summary': summary, 'results': results}, indent=2))
