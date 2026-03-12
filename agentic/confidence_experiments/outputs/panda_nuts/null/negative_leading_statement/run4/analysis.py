import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

csv_path = 'panda_nuts.csv'
df = pd.read_csv(csv_path)

# Clean / normalize
# Ensure help and sex are categorical
for col in ['sex','help','hammer']:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Efficiency: nuts opened per second (avoid divide by zero)
# seconds min is 2.5 per metadata, but guard anyway

df['efficiency'] = df['nuts_opened'] / df['seconds'].replace(0, np.nan)

# Drop rows with missing efficiency or predictors
model_df = df.dropna(subset=['efficiency','age','sex','help','chimpanzee']).copy()

# Mixed effects model with random intercept per chimpanzee
mixed_result = None
mixed_error = None
try:
    mixed = smf.mixedlm('efficiency ~ age + C(sex) + C(help)', model_df, groups=model_df['chimpanzee'])
    mixed_result = mixed.fit(reml=False, method='lbfgs')
except Exception as e:
    mixed_error = str(e)

# OLS with chimpanzee fixed effects as fallback / robustness
ols = smf.ols('efficiency ~ age + C(sex) + C(help) + C(chimpanzee)', model_df).fit()

# Simple OLS without chimpanzee effect
ols_simple = smf.ols('efficiency ~ age + C(sex) + C(help)', model_df).fit()

# Summaries to JSON-like dict
out = {
    'n_rows': int(model_df.shape[0]),
    'efficiency_summary': model_df['efficiency'].describe().to_dict(),
    'mixed_success': mixed_result is not None,
    'mixed_error': mixed_error,
}

if mixed_result is not None:
    out['mixed_params'] = mixed_result.params.to_dict()
    out['mixed_pvalues'] = mixed_result.pvalues.to_dict()
    out['mixed_bse'] = mixed_result.bse.to_dict()
    out['mixed_llf'] = mixed_result.llf
    out['mixed_aic'] = mixed_result.aic

out['ols_params'] = ols.params.to_dict()
out['ols_pvalues'] = ols.pvalues.to_dict()
out['ols_bse'] = ols.bse.to_dict()
out['ols_r2'] = ols.rsquared
out['ols_simple_params'] = ols_simple.params.to_dict()
out['ols_simple_pvalues'] = ols_simple.pvalues.to_dict()
out['ols_simple_bse'] = ols_simple.bse.to_dict()
out['ols_simple_r2'] = ols_simple.rsquared

# Also compute group comparisons for help and sex
# mean efficiency by help/sex
out['mean_eff_by_help'] = model_df.groupby('help')['efficiency'].mean().to_dict()
out['mean_eff_by_sex'] = model_df.groupby('sex')['efficiency'].mean().to_dict()

# Save results
import json
with open('analysis_results.json','w') as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
