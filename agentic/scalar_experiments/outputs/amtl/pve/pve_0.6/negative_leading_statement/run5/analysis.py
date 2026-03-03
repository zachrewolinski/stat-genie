import json
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleaning
_df = _df.dropna(subset=['num_amtl', 'age', 'prob_male', 'tooth_class', 'genus', 'specimen'])

# Indicator for modern humans
_df['human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Fit OLS with cluster-robust SE by specimen
formula = 'num_amtl ~ human + age + prob_male + C(tooth_class)'
model = smf.ols(formula, data=_df).fit()
robust = model.get_robustcov_results(cov_type='cluster', groups=_df['specimen'])

# Extract coefficient for human (robust results return arrays)
param_names = robust.model.exog_names
params = dict(zip(param_names, robust.params))
ses = dict(zip(param_names, robust.bse))
pvals = dict(zip(param_names, robust.pvalues))
ci = pd.DataFrame(robust.conf_int(), index=param_names, columns=['low', 'high'])

b = params['human']
se = ses['human']
p = pvals['human']
ci_low, ci_high = ci.loc['human', 'low'], ci.loc['human', 'high']

# Summary stats by genus
summary = _df.groupby('genus')['num_amtl'].agg(['mean', 'std', 'count'])

# Also compare with genus-level model (Homo as reference) for robustness
model_genus = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=_df).fit()
robust_genus = model_genus.get_robustcov_results(cov_type='cluster', groups=_df['specimen'])

# Collect non-human genus coefficients vs Homo (reference)
genus_param_names = robust_genus.model.exog_names
coef_genus = {
    name: float(val)
    for name, val in zip(genus_param_names, robust_genus.params)
    if name.startswith('C(genus)')
}

# Save a JSON with key results for later use
results = {
    'n_rows': int(len(_df)),
    'human_coef': float(b),
    'human_se': float(se),
    'human_p': float(p),
    'human_ci_low': float(ci_low),
    'human_ci_high': float(ci_high),
    'genus_summary': summary.reset_index().to_dict(orient='records'),
    'genus_model_coefs': {k: float(v) for k, v in coef_genus.items()},
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
