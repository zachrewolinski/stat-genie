import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Basic cleaning
_df['efficiency'] = _df['nuts_opened'] / _df['seconds']

# Ensure categorical
_df['sex'] = _df['sex'].astype('category')
_df['help'] = _df['help'].astype('category')

# OLS model with heteroskedasticity-robust SEs
ols = smf.ols('efficiency ~ age + C(sex) + C(help)', data=_df).fit(cov_type='HC3')

# Cluster-robust SEs by chimpanzee to account for repeated measures
ols_cluster = ols.get_robustcov_results(cov_type='cluster', groups=_df['chimpanzee'])

# Compute effect sizes for categorical differences using OLS predictions
# Baseline: sex=f, help=N (alphabetical baseline in statsmodels: first category)
# We'll compute marginal means at mean age
mean_age = _df['age'].mean()

def predict_eff(age, sex, help):
    return ols.predict(pd.DataFrame({'age':[age], 'sex':[sex], 'help':[help]})).iloc[0]

sex_levels = list(_df['sex'].cat.categories)
help_levels = list(_df['help'].cat.categories)

preds = {}
for s in sex_levels:
    for h in help_levels:
        preds[(s,h)] = predict_eff(mean_age, s, h)

# Summaries
summary = {
    'n_rows': len(_df),
    'n_chimpanzees': _df['chimpanzee'].nunique(),
    'efficiency_mean': _df['efficiency'].mean(),
    'efficiency_std': _df['efficiency'].std(),
    'ols_params': ols.params.to_dict(),
    'ols_pvalues': ols.pvalues.to_dict(),
    'ols_cluster_pvalues': dict(zip(ols.params.index, ols_cluster.pvalues)),
    'ols_r2': float(ols.rsquared),
    'preds_at_mean_age': {f'sex={s},help={h}': float(preds[(s,h)]) for s in sex_levels for h in help_levels},
}

import json
with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(ols.summary())
print('\nOLS cluster-robust summary:')
print(ols_cluster.summary())
