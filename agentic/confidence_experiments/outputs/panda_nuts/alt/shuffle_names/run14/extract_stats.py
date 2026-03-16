import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('panda_nuts.csv')

rename_map = {
    'age': 'chimp_id',
    'hammer': 'age_years',
    'nuts_opened': 'sex',
    'sex': 'hammer_type',
    'help': 'nuts_opened',
    'chimpanzee': 'session_seconds',
    'seconds': 'helped'
}

df = df.rename(columns=rename_map)

help_map = {'y': 1, 'Y': 1, 'yes': 1, 'Yes': 1, 'N': 0, 'n': 0, 'no': 0, 'No': 0}
df['helped_bin'] = df['helped'].map(help_map)

# Efficiency

df['efficiency'] = df['nuts_opened'] / df['session_seconds']

model = smf.ols('efficiency ~ age_years + C(sex) + helped_bin', data=df).fit(cov_type='HC3')
params = model.params
pvals = model.pvalues

# Log model for robustness
min_pos = df.loc[df['efficiency'] > 0, 'efficiency'].min()
const = min_pos / 2

df['log_eff'] = np.log(df['efficiency'] + const)
model_log = smf.ols('log_eff ~ age_years + C(sex) + helped_bin', data=df).fit(cov_type='HC3')
params_log = model_log.params
pvals_log = model_log.pvalues

out = {
    'n': len(df),
    'eff_mean': df['efficiency'].mean(),
    'eff_median': df['efficiency'].median(),
    'help_counts': df['helped'].value_counts().to_dict(),
    'sex_counts': df['sex'].value_counts().to_dict(),
    'group_means_help': df.groupby('helped')['efficiency'].mean().to_dict(),
    'group_means_sex': df.groupby('sex')['efficiency'].mean().to_dict(),
    'corr_age_eff': df['efficiency'].corr(df['age_years']),
    'ols_params': params.to_dict(),
    'ols_pvals': pvals.to_dict(),
    'log_params': params_log.to_dict(),
    'log_pvals': pvals_log.to_dict(),
    'r2': model.rsquared,
    'r2_log': model_log.rsquared,
}

import json
print(json.dumps(out, indent=2))
