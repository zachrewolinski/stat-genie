import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm


df = pd.read_csv('panda_nuts.csv')

# Keep relevant columns
cols = ['age','sex','help','nuts_opened','seconds']
missing_cols = [c for c in cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

sub = df[cols].copy()

# Clean categorical
sub['sex'] = sub['sex'].astype(str).str.strip()
sub['help'] = sub['help'].astype(str).str.strip()

# Drop missing or nonpositive seconds
sub = sub.replace({'': np.nan})
sub = sub.dropna(subset=['age','sex','help','nuts_opened','seconds'])
sub = sub[sub['seconds'] > 0]

# Efficiency: nuts per second
sub['efficiency'] = sub['nuts_opened'] / sub['seconds']

# Also log efficiency for robustness
sub['log_eff'] = np.log(sub['efficiency'] + 1e-6)

# Model: linear regression on efficiency
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=sub).fit(cov_type='HC3')
model_log = smf.ols('log_eff ~ age + C(sex) + C(help)', data=sub).fit(cov_type='HC3')

# Poisson GLM for counts with log(seconds) offset (rate model)
sub['log_seconds'] = np.log(sub['seconds'])
glm = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=sub,
    family=sm.families.Poisson(),
    offset=sub['log_seconds']
).fit(cov_type='HC3')

# Group summaries
group_sex = sub.groupby('sex')['efficiency'].agg(['mean', 'std', 'count']).reset_index()
group_help = sub.groupby('help')['efficiency'].agg(['mean', 'std', 'count']).reset_index()

summary = {
    'n_rows': int(len(sub)),
    'efficiency_mean': float(sub['efficiency'].mean()),
    'efficiency_sd': float(sub['efficiency'].std()),
    'sex_counts': sub['sex'].value_counts().to_dict(),
    'help_counts': sub['help'].value_counts().to_dict(),
    'ols_params': model.params.to_dict(),
    'ols_pvalues': model.pvalues.to_dict(),
    'ols_r2': float(model.rsquared),
    'log_params': model_log.params.to_dict(),
    'log_pvalues': model_log.pvalues.to_dict(),
    'log_r2': float(model_log.rsquared),
    'glm_params': glm.params.to_dict(),
    'glm_pvalues': glm.pvalues.to_dict(),
    'sex_efficiency_summary': group_sex.to_dict(orient='records'),
    'help_efficiency_summary': group_help.to_dict(orient='records'),
}

print(json.dumps(summary, indent=2))
