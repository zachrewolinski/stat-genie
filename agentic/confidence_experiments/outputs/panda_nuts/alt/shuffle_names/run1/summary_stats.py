import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv('panda_nuts.csv')

rename_map = {
    'age': 'chimpanzee_id',
    'hammer': 'age_years',
    'nuts_opened': 'sex',
    'sex': 'hammer_type',
    'help': 'nuts_opened_count',
    'chimpanzee': 'seconds',
    'seconds': 'help_received'
}

df = _df.rename(columns=rename_map)

# Types
for col in ['sex', 'help_received']:
    df[col] = df[col].astype('category')

for col in ['age_years', 'nuts_opened_count', 'seconds']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

analysis_df = df.dropna(subset=['age_years', 'sex', 'help_received', 'nuts_opened_count', 'seconds']).copy()
analysis_df['log_seconds'] = np.log(analysis_df['seconds'])

model = smf.glm(
    formula='nuts_opened_count ~ age_years + sex + help_received',
    data=analysis_df,
    family=sm.families.Poisson(),
    offset=analysis_df['log_seconds']
).fit(cov_type='HC0')

params = model.params
conf = model.conf_int()

irr = np.exp(params)
irr_ci = np.exp(conf)

summary = pd.DataFrame({
    'coef': params,
    'pval': model.pvalues,
    'irr': irr,
    'irr_ci_low': irr_ci[0],
    'irr_ci_high': irr_ci[1]
})

summary.to_csv('poisson_summary.csv')
print(summary)
