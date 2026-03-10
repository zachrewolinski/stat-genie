import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
DF = pd.read_csv('panda_nuts.csv')

# Compute efficiency: nuts opened per second
DF['efficiency'] = DF['nuts_opened'] / DF['seconds']

# Ensure categorical types
DF['sex'] = DF['sex'].astype('category')
DF['help'] = DF['help'].astype('category')

# Primary model: efficiency ~ age + sex + help, cluster-robust SEs by chimpanzee
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=DF).fit(
    cov_type='cluster', cov_kwds={'groups': DF['chimpanzee']}
)

# Extract key stats
params = model.params
conf = model.conf_int()

results = {
    'n': int(DF.shape[0]),
    'unique_chimpanzees': int(DF['chimpanzee'].nunique()),
    'efficiency_mean': float(DF['efficiency'].mean()),
    'efficiency_std': float(DF['efficiency'].std(ddof=1)),
    'age_coef': float(params['age']),
    'age_p': float(model.pvalues['age']),
    'age_ci_low': float(conf.loc['age', 0]),
    'age_ci_high': float(conf.loc['age', 1]),
    'sex_m_coef': float(params.get('C(sex)[T.m]', np.nan)),
    'sex_m_p': float(model.pvalues.get('C(sex)[T.m]', np.nan)),
    'sex_m_ci_low': float(conf.loc['C(sex)[T.m]', 0]) if 'C(sex)[T.m]' in conf.index else float('nan'),
    'sex_m_ci_high': float(conf.loc['C(sex)[T.m]', 1]) if 'C(sex)[T.m]' in conf.index else float('nan'),
    'help_y_coef': float(params.get('C(help)[T.y]', np.nan)),
    'help_y_p': float(model.pvalues.get('C(help)[T.y]', np.nan)),
    'help_y_ci_low': float(conf.loc['C(help)[T.y]', 0]) if 'C(help)[T.y]' in conf.index else float('nan'),
    'help_y_ci_high': float(conf.loc['C(help)[T.y]', 1]) if 'C(help)[T.y]' in conf.index else float('nan'),
    'r2': float(model.rsquared),
}

# A robustness check: include hammer type as covariate
model_hammer = smf.ols('efficiency ~ age + C(sex) + C(help) + C(hammer)', data=DF).fit(
    cov_type='cluster', cov_kwds={'groups': DF['chimpanzee']}
)

robust = {
    'age_p': float(model_hammer.pvalues['age']),
    'sex_m_p': float(model_hammer.pvalues.get('C(sex)[T.m]', np.nan)),
    'help_y_p': float(model_hammer.pvalues.get('C(help)[T.y]', np.nan)),
    'r2': float(model_hammer.rsquared),
}

# Save stats for manual interpretation
with open('analysis_results.json', 'w') as f:
    json.dump({'primary': results, 'hammer_control': robust}, f, indent=2)

print(json.dumps({'primary': results, 'hammer_control': robust}, indent=2))
