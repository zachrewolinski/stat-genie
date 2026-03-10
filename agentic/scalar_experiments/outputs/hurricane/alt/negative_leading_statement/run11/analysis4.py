import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

_df = pd.read_csv('hurricane.csv')
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# baseline OLS
base = smf.ols('log_deaths ~ masfem + wind + min + category', data=_df).fit()

# Trim top 5% and 10% of deaths
results = {
    'base_coef': float(base.params['masfem']),
    'base_pvalue': float(base.pvalues['masfem']),
}

for pct in [0.95, 0.90]:
    cutoff = _df['alldeaths'].quantile(pct)
    trimmed = _df[_df['alldeaths'] <= cutoff].copy()
    m = smf.ols('log_deaths ~ masfem + wind + min + category', data=trimmed).fit()
    results[f'trim_{int(pct*100)}_n'] = len(trimmed)
    results[f'trim_{int(pct*100)}_coef'] = float(m.params['masfem'])
    results[f'trim_{int(pct*100)}_pvalue'] = float(m.pvalues['masfem'])

import json
with open('analysis_trimmed.json','w') as f:
    json.dump(results, f, indent=2)

print(results)
