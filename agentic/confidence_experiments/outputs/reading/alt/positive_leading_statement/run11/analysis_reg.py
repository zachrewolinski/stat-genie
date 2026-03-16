import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

subset = pd.read_csv('reading.csv')
subset['reader_view'] = pd.to_numeric(subset['reader_view'], errors='coerce')
subset['speed'] = pd.to_numeric(subset['speed'], errors='coerce')
subset = subset[subset['dyslexia_bin'] == 1].copy()
subset = subset.dropna(subset=['reader_view','speed','uuid'])
subset = subset[subset['speed'] > 0]
subset['log_speed'] = np.log(subset['speed'])

for col in ['page_id','device','gender','education','english_native']:
    if col in subset.columns:
        subset[col] = subset[col].astype('category')

formula = 'log_speed ~ reader_view + C(page_id) + C(device) + age'

model = smf.ols(formula, data=subset).fit(cov_type='cluster', cov_kwds={'groups': subset['uuid']})
print(model.params['reader_view'])
print(model.bse['reader_view'])
print(model.pvalues['reader_view'])
