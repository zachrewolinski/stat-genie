import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
df = pd.read_csv('reading.csv')

# Ensure speed positive
df = df[df['speed'].notna() & (df['speed'] > 0)]

# Filter dyslexic participants (dyslexia_bin == 1)
if 'dyslexia_bin' in df.columns:
    ddf = df[df['dyslexia_bin'] == 1].copy()
else:
    ddf = df[df['dyslexia'] > 0].copy()

# Basic counts
total_n = len(ddf)
unique_uuid = ddf['uuid'].nunique()

# Descriptive stats by reader_view
desc = ddf.groupby('reader_view')['speed'].agg(['count','mean','median','std']).reset_index()

# Paired analysis: average speed per uuid per reader_view
pivot = ddf.pivot_table(index='uuid', columns='reader_view', values='speed', aggfunc='mean')
paired = pivot.dropna(subset=[0, 1])
paired_n = len(paired)

paired_t = None
if paired_n > 1:
    t_stat, p_val = stats.ttest_rel(paired[1], paired[0])
    # effect size (Cohen's d for paired samples)
    diff = paired[1] - paired[0]
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
    paired_t = {'t': t_stat, 'p': p_val, 'd': d, 'mean_diff': diff.mean()}

# Regression with cluster-robust SE by uuid
# Use log-speed to mitigate skew
ddf = ddf.copy()
ddf['log_speed'] = np.log(ddf['speed'])

# Include page_id fixed effects to control for text differences
# If page_id not present, skip
formula = 'log_speed ~ reader_view'
if 'page_id' in ddf.columns:
    formula = 'log_speed ~ reader_view + C(page_id)'

model = smf.ols(formula, data=ddf).fit(cov_type='cluster', cov_kwds={'groups': ddf['uuid']})

coef = model.params.get('reader_view', np.nan)
se = model.bse.get('reader_view', np.nan)
pval = model.pvalues.get('reader_view', np.nan)

# Convert log effect to percent change
pct_change = (np.exp(coef) - 1) * 100 if pd.notna(coef) else np.nan

results = {
    'total_n': int(total_n),
    'unique_uuid': int(unique_uuid),
    'desc': desc.to_dict(orient='records'),
    'paired_n': int(paired_n),
    'paired_t': paired_t,
    'regression': {
        'coef_log': coef,
        'se_log': se,
        'pval': pval,
        'pct_change': pct_change,
        'formula': formula,
        'r2': model.rsquared,
    }
}

print(json.dumps(results, indent=2))
