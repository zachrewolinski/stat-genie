import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Basic cleaning
# Ensure numeric columns
for col in ['reader_view', 'speed', 'dyslexia_bin', 'age', 'num_words']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Subset to dyslexia participants
subset = df[df['dyslexia_bin'] == 1].copy()

# Drop missing critical values
subset = subset.dropna(subset=['reader_view', 'speed', 'uuid'])

# Two groups
rv1 = subset[subset['reader_view'] == 1]['speed']
rv0 = subset[subset['reader_view'] == 0]['speed']

# Summary stats
summary = {
    'n_total': int(subset.shape[0]),
    'n_rv1': int(rv1.shape[0]),
    'n_rv0': int(rv0.shape[0]),
    'mean_rv1': float(rv1.mean()) if rv1.shape[0] else np.nan,
    'mean_rv0': float(rv0.mean()) if rv0.shape[0] else np.nan,
    'median_rv1': float(rv1.median()) if rv1.shape[0] else np.nan,
    'median_rv0': float(rv0.median()) if rv0.shape[0] else np.nan,
    'sd_rv1': float(rv1.std(ddof=1)) if rv1.shape[0] else np.nan,
    'sd_rv0': float(rv0.std(ddof=1)) if rv0.shape[0] else np.nan,
}

# Welch t-test
if rv1.shape[0] > 1 and rv0.shape[0] > 1:
    ttest = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
    ttest_res = {'t_stat': float(ttest.statistic), 'p_value': float(ttest.pvalue)}
else:
    ttest_res = {'t_stat': np.nan, 'p_value': np.nan}

# Mann-Whitney U test (two-sided)
try:
    if rv1.shape[0] > 0 and rv0.shape[0] > 0:
        mwu = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
        mwu_res = {'u_stat': float(mwu.statistic), 'p_value': float(mwu.pvalue)}
    else:
        mwu_res = {'u_stat': np.nan, 'p_value': np.nan}
except Exception:
    mwu_res = {'u_stat': np.nan, 'p_value': np.nan}

# Effect size (Cohen's d)
if rv1.shape[0] > 1 and rv0.shape[0] > 1:
    s1 = rv1.var(ddof=1)
    s0 = rv0.var(ddof=1)
    n1 = rv1.shape[0]
    n0 = rv0.shape[0]
    pooled_sd = np.sqrt(((n1 - 1) * s1 + (n0 - 1) * s0) / (n1 + n0 - 2))
    cohens_d = (rv1.mean() - rv0.mean()) / pooled_sd if pooled_sd != 0 else np.nan
else:
    cohens_d = np.nan

# Regression with covariates; use log(speed)
# Remove non-positive speeds
subset_reg = subset[subset['speed'] > 0].copy()
subset_reg['log_speed'] = np.log(subset_reg['speed'])

# Ensure categorical types for modeling
for col in ['page_id', 'device', 'gender', 'education', 'english_native', 'language']:
    if col in subset_reg.columns:
        subset_reg[col] = subset_reg[col].astype('category')

# Build formula with a few controls; avoid overfitting if columns missing
controls = []
if 'page_id' in subset_reg.columns:
    controls.append('C(page_id)')
if 'device' in subset_reg.columns:
    controls.append('C(device)')
if 'age' in subset_reg.columns:
    controls.append('age')
if 'gender' in subset_reg.columns:
    controls.append('C(gender)')
if 'education' in subset_reg.columns:
    controls.append('C(education)')
if 'english_native' in subset_reg.columns:
    controls.append('C(english_native)')
if 'num_words' in subset_reg.columns:
    controls.append('num_words')
if 'Flesch_Kincaid' in subset_reg.columns:
    controls.append('Flesch_Kincaid')

formula = 'log_speed ~ reader_view'
if controls:
    formula += ' + ' + ' + '.join(controls)

reg_res = {'coef': np.nan, 'se': np.nan, 'p_value': np.nan}
try:
    model = smf.ols(formula, data=subset_reg).fit(cov_type='cluster', cov_kwds={'groups': subset_reg['uuid']})
    if 'reader_view' in model.params:
        reg_res = {
            'coef': float(model.params['reader_view']),
            'se': float(model.bse['reader_view']),
            'p_value': float(model.pvalues['reader_view'])
        }
except Exception:
    pass

results = {
    'summary': summary,
    'ttest': ttest_res,
    'mwu': mwu_res,
    'cohens_d': float(cohens_d) if not np.isnan(cohens_d) else np.nan,
    'regression': reg_res,
    'formula': formula
}

print(json.dumps(results, indent=2))
