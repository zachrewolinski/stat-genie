import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('reading.csv')

# Focus on dyslexia individuals (binary)
if 'dyslexia_bin' in df.columns:
    df_dys = df[df['dyslexia_bin'] == 1].copy()
else:
    # Fallback: treat dyslexia > 0 as dyslexia
    df_dys = df[df['dyslexia'] > 0].copy()

# Clean speed
df_dys = df_dys.replace([np.inf, -np.inf], np.nan)
df_dys = df_dys.dropna(subset=['speed', 'reader_view'])
df_dys = df_dys[df_dys['speed'] > 0]

# Basic counts
n_rows = len(df_dys)
n_uuid = df_dys['uuid'].nunique()
counts = df_dys['reader_view'].value_counts().to_dict()

# Summary stats
summary = df_dys.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std']).to_dict()

# Welch t-test on log speed
df_dys['log_speed'] = np.log(df_dys['speed'])
grp1 = df_dys[df_dys['reader_view'] == 1]['log_speed']
grp0 = df_dys[df_dys['reader_view'] == 0]['log_speed']

ttest_res = stats.ttest_ind(grp1, grp0, equal_var=False)

# Effect size (Cohen's d) on log speed
def cohens_d(a, b):
    na = len(a)
    nb = len(b)
    sa = np.var(a, ddof=1)
    sb = np.var(b, ddof=1)
    s = np.sqrt(((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / s


d_log = cohens_d(grp1, grp0)

# Regression with controls (clustered SE by uuid)
controls = []
for col in ['page_id', 'device', 'education', 'language', 'english_native']:
    if col in df_dys.columns:
        controls.append(f'C({col})')

for col in ['num_words', 'age', 'gender', 'retake_trial', 'correct_rate', 'Flesch_Kincaid']:
    if col in df_dys.columns:
        controls.append(col)

formula = 'log_speed ~ reader_view'
if controls:
    formula += ' + ' + ' + '.join(controls)

# Ensure rows used in model have no missing values in required columns
needed_cols = ['log_speed', 'reader_view', 'uuid']
for c in controls:
    if c.startswith('C('):
        needed_cols.append(c[2:-1])
    else:
        needed_cols.append(c)

needed_cols = list(dict.fromkeys(needed_cols))
df_model = df_dys.dropna(subset=needed_cols).copy()

model = smf.ols(formula, data=df_model).fit(cov_type='cluster', cov_kwds={'groups': df_model['uuid']})

coef = model.params.get('reader_view', np.nan)
pval = model.pvalues.get('reader_view', np.nan)

# Convert log coefficient to percent change
pct_change = (np.exp(coef) - 1) * 100 if np.isfinite(coef) else np.nan

# Paired analysis by uuid (mean log speed by condition)
pivot = df_dys.pivot_table(index='uuid', columns='reader_view', values='log_speed', aggfunc='mean')
paired = pivot.dropna()
paired_res = None
if len(paired) >= 5:
    paired_res = stats.ttest_rel(paired[1], paired[0])

# Collect results
results = {
    'n_rows': n_rows,
    'n_uuid': n_uuid,
    'counts_reader_view': counts,
    'summary_speed_by_reader_view': summary,
    'ttest_log_speed': {
        't_stat': float(ttest_res.statistic),
        'p_value': float(ttest_res.pvalue),
        'cohens_d_log': float(d_log),
    },
    'regression_log_speed': {
        'formula': formula,
        'coef_reader_view': float(coef),
        'p_value_reader_view': float(pval),
        'pct_change_reader_view': float(pct_change),
        'nobs': int(model.nobs),
    },
}
if paired_res is not None:
    results['paired_log_speed'] = {
        'n_pairs': int(len(paired)),
        't_stat': float(paired_res.statistic),
        'p_value': float(paired_res.pvalue),
    }

print(json.dumps(results, indent=2))
