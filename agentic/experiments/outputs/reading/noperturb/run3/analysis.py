import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('reading.csv')

# Focus on participants with dyslexia (binary indicator)
# Use dyslexia_bin if available, otherwise fall back to dyslexia >= 1
if 'dyslexia_bin' in _df.columns:
    df = _df[_df['dyslexia_bin'] == 1].copy()
else:
    df = _df[_df['dyslexia'] >= 1].copy()

# Basic cleaning
# Remove non-positive or missing speeds
if 'speed' not in df.columns:
    raise ValueError('speed column not found')

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['speed', 'reader_view'])
df = df[df['speed'] > 0]

# Group stats by reader_view
summary = (
    df.groupby('reader_view')['speed']
    .agg(['count', 'mean', 'median', 'std'])
    .rename(index={0: 'NoReaderView', 1: 'ReaderView'})
)

# Welch t-test on log speed for skewed distribution
log_speed = np.log(df['speed'])
rv = df['reader_view']
log_speed_rv = log_speed[rv == 1]
log_speed_no = log_speed[rv == 0]

ttest_res = stats.ttest_ind(log_speed_rv, log_speed_no, equal_var=False, nan_policy='omit')

# Regression with controls: page_id and num_words (content), device, language
# Use log(speed) outcome for stability
# Only include controls if present
controls = []
for col in ['page_id', 'num_words', 'device', 'language']:
    if col in df.columns:
        controls.append(col)

formula = 'np.log(speed) ~ reader_view'
if controls:
    formula += ' + ' + ' + '.join([f'C({c})' if df[c].dtype == 'object' or df[c].dtype.name == 'category' else c for c in controls])

model = smf.ols(formula, data=df).fit(cov_type='HC3')

# Collect results
results = {
    'n_dyslexia': int(df.shape[0]),
    'summary_by_reader_view': summary,
    'ttest_log_speed_stat': float(ttest_res.statistic) if np.isfinite(ttest_res.statistic) else np.nan,
    'ttest_log_speed_pvalue': float(ttest_res.pvalue) if np.isfinite(ttest_res.pvalue) else np.nan,
    'reg_coef_reader_view': float(model.params.get('reader_view', np.nan)),
    'reg_pvalue_reader_view': float(model.pvalues.get('reader_view', np.nan)),
    'reg_nobs': int(model.nobs),
}

print('Dyslexia subset size:', results['n_dyslexia'])
print('\nSummary by reader_view:')
print(summary)
print('\nWelch t-test on log(speed): stat=%.4f p=%.6f' % (results['ttest_log_speed_stat'], results['ttest_log_speed_pvalue']))
print('\nRegression (log speed) coef reader_view=%.4f p=%.6f' % (results['reg_coef_reader_view'], results['reg_pvalue_reader_view']))

# Save key results for downstream use
results_df = summary.copy()
results_df.to_csv('analysis_summary_by_reader_view.csv')

with open('analysis_results.txt', 'w') as f:
    f.write('Dyslexia subset size: %d\n' % results['n_dyslexia'])
    f.write('Welch t-test on log(speed): stat=%.6f p=%.8f\n' % (results['ttest_log_speed_stat'], results['ttest_log_speed_pvalue']))
    f.write('Regression (log speed) coef reader_view=%.6f p=%.8f\n' % (results['reg_coef_reader_view'], results['reg_pvalue_reader_view']))
