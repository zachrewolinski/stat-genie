import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf

# Load data
DATA_PATH = 'reading.csv'

df = pd.read_csv(DATA_PATH)

# Focus on participants with dyslexia (binary indicator)
# Ensure numeric dtype
if 'dyslexia_bin' in df.columns:
    dys_df = df[df['dyslexia_bin'] == 1].copy()
else:
    # Fallback: treat dyslexia > 0 as dyslexia
    dys_df = df[df['dyslexia'] > 0].copy()

# Clean data: drop missing speed or reader_view
subset = dys_df.dropna(subset=['speed', 'reader_view']).copy()

# Ensure reader_view is binary 0/1
subset = subset[subset['reader_view'].isin([0, 1])]

# Basic stats by reader_view
summary = subset.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std']).reset_index()

# Effect size (Cohen's d)
rv0 = subset[subset['reader_view'] == 0]['speed']
rv1 = subset[subset['reader_view'] == 1]['speed']

# Welch t-test
welch_t = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')

# Mann-Whitney U (non-parametric)
try:
    mw = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
except ValueError:
    mw = None

# Cohen's d (using pooled SD)
mean1, mean0 = rv1.mean(), rv0.mean()
var1, var0 = rv1.var(ddof=1), rv0.var(ddof=1)
pooled_sd = np.sqrt(((len(rv1)-1)*var1 + (len(rv0)-1)*var0) / (len(rv1)+len(rv0)-2)) if (len(rv1)+len(rv0)-2) > 0 else np.nan
cohens_d = (mean1 - mean0) / pooled_sd if pooled_sd and not np.isnan(pooled_sd) else np.nan

# Percentage difference in median
median_diff = rv1.median() - rv0.median()
median_pct = (median_diff / rv0.median()) * 100 if rv0.median() != 0 else np.nan

# Regression with cluster-robust SE (uuid)
# Use log speed to reduce skew
subset = subset.copy()
subset['log_speed'] = np.log(subset['speed'])

# Build formula with limited controls to avoid overfitting
controls = []
for col in ['page_id', 'device', 'english_native']:
    if col in subset.columns:
        controls.append(f'C({col})')
for col in ['age', 'correct_rate', 'retake_trial']:
    if col in subset.columns:
        controls.append(col)

formula = 'log_speed ~ reader_view'
if controls:
    formula += ' + ' + ' + '.join(controls)

# Ensure groups align with rows actually used in the model
model_data_cols = ['uuid', 'log_speed', 'reader_view'] + [c.replace('C(', '').replace(')', '') for c in controls]
model_data = subset[model_data_cols].dropna().copy()

model = smf.ols(formula=formula, data=model_data).fit(
    cov_type='cluster',
    cov_kwds={'groups': model_data['uuid']}
)

# Extract reader_view coefficient
coef = model.params.get('reader_view', np.nan)
pval = model.pvalues.get('reader_view', np.nan)

# Convert log coef to percent change approx
pct_change = (np.exp(coef) - 1) * 100 if not np.isnan(coef) else np.nan

results = {
    'n_total_dyslexia': int(len(subset)),
    'summary_by_reader_view': summary.to_dict(orient='records'),
    'welch_t_stat': float(welch_t.statistic),
    'welch_p_value': float(welch_t.pvalue),
    'mannwhitney_u_stat': float(mw.statistic) if mw else None,
    'mannwhitney_p_value': float(mw.pvalue) if mw else None,
    'cohens_d': float(cohens_d),
    'median_diff': float(median_diff),
    'median_pct_diff': float(median_pct),
    'regression_formula': formula,
    'reg_reader_view_coef_log': float(coef),
    'reg_reader_view_p': float(pval),
    'reg_reader_view_pct_change': float(pct_change)
}

print(json.dumps(results, indent=2))
