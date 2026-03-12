import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
csv_path = 'reading.csv'
df = pd.read_csv(csv_path)

# Ensure numeric columns
for col in ['reader_view','dyslexia','dyslexia_bin','speed']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Define dyslexia subset: dyslexia_bin == 1 (preferred), else dyslexia > 0
if 'dyslexia_bin' in df.columns:
    dys_df = df[df['dyslexia_bin'] == 1].copy()
    dys_def = 'dyslexia_bin == 1'
elif 'dyslexia' in df.columns:
    dys_df = df[df['dyslexia'] > 0].copy()
    dys_def = 'dyslexia > 0'
else:
    raise ValueError('No dyslexia indicator found')

# Keep rows with speed and reader_view
analysis_df = dys_df[['uuid','reader_view','speed']].dropna()
analysis_df = analysis_df[(analysis_df['reader_view'].isin([0,1]))]

# Basic group stats
summary = analysis_df.groupby('reader_view')['speed'].agg(['count','mean','median','std']).reset_index()

# Effect sizes and tests (independent)
rv0 = analysis_df[analysis_df['reader_view'] == 0]['speed']
rv1 = analysis_df[analysis_df['reader_view'] == 1]['speed']

# Welch t-test
welch_t = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')

# Mann-Whitney U test
try:
    mwu = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
except Exception as e:
    mwu = None

# Cohen's d (Welch)
mean1, mean0 = rv1.mean(), rv0.mean()
var1, var0 = rv1.var(ddof=1), rv0.var(ddof=1)
# pooled SD for unequal n
n1, n0 = rv1.shape[0], rv0.shape[0]
pooled_sd = np.sqrt(((n1-1)*var1 + (n0-1)*var0) / (n1 + n0 - 2)) if (n1+n0-2) > 0 else np.nan
cohens_d = (mean1 - mean0) / pooled_sd if pooled_sd and not np.isnan(pooled_sd) else np.nan

# Paired analysis: per-uuid means for those with both conditions
pivot = analysis_df.pivot_table(index='uuid', columns='reader_view', values='speed', aggfunc='mean')
paired = pivot.dropna()
paired_diff = paired[1] - paired[0]

paired_t = stats.ttest_rel(paired[1], paired[0]) if paired.shape[0] > 1 else None

# Wilcoxon signed-rank for paired
try:
    wilcoxon = stats.wilcoxon(paired_diff) if paired.shape[0] > 0 else None
except Exception:
    wilcoxon = None

# Mixed effects model (random intercept for uuid)
# Only run if enough data
mixed_result = None
try:
    if analysis_df['uuid'].nunique() > 1:
        # Reduce extreme outliers? use log(speed) to stabilize
        analysis_df = analysis_df.copy()
        analysis_df['log_speed'] = np.log1p(analysis_df['speed'])
        model = smf.mixedlm('log_speed ~ reader_view', analysis_df, groups=analysis_df['uuid'])
        mixed_result = model.fit(reml=False, method='lbfgs', maxiter=200, disp=False)
except Exception:
    mixed_result = None

output = {
    'dyslexia_definition': dys_def,
    'n_rows': int(analysis_df.shape[0]),
    'n_participants': int(analysis_df['uuid'].nunique()),
    'group_summary': summary.to_dict(orient='records'),
    'welch_t': {
        'stat': float(welch_t.statistic),
        'p': float(welch_t.pvalue)
    },
    'mwu': None if mwu is None else {'stat': float(mwu.statistic), 'p': float(mwu.pvalue)},
    'cohens_d': float(cohens_d),
    'paired': {
        'n_pairs': int(paired.shape[0]),
        'mean_diff': float(paired_diff.mean()) if paired.shape[0] > 0 else None,
        'median_diff': float(paired_diff.median()) if paired.shape[0] > 0 else None,
        'paired_t': None if paired_t is None else {'stat': float(paired_t.statistic), 'p': float(paired_t.pvalue)},
        'wilcoxon': None if wilcoxon is None else {'stat': float(wilcoxon.statistic), 'p': float(wilcoxon.pvalue)}
    },
    'mixedlm': None
}

if mixed_result is not None:
    output['mixedlm'] = {
        'coef_reader_view': float(mixed_result.params.get('reader_view', np.nan)),
        'p_reader_view': float(mixed_result.pvalues.get('reader_view', np.nan)),
        'nobs': int(mixed_result.nobs)
    }

print(json.dumps(output, indent=2))
