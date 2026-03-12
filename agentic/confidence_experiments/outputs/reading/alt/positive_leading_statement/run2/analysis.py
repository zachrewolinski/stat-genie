import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
csv_path = 'reading.csv'

df = pd.read_csv(csv_path)

# Filter dyslexic participants
# dyslexia_bin: 1 indicates dyslexia
if 'dyslexia_bin' in df.columns:
    dys_df = df[df['dyslexia_bin'] == 1].copy()
else:
    dys_df = df[df['dyslexia'].isin([1, 2])].copy()

# Basic cleanup
for col in ['speed', 'reader_view']:
    dys_df = dys_df[pd.notnull(dys_df[col])]

# Ensure numeric
for col in ['speed', 'reader_view']:
    dys_df[col] = pd.to_numeric(dys_df[col], errors='coerce')

dys_df = dys_df.dropna(subset=['speed', 'reader_view'])

# Descriptives by reader_view
summary = dys_df.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std'])

# Welch t-test
rv1 = dys_df[dys_df['reader_view'] == 1]['speed']
rv0 = dys_df[dys_df['reader_view'] == 0]['speed']

welch_t = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')

# Cohen's d for independent samples
mean1, mean0 = rv1.mean(), rv0.mean()
var1, var0 = rv1.var(ddof=1), rv0.var(ddof=1)
# Pooled SD (Welch)
pooled_sd = np.sqrt((var1 + var0) / 2)
cohens_d = (mean1 - mean0) / pooled_sd if pooled_sd > 0 else np.nan

# Participant-level paired analysis if possible
paired_results = {}
if 'uuid' in dys_df.columns:
    # Compute mean speed per participant per condition
    per_uuid = dys_df.groupby(['uuid', 'reader_view'])['speed'].mean().reset_index()
    # Pivot to wide
    wide = per_uuid.pivot(index='uuid', columns='reader_view', values='speed')
    if 0 in wide.columns and 1 in wide.columns:
        paired = wide.dropna(subset=[0, 1])
        if len(paired) > 1:
            diff = paired[1] - paired[0]
            paired_t = stats.ttest_rel(paired[1], paired[0], nan_policy='omit')
            # Cohen's d for paired samples
            d_paired = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
            paired_results = {
                'n_pairs': len(paired),
                'mean_diff': diff.mean(),
                'median_diff': diff.median(),
                't_stat': paired_t.statistic,
                'p_value': paired_t.pvalue,
                'cohens_d': d_paired,
                'mean_rv1': paired[1].mean(),
                'mean_rv0': paired[0].mean(),
            }

# Regression (log speed) with controls
# Prepare data for regression
reg_cols = [
    'speed', 'reader_view', 'page_id', 'num_words', 'device', 'language',
    'age', 'gender', 'education', 'english_native', 'Flesch_Kincaid',
    'correct_rate', 'retake_trial'
]

reg_df = dys_df.copy()
reg_df = reg_df.dropna(subset=[c for c in reg_cols if c in reg_df.columns])

reg_result = None
if len(reg_df) > 0:
    formula = (
        'np.log(speed) ~ reader_view + C(page_id) + num_words + C(device) + C(language) '
        '+ age + C(gender) + C(education) + C(english_native) + Flesch_Kincaid '
        '+ correct_rate + retake_trial'
    )
    try:
        reg_result = smf.ols(formula, data=reg_df).fit(cov_type='HC3')
    except Exception as e:
        reg_result = e

output = {
    'n_dyslexic_rows': int(len(dys_df)),
    'summary_by_reader_view': summary.to_dict(),
    'welch_t': {
        't_stat': float(welch_t.statistic),
        'p_value': float(welch_t.pvalue),
        'mean_rv1': float(mean1),
        'mean_rv0': float(mean0),
        'mean_diff': float(mean1 - mean0),
        'median_rv1': float(rv1.median()),
        'median_rv0': float(rv0.median()),
        'cohens_d': float(cohens_d),
        'n_rv1': int(rv1.shape[0]),
        'n_rv0': int(rv0.shape[0]),
    },
    'paired_analysis': paired_results,
}

if reg_result is not None and not isinstance(reg_result, Exception):
    coef = reg_result.params.get('reader_view', np.nan)
    pval = reg_result.pvalues.get('reader_view', np.nan)
    output['regression'] = {
        'n_used': int(reg_df.shape[0]),
        'coef_reader_view': float(coef),
        'p_value': float(pval),
        'pct_change': float((np.exp(coef) - 1) * 100) if pd.notnull(coef) else np.nan,
        'r_squared': float(reg_result.rsquared),
    }
else:
    output['regression'] = {'error': str(reg_result)}

with open('analysis_output.json', 'w') as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
