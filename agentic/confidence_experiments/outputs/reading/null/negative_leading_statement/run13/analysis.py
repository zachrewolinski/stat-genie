import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('reading.csv')

# Focus on dyslexia participants
# dyslexia_bin: 1 indicates dyslexia (includes severe)
if 'dyslexia_bin' in df.columns:
    dys = df[df['dyslexia_bin'] == 1].copy()
else:
    dys = df[df['dyslexia'].astype(float) > 0].copy()

# Basic cleaning: drop missing speed or reader_view
cols_needed = ['speed', 'reader_view', 'uuid', 'page_id']
for c in cols_needed:
    if c not in dys.columns:
        raise ValueError(f"Missing column {c}")

dys = dys.dropna(subset=['speed', 'reader_view', 'uuid', 'page_id']).copy()

# Ensure numeric reader_view

dys['reader_view'] = pd.to_numeric(dys['reader_view'], errors='coerce')

# Remove any non-binary values

dys = dys[dys['reader_view'].isin([0, 1])].copy()

# Summary stats

n_total = len(dys)
counts = dys['reader_view'].value_counts().to_dict()

# Group stats

g0 = dys[dys['reader_view'] == 0]['speed']
g1 = dys[dys['reader_view'] == 1]['speed']

mean0, mean1 = g0.mean(), g1.mean()
med0, med1 = g0.median(), g1.median()
std0, std1 = g0.std(ddof=1), g1.std(ddof=1)

# Effect size (Cohen's d, pooled SD)

n0, n1 = len(g0), len(g1)
pooled_sd = np.sqrt(((n0 - 1) * std0**2 + (n1 - 1) * std1**2) / (n0 + n1 - 2)) if (n0 + n1 - 2) > 0 else np.nan
cohens_d = (mean1 - mean0) / pooled_sd if pooled_sd and pooled_sd > 0 else np.nan

# Welch t-test

t_stat, t_p = stats.ttest_ind(g1, g0, equal_var=False, nan_policy='omit')

# Mann-Whitney U (two-sided)
try:
    u_stat, u_p = stats.mannwhitneyu(g1, g0, alternative='two-sided')
except ValueError:
    u_stat, u_p = np.nan, np.nan

# Mixed effects model: speed ~ reader_view + C(page_id)
# Random intercept by uuid

mixed_result = None
mixed_p = None
mixed_coef = None

try:
    model = smf.mixedlm('speed ~ reader_view + C(page_id)', dys, groups=dys['uuid'])
    mixed_result = model.fit(reml=False, method='lbfgs')
    mixed_coef = mixed_result.params.get('reader_view', np.nan)
    mixed_p = mixed_result.pvalues.get('reader_view', np.nan)
except Exception as e:
    mixed_result = None
    mixed_coef = np.nan
    mixed_p = np.nan

# Sensitivity: exclude retake trials if available
sens = {}
if 'retake_trial' in dys.columns:
    dys_no_retake = dys[dys['retake_trial'] == 0].copy()
    g0_nr = dys_no_retake[dys_no_retake['reader_view'] == 0]['speed']
    g1_nr = dys_no_retake[dys_no_retake['reader_view'] == 1]['speed']
    if len(g0_nr) > 1 and len(g1_nr) > 1:
        t_stat_nr, t_p_nr = stats.ttest_ind(g1_nr, g0_nr, equal_var=False, nan_policy='omit')
        mean0_nr, mean1_nr = g0_nr.mean(), g1_nr.mean()
        sens = {
            'n_no_retake': len(dys_no_retake),
            'mean0_nr': mean0_nr,
            'mean1_nr': mean1_nr,
            't_p_nr': t_p_nr,
        }

# Package results

results = {
    'n_total': n_total,
    'counts_reader_view': counts,
    'mean0': mean0,
    'mean1': mean1,
    'median0': med0,
    'median1': med1,
    'std0': std0,
    'std1': std1,
    'cohens_d': cohens_d,
    't_p': t_p,
    'u_p': u_p,
    'mixed_coef': mixed_coef,
    'mixed_p': mixed_p,
    'sensitivity': sens,
}

print(json.dumps(results, indent=2))
