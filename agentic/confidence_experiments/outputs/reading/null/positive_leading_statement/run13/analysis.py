import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('reading.csv')

# Ensure expected columns exist
required = ['speed', 'reader_view', 'dyslexia_bin']
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Filter to dyslexic participants
# dyslexia_bin: 1 = dyslexia, 0 = no dyslexia
# Exclude missing speed or reader_view
sub = df[(df['dyslexia_bin'] == 1) & df['speed'].notna() & df['reader_view'].notna()].copy()

# Basic group stats
sub['reader_view'] = sub['reader_view'].astype(int)

rv1 = sub[sub['reader_view'] == 1]['speed']
rv0 = sub[sub['reader_view'] == 0]['speed']

stats_out = {
    'n_dyslexic_total': int(sub.shape[0]),
    'n_reader_view_on': int(rv1.shape[0]),
    'n_reader_view_off': int(rv0.shape[0]),
    'mean_speed_on': float(rv1.mean()),
    'mean_speed_off': float(rv0.mean()),
    'median_speed_on': float(rv1.median()),
    'median_speed_off': float(rv0.median()),
    'std_speed_on': float(rv1.std(ddof=1)),
    'std_speed_off': float(rv0.std(ddof=1)),
}

# Welch t-test
if len(rv1) > 1 and len(rv0) > 1:
    t_stat, t_p = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
else:
    t_stat, t_p = np.nan, np.nan

# Mann-Whitney U test (nonparametric)
if len(rv1) > 1 and len(rv0) > 1:
    try:
        u_stat, u_p = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    except Exception:
        u_stat, u_p = np.nan, np.nan
else:
    u_stat, u_p = np.nan, np.nan

# Effect size: Cohen's d (Hedges' g with correction)
# d = (mean1 - mean0) / pooled sd
if len(rv1) > 1 and len(rv0) > 1:
    s1 = rv1.std(ddof=1)
    s0 = rv0.std(ddof=1)
    n1 = len(rv1)
    n0 = len(rv0)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
    d = (rv1.mean() - rv0.mean()) / pooled if pooled > 0 else np.nan
    # Hedges' g correction
    J = 1 - (3 / (4*(n1+n0) - 9))
    g = d * J
else:
    d, g = np.nan, np.nan

# Log-speed model to reduce skew; include page and num_words controls if present
# Use only relevant columns and drop missing
model_results = None
if 'page_id' in sub.columns and 'num_words' in sub.columns:
    model_df = sub[['speed', 'reader_view', 'page_id', 'num_words', 'device', 'language']].copy()
    model_df = model_df.dropna()
    # add small constant before log to avoid issues (speed should be positive)
    model_df['log_speed'] = np.log(model_df['speed'])
    # Build model with categorical controls
    try:
        model = smf.ols('log_speed ~ reader_view + C(page_id) + num_words + C(device) + C(language)', data=model_df)
        model_results = model.fit()
    except Exception:
        model_results = None

output = {
    'group_stats': stats_out,
    't_test': {'t_stat': float(t_stat) if t_stat==t_stat else None, 'p_value': float(t_p) if t_p==t_p else None},
    'mannwhitney': {'u_stat': float(u_stat) if u_stat==u_stat else None, 'p_value': float(u_p) if u_p==u_p else None},
    'effect_size': {'cohens_d': float(d) if d==d else None, 'hedges_g': float(g) if g==g else None},
}

if model_results is not None:
    coef = model_results.params.get('reader_view', np.nan)
    pval = model_results.pvalues.get('reader_view', np.nan)
    output['log_speed_model'] = {
        'n_obs': int(model_results.nobs),
        'coef_reader_view': float(coef) if coef==coef else None,
        'p_value': float(pval) if pval==pval else None,
    }
else:
    output['log_speed_model'] = None

print(json.dumps(output, indent=2))
