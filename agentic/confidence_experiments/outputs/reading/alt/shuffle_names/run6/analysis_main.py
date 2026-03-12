import json
import pandas as pd
import numpy as np
from scipy import stats

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Map columns based on inference
participant_col = 'speed'  # uuid
reader_view_col = 'language'  # binary 0/1
reading_speed_col = 'running_time'  # words per minute
# dyslexia severity column
# device column values 0,1,2 (0 = no dyslexia, 1/2 = dyslexia)
dyslexia_col = 'device'

# Filter dyslexic participants (severity > 0)
df_dys = df[df[dyslexia_col] > 0].copy()

# Basic group summaries
summary = df_dys.groupby(reader_view_col)[reading_speed_col].agg(['count', 'mean', 'median', 'std'])

# Welch t-test (independent samples)
rv0 = df_dys[df_dys[reader_view_col] == 0][reading_speed_col]
rv1 = df_dys[df_dys[reader_view_col] == 1][reading_speed_col]

# If either group is empty, handle gracefully
if len(rv0) > 1 and len(rv1) > 1:
    t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
    # Cohen's d (Welch) using pooled SD with unequal n
    s1, s0 = rv1.std(ddof=1), rv0.std(ddof=1)
    n1, n0 = rv1.shape[0], rv0.shape[0]
    # pooled SD (unequal n)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
    d = (rv1.mean() - rv0.mean()) / sp if sp > 0 else np.nan
else:
    t_stat, p_val, d = np.nan, np.nan, np.nan

# Nonparametric test
if len(rv0) > 0 and len(rv1) > 0:
    try:
        u_stat, p_u = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    except ValueError:
        u_stat, p_u = np.nan, np.nan
else:
    u_stat, p_u = np.nan, np.nan

# Within-subject (paired) analysis for participants with both conditions
paired = (
    df_dys
    .groupby([participant_col, reader_view_col])[reading_speed_col]
    .mean()
    .unstack(reader_view_col)
)

paired = paired.dropna(subset=[0, 1])

if paired.shape[0] > 1:
    t_stat_p, p_val_p = stats.ttest_rel(paired[1], paired[0], nan_policy='omit')
    # Cohen's d for paired
    diff = paired[1] - paired[0]
    d_p = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
else:
    t_stat_p, p_val_p, d_p = np.nan, np.nan, np.nan

results = {
    'group_summary': summary.to_dict(),
    'welch_ttest': {'t': t_stat, 'p': p_val, 'cohens_d': d, 'n0': len(rv0), 'n1': len(rv1)},
    'mannwhitney': {'u': u_stat, 'p': p_u},
    'paired_ttest': {'t': t_stat_p, 'p': p_val_p, 'cohens_d': d_p, 'n_pairs': int(paired.shape[0])},
}

print(json.dumps(results, indent=2))
