import json
import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv('reading.csv')

# Identify columns
# feature3: reader view (1/0)
# feature17: dyslexia (1/0)
# feature12: dyslexia severity (0/1/2)
# feature20: reading speed (wpm)

# Focus on dyslexic individuals (feature17==1)
dys = df[df['feature17'] == 1].copy()

# Basic counts
total_rows = len(dys)
n_participants = dys['feature1'].nunique()

# Participant-level mean reading speed by reader view
pivot = dys.pivot_table(index='feature1', columns='feature3', values='feature20', aggfunc='mean')
# Keep participants with both conditions
paired = pivot.dropna(subset=[0, 1])
n_paired = len(paired)

# Paired t-test
if n_paired > 1:
    t_stat, p_val = stats.ttest_rel(paired[1], paired[0])
    diff = paired[1] - paired[0]
    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1)
    # Cohen's d for paired samples
    cohens_d = mean_diff / sd_diff if sd_diff != 0 else np.nan
else:
    t_stat = p_val = mean_diff = sd_diff = cohens_d = np.nan

# Also compute overall (non-paired) difference for context
mean_on = dys[dys['feature3'] == 1]['feature20'].mean()
mean_off = dys[dys['feature3'] == 0]['feature20'].mean()
n_on = (dys['feature3'] == 1).sum()
n_off = (dys['feature3'] == 0).sum()

# Effect size (Cohen's d for independent samples)
if n_on > 1 and n_off > 1:
    pooled_sd = np.sqrt(
        (
            (n_on - 1) * dys[dys['feature3'] == 1]['feature20'].var(ddof=1)
            + (n_off - 1) * dys[dys['feature3'] == 0]['feature20'].var(ddof=1)
        )
        / (n_on + n_off - 2)
    )
    cohens_d_ind = (mean_on - mean_off) / pooled_sd if pooled_sd != 0 else np.nan
else:
    cohens_d_ind = np.nan

# Summaries for reader view effect in dyslexic participants
summary = {
    'total_rows_dys': int(total_rows),
    'n_participants_dys': int(n_participants),
    'n_rows_reader_view_on': int(n_on),
    'n_rows_reader_view_off': int(n_off),
    'paired_participants': int(n_paired),
    'mean_speed_on': float(mean_on),
    'mean_speed_off': float(mean_off),
    'paired_mean_diff': float(mean_diff) if not np.isnan(mean_diff) else None,
    'paired_t_stat': float(t_stat) if not np.isnan(t_stat) else None,
    'paired_p_value': float(p_val) if not np.isnan(p_val) else None,
    'paired_cohens_d': float(cohens_d) if not np.isnan(cohens_d) else None,
    'independent_cohens_d': float(cohens_d_ind) if not np.isnan(cohens_d_ind) else None,
}

print(json.dumps(summary, indent=2))
