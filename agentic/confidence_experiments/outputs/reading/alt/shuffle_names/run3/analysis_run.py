import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('reading.csv')

# Map columns based on data inspection
participant_col = 'speed'          # unique participant id
reader_view_col = 'language'       # 0/1 indicator for reader view
speed_col = 'running_time'         # reading speed (wpm) inferred from data
# dyslexia status inferred from device (0=none, 1=dyslexia, 2=severe)
dyslexia_col = 'device'

# define dyslexia indicator
# treat dyslexic as device > 0

df = df.copy()
df['dyslexic'] = df[dyslexia_col].apply(lambda x: 1 if pd.notna(x) and x > 0 else (0 if pd.notna(x) else np.nan))

# filter dyslexic rows

dys_df = df[df['dyslexic'] == 1]

# Ensure reader_view is binary 0/1
# drop missing

dys_df = dys_df.dropna(subset=[reader_view_col, speed_col, participant_col])

# Compute within-subject means per condition
pivot = dys_df.pivot_table(index=participant_col, columns=reader_view_col, values=speed_col, aggfunc='mean')

# participants with both conditions
paired = pivot.dropna()

# Extract arrays
rv0 = paired.get(0)
rv1 = paired.get(1)

# Paired t-test
if rv0 is not None and rv1 is not None and len(paired) > 1:
    t_stat, p_val = stats.ttest_rel(rv1, rv0, nan_policy='omit')
else:
    t_stat, p_val = np.nan, np.nan

# Wilcoxon signed-rank test (if enough pairs and not all equal)
wilcoxon_p = np.nan
if rv0 is not None and rv1 is not None and len(paired) > 0:
    diff = rv1 - rv0
    if (diff != 0).any() and len(diff) >= 10:
        try:
            _, wilcoxon_p = stats.wilcoxon(diff)
        except Exception:
            wilcoxon_p = np.nan

# Effect size: paired Cohen's d (mean diff / std diff)
if rv0 is not None and rv1 is not None and len(paired) > 1:
    diff = rv1 - rv0
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) != 0 else np.nan
else:
    d = np.nan

# Also compute row-level (unpaired) comparison for context
rv0_all = dys_df[dys_df[reader_view_col] == 0][speed_col]
rv1_all = dys_df[dys_df[reader_view_col] == 1][speed_col]

if len(rv0_all) > 1 and len(rv1_all) > 1:
    t_stat_ind, p_val_ind = stats.ttest_ind(rv1_all, rv0_all, equal_var=False, nan_policy='omit')
else:
    t_stat_ind, p_val_ind = np.nan, np.nan

# Descriptive stats

def desc(series):
    return {
        'n': int(series.count()),
        'mean': float(series.mean()),
        'median': float(series.median()),
        'std': float(series.std(ddof=1))
    }

summary = {
    'n_dyslexic_rows': int(dys_df.shape[0]),
    'n_dyslexic_participants': int(dys_df[participant_col].nunique()),
    'paired_participants': int(len(paired)),
    'reader_view_0': desc(rv0_all),
    'reader_view_1': desc(rv1_all),
    'paired_means_rv0': float(rv0.mean()) if rv0 is not None and len(paired) else np.nan,
    'paired_means_rv1': float(rv1.mean()) if rv1 is not None and len(paired) else np.nan,
    'mean_diff_paired': float((rv1 - rv0).mean()) if rv0 is not None and rv1 is not None and len(paired) else np.nan,
    't_test_paired_p': float(p_val) if p_val is not None else np.nan,
    'wilcoxon_p': float(wilcoxon_p) if wilcoxon_p is not None else np.nan,
    'cohens_d_paired': float(d) if d is not None else np.nan,
    't_test_ind_p': float(p_val_ind) if p_val_ind is not None else np.nan,
}

print(summary)
