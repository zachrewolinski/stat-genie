import json
import pandas as pd
import numpy as np
from scipy import stats

# Load data
csv_path = 'reading.csv'
df = pd.read_csv(csv_path)

# Filter to dyslexia participants (binary)
# Use dyslexia_bin == 1, drop missing
if 'dyslexia_bin' in df.columns:
    dys = df[df['dyslexia_bin'] == 1].copy()
else:
    # fallback: dyslexia >=1
    dys = df[df['dyslexia'] >= 1].copy()

# Ensure needed columns
for col in ['reader_view', 'speed', 'uuid', 'page_id']:
    if col not in dys.columns:
        raise ValueError(f"Missing {col}")

# Drop missing speed/reader_view
dys = dys.dropna(subset=['reader_view', 'speed'])

# Basic group stats
group_stats = dys.groupby('reader_view')['speed'].agg(['count','mean','median','std'])

# Per-participant average speed per condition (to reduce repeated measures)
subj_means = (
    dys.groupby(['uuid','reader_view'])['speed']
    .mean()
    .reset_index()
)

# Pivot to get paired data
paired = subj_means.pivot(index='uuid', columns='reader_view', values='speed')
# Keep only participants with both conditions
paired = paired.dropna()

paired_diff = None
paired_t = None
paired_w = None
if len(paired) > 1:
    diff = paired[1] - paired[0]
    paired_diff = diff
    # Paired t-test
    paired_t = stats.ttest_rel(paired[1], paired[0])
    # Wilcoxon signed-rank
    try:
        paired_w = stats.wilcoxon(paired[1], paired[0])
    except Exception:
        paired_w = None

# Also run simple independent t-test on all rows
ind_t = stats.ttest_ind(
    dys[dys['reader_view'] == 1]['speed'],
    dys[dys['reader_view'] == 0]['speed'],
    equal_var=False,
    nan_policy='omit'
)

# Effect size for paired (Cohen's dz)
cohen_dz = None
if paired_diff is not None and paired_diff.std(ddof=1) > 0:
    cohen_dz = paired_diff.mean() / paired_diff.std(ddof=1)

# Effect size for independent (Hedges g)
rv1 = dys[dys['reader_view']==1]['speed'].dropna()
rv0 = dys[dys['reader_view']==0]['speed'].dropna()

def hedges_g(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return None
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled = ((nx-1)*vx + (ny-1)*vy) / (nx+ny-2)
    if pooled <= 0:
        return None
    d = (x.mean() - y.mean()) / np.sqrt(pooled)
    # small sample correction
    j = 1 - (3 / (4*(nx+ny) - 9))
    return d * j

g_ind = hedges_g(rv1, rv0)

# Summaries
summary = {
    'n_rows_dyslexia': len(dys),
    'n_participants_dyslexia': dys['uuid'].nunique(),
    'group_stats': group_stats.to_dict(),
    'n_paired_participants': len(paired),
    'paired_t': {
        'statistic': None if paired_t is None else float(paired_t.statistic),
        'pvalue': None if paired_t is None else float(paired_t.pvalue),
        'mean_diff': None if paired_diff is None else float(paired_diff.mean()),
    },
    'paired_wilcoxon': None if paired_w is None else {
        'statistic': float(paired_w.statistic),
        'pvalue': float(paired_w.pvalue)
    },
    'ind_t': {
        'statistic': float(ind_t.statistic),
        'pvalue': float(ind_t.pvalue)
    },
    'effect_sizes': {
        'cohen_dz': None if cohen_dz is None else float(cohen_dz),
        'hedges_g_ind': None if g_ind is None else float(g_ind),
    }
}

with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
