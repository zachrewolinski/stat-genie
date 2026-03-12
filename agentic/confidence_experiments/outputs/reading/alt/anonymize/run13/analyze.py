import pandas as pd
import numpy as np
from scipy import stats

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Identify key columns
id_col = 'feature1'
page_col = 'feature2'
reader_view_col = 'feature3'
read_time_col = 'feature5'  # time minus scrolling (ms)
words_col = 'feature7'
reading_speed_col = 'feature20'
dyslexia_col = 'feature17'

# Clean numeric columns
for c in [reader_view_col, read_time_col, words_col, reading_speed_col, dyslexia_col]:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Basic derived reading speed (words per minute) using reading time minus scrolling
# Avoid divide-by-zero
df['derived_wpm'] = df[words_col] / (df[read_time_col] / 60000.0)

# Correlation between feature20 and derived speed to see if feature20 is reading speed
valid = df[[reading_speed_col, 'derived_wpm']].dropna()
cor = valid[reading_speed_col].corr(valid['derived_wpm'])

# Dyslexia subset
dys = df[df[dyslexia_col] == 1].copy()

# Choose speed metric: if feature20 strongly correlates with derived, use feature20; else derived
use_col = reading_speed_col if cor > 0.7 else 'derived_wpm'

# Exclude non-positive or extreme derived? We'll do basic filtering for unrealistic values
# Keep within 1st-99th percentile to reduce outlier impact
speed = dys[use_col].dropna()
low, high = speed.quantile([0.01, 0.99])

dys = dys[(dys[use_col] >= low) & (dys[use_col] <= high)].copy()

# Aggregate per participant per condition
agg = (
    dys.groupby([id_col, reader_view_col])[use_col]
    .mean()
    .reset_index()
)

# Pivot to wide for paired analysis
wide = agg.pivot(index=id_col, columns=reader_view_col, values=use_col)

# Determine paired participants (both conditions)
paired = wide.dropna()

results = {}

# Paired t-test if enough paired data
if len(paired) >= 10:
    tstat, pval = stats.ttest_rel(paired[1], paired[0])
    diff = paired[1] - paired[0]
    mean_diff = diff.mean()
    # Cohen's d for paired = mean_diff / sd_diff
    d = mean_diff / diff.std(ddof=1)
    results['paired_n'] = len(paired)
    results['paired_t'] = tstat
    results['paired_p'] = pval
    results['paired_mean_diff'] = mean_diff
    results['paired_d'] = d

# Unpaired comparison (all data, per-participant mean if not paired)
# Use per participant mean for each condition; then compare independent samples
rv1 = agg[agg[reader_view_col] == 1][use_col]
rv0 = agg[agg[reader_view_col] == 0][use_col]

if len(rv1) >= 10 and len(rv0) >= 10:
    tstat_u, pval_u = stats.ttest_ind(rv1, rv0, equal_var=False)
    # Cohen's d for independent samples (pooled SD)
    n1, n0 = len(rv1), len(rv0)
    s1, s0 = rv1.std(ddof=1), rv0.std(ddof=1)
    sp = np.sqrt(((n1-1)*s1**2 + (n0-1)*s0**2) / (n1+n0-2))
    d_u = (rv1.mean() - rv0.mean()) / sp
    results['unpaired_n1'] = n1
    results['unpaired_n0'] = n0
    results['unpaired_t'] = tstat_u
    results['unpaired_p'] = pval_u
    results['unpaired_mean_diff'] = rv1.mean() - rv0.mean()
    results['unpaired_d'] = d_u

# Descriptives
results['use_col'] = use_col
results['cor_feature20_vs_derived'] = cor
results['dys_rows'] = len(dys)
results['rv1_mean'] = rv1.mean()
results['rv0_mean'] = rv0.mean()
results['rv1_median'] = rv1.median()
results['rv0_median'] = rv0.median()
results['rv1_sd'] = rv1.std(ddof=1)
results['rv0_sd'] = rv0.std(ddof=1)

# Nonparametric test (Wilcoxon) for paired if available
if len(paired) >= 10:
    try:
        wstat, wp = stats.wilcoxon(paired[1], paired[0])
        results['paired_wilcoxon_p'] = wp
    except Exception as e:
        results['paired_wilcoxon_error'] = str(e)

# Nonparametric test (Mann-Whitney) for unpaired
if len(rv1) >= 10 and len(rv0) >= 10:
    ustat, up = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    results['unpaired_mannwhitney_p'] = up

# Save results
import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
