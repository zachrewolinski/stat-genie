import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import ttest_ind

# Load data
df = pd.read_csv('reading.csv')

# Basic cleaning: ensure numeric columns
for col in ['reader_view', 'dyslexia_bin', 'speed']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Focus on participants with dyslexia
if 'dyslexia_bin' in df.columns:
    dys = df[df['dyslexia_bin'] == 1].copy()
else:
    # fallback: treat dyslexia 1 or 2 as dyslexia
    dys = df[df['dyslexia'].isin([1, 2])].copy()

# Filter valid speeds
if 'speed' not in dys.columns:
    raise ValueError('speed column missing')

speed = dys['speed']

# Separate reader_view groups
rv1 = dys[dys['reader_view'] == 1]['speed'].dropna()
rv0 = dys[dys['reader_view'] == 0]['speed'].dropna()

# Summary stats
summary = {
    'n_dyslexia_total': len(dys),
    'n_reader_view_1': len(rv1),
    'n_reader_view_0': len(rv0),
    'mean_speed_rv1': rv1.mean(),
    'mean_speed_rv0': rv0.mean(),
    'median_speed_rv1': rv1.median(),
    'median_speed_rv0': rv0.median(),
}

# Welch t-test
if len(rv1) > 1 and len(rv0) > 1:
    t_stat, p_val, dfree = ttest_ind(rv1, rv0, usevar='unequal')
else:
    t_stat, p_val, dfree = np.nan, np.nan, np.nan

summary.update({
    't_stat': t_stat,
    'p_val': p_val,
    'dfree': dfree,
})

# Effect size (Cohen's d using pooled SD)
if len(rv1) > 1 and len(rv0) > 1:
    s1 = rv1.std(ddof=1)
    s0 = rv0.std(ddof=1)
    n1 = len(rv1)
    n0 = len(rv0)
    pooled = np.sqrt(((n1-1)*s1**2 + (n0-1)*s0**2) / (n1+n0-2))
    d = (rv1.mean() - rv0.mean()) / pooled if pooled > 0 else np.nan
else:
    d = np.nan
summary['cohens_d'] = d

# Save a small report for reproducibility
report = pd.DataFrame([summary])
report.to_csv('analysis_summary.csv', index=False)

print(report.to_string(index=False))
