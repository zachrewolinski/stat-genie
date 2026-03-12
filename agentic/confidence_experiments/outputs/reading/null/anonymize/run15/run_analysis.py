import pandas as pd
import numpy as np
from scipy import stats

# Load data
csv_path = 'reading.csv'
df = pd.read_csv(csv_path)

# Define dyslexia subset
# Use feature17: indicates if reader has dyslexia (1) or not (0)
dys_df = df[df['feature17'] == 1].copy()

# Drop rows with missing needed columns
needed_cols = ['feature3', 'feature5', 'feature4', 'feature7']
dys_df = dys_df.dropna(subset=needed_cols)

# Compute reading speed (words per minute) using time on page minus scrolling
# feature5 is time in ms minus scrolling duration
# feature7 is number of words

dys_df['wpm_reading'] = dys_df['feature7'] / (dys_df['feature5'] / 60000.0)

dys_df['wpm_total'] = dys_df['feature7'] / (dys_df['feature4'] / 60000.0)

# Split by reader view activated (feature3)
rv_on = dys_df[dys_df['feature3'] == 1]
rv_off = dys_df[dys_df['feature3'] == 0]

# Helper stats

def summarize(group, col):
    return {
        'n': group[col].shape[0],
        'mean': group[col].mean(),
        'median': group[col].median(),
        'std': group[col].std()
    }

summary = {
    'wpm_reading': {
        'rv_on': summarize(rv_on, 'wpm_reading'),
        'rv_off': summarize(rv_off, 'wpm_reading')
    },
    'wpm_total': {
        'rv_on': summarize(rv_on, 'wpm_total'),
        'rv_off': summarize(rv_off, 'wpm_total')
    }
}

# Winsorize to reduce extreme outliers for sensitivity

def winsorize_series(s, lower=0.01, upper=0.99):
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)

# Statistical tests

def welch_t(a, b):
    return stats.ttest_ind(a, b, equal_var=False)

# compute effect sizes (Cohen's d)

def cohens_d(a, b):
    na = len(a)
    nb = len(b)
    if na < 2 or nb < 2:
        return np.nan
    sa = a.var(ddof=1)
    sb = b.var(ddof=1)
    sp = ((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2)
    return (a.mean() - b.mean()) / np.sqrt(sp)

results = {}
for col in ['wpm_reading', 'wpm_total']:
    a = rv_on[col]
    b = rv_off[col]
    # raw
    t_raw = welch_t(a, b)
    mw_raw = stats.mannwhitneyu(a, b, alternative='two-sided')
    d_raw = cohens_d(a, b)

    # log transform (add small constant to avoid log of zero)
    log_a = np.log(a + 1e-6)
    log_b = np.log(b + 1e-6)
    t_log = welch_t(log_a, log_b)
    d_log = cohens_d(log_a, log_b)

    # winsorized
    a_w = winsorize_series(a)
    b_w = winsorize_series(b)
    t_w = welch_t(a_w, b_w)
    d_w = cohens_d(a_w, b_w)

    results[col] = {
        't_raw': {'stat': float(t_raw.statistic), 'p': float(t_raw.pvalue)},
        'mw_raw': {'stat': float(mw_raw.statistic), 'p': float(mw_raw.pvalue)},
        'd_raw': float(d_raw),
        't_log': {'stat': float(t_log.statistic), 'p': float(t_log.pvalue)},
        'd_log': float(d_log),
        't_winsor': {'stat': float(t_w.statistic), 'p': float(t_w.pvalue)},
        'd_winsor': float(d_w)
    }

print('Summary:', summary)
print('Results:', results)

# Save for inspection
import json
with open('analysis_results.json', 'w') as f:
    json.dump({'summary': summary, 'results': results}, f, indent=2)
