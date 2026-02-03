import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import ttest_ind

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Focus on participants with dyslexia (binary flag == 1)
df = df[df['dyslexia_bin'] == 1].copy()

# Remove missing or non-positive speeds (if any)
df = df[np.isfinite(df['speed'])]

# Split by reader_view
rv0 = df[df['reader_view'] == 0]['speed']
rv1 = df[df['reader_view'] == 1]['speed']

# Basic stats
summary = pd.DataFrame({
    'reader_view': [0, 1],
    'n': [rv0.shape[0], rv1.shape[0]],
    'mean_speed': [rv0.mean(), rv1.mean()],
    'median_speed': [rv0.median(), rv1.median()],
})

# Welch's t-test on raw speed
raw_t, raw_p, raw_df = ttest_ind(rv1, rv0, usevar='unequal')

# Log-transform to reduce skew; add small epsilon to avoid log(0)
eps = 1e-6
rv0_log = np.log(rv0 + eps)
rv1_log = np.log(rv1 + eps)
log_t, log_p, log_df = ttest_ind(rv1_log, rv0_log, usevar='unequal')

# Effect sizes (difference in means)
raw_diff = rv1.mean() - rv0.mean()
log_diff = rv1_log.mean() - rv0_log.mean()

# Save results
summary.to_csv('analysis_summary.csv', index=False)

with open('analysis_results.txt', 'w') as f:
    f.write('Dyslexia-only sample sizes and speed summary\n')
    f.write(summary.to_string(index=False))
    f.write('\n\n')
    f.write(f'Raw speed Welch t-test: t={raw_t:.3f}, p={raw_p:.6f}, df={raw_df:.1f}\n')
    f.write(f'Raw mean difference (rv1 - rv0): {raw_diff:.3f}\n')
    f.write('\n')
    f.write(f'Log speed Welch t-test: t={log_t:.3f}, p={log_p:.6f}, df={log_df:.1f}\n')
    f.write(f'Log mean difference (rv1 - rv0): {log_diff:.6f}\n')

print(summary)
print('Raw t-test', raw_t, raw_p, raw_df)
print('Log t-test', log_t, log_p, log_df)
