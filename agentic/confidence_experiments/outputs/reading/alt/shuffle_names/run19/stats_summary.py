import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('reading.csv')

df['reader_view'] = df['language']
df['dyslexia_status'] = df['device']
df['wpm'] = df['num_words'] / (df['running_time'] / 60)

analysis_df = df[(df['running_time'] > 0) & df['num_words'].notna()]
analysis_df = analysis_df.dropna(subset=['reader_view', 'dyslexia_status'])
analysis_df['is_dyslexic'] = analysis_df['dyslexia_status'].isin([1.0, 2.0])
subset = analysis_df[analysis_df['is_dyslexic']]

rv0 = subset[subset['reader_view'] == 0]['wpm']
rv1 = subset[subset['reader_view'] == 1]['wpm']

mean0, mean1 = rv0.mean(), rv1.mean()
std0, std1 = rv0.std(ddof=1), rv1.std(ddof=1)

# paired by participant
paired = subset.groupby(['speed','reader_view'])['wpm'].mean().unstack().dropna()

# paired diff stats
if len(paired) > 1:
    diff = paired[1] - paired[0]
    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1)
    n = len(diff)
    se = sd_diff / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_low, ci_high = mean_diff - t_crit*se, mean_diff + t_crit*se
    tstat, pval = stats.ttest_rel(paired[1], paired[0])
    dz = mean_diff / sd_diff if sd_diff != 0 else np.nan
else:
    mean_diff = sd_diff = n = se = tstat = pval = dz = np.nan
    ci_low = ci_high = np.nan

print('Dyslexic rows:', len(subset))
print('Participants:', subset['speed'].nunique())
print('Reader_view counts:', subset['reader_view'].value_counts().to_dict())
print('Mean WPM reader_view=0:', mean0)
print('Mean WPM reader_view=1:', mean1)
print('Paired mean diff (rv1-rv0):', mean_diff)
print('Paired t-test t:', tstat, 'p:', pval, 'dz:', dz)
print('95% CI diff:', (ci_low, ci_high))
