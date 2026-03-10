import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

pd.set_option('display.max_columns', None)

df = pd.read_csv('reading.csv')

# Identify key columns based on value patterns
reader_view = df['language']  # 0/1 indicator
# dyslexia status likely in device (0,1,2)
# define dyslexic as device in {1,2}

df = df.copy()
df['reader_view'] = reader_view

df['dyslexia_status'] = df['device']

# reading speed (words per minute) using running_time as seconds
# guard against non-positive

df['wpm'] = df['num_words'] / (df['running_time'] / 60)

# remove rows with missing key info or non-positive times
analysis_df = df[(df['running_time'] > 0) & df['num_words'].notna()]
analysis_df = analysis_df.dropna(subset=['reader_view', 'dyslexia_status'])

# define dyslexic subset
analysis_df['is_dyslexic'] = analysis_df['dyslexia_status'].isin([1.0, 2.0])

# Descriptive stats
subset = analysis_df[analysis_df['is_dyslexic']]

print('Total rows:', len(analysis_df))
print('Dyslexic rows:', len(subset))

print('\nReader view counts in dyslexic subset:')
print(subset['reader_view'].value_counts())

print('\nWPM summary (dyslexic subset):')
print(subset.groupby('reader_view')['wpm'].describe())

# Check if within-subject: number of reader_view levels per participant
# participant id is in 'speed' (uuid-like)
view_counts = subset.groupby('speed')['reader_view'].nunique()
print('\nParticipants with both conditions (dyslexic subset):', (view_counts==2).sum(), 'of', len(view_counts))

# if within-subject, compute per-participant mean wpm by condition and paired test
paired = subset.groupby(['speed','reader_view'])['wpm'].mean().unstack()
paired = paired.dropna()
print('\nPaired sample size:', len(paired))
if len(paired) > 1:
    tstat, pval = stats.ttest_rel(paired[1], paired[0])
    # effect size for paired differences (Cohen's dz)
    diff = paired[1] - paired[0]
    dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) != 0 else np.nan
    print('Paired t-test: t=', tstat, 'p=', pval, 'dz=', dz, 'mean_diff=', diff.mean())

# also do independent t-test (if mostly between-subject) for robustness
rv1 = subset[subset['reader_view'] == 1]['wpm']
rv0 = subset[subset['reader_view'] == 0]['wpm']

# Welch t-test
wt = stats.ttest_ind(rv1, rv0, equal_var=False)
print('\nWelch t-test: t=', wt.statistic, 'p=', wt.pvalue)

# effect size Cohen's d (independent)
mean1, mean0 = rv1.mean(), rv0.mean()
var1, var0 = rv1.var(ddof=1), rv0.var(ddof=1)
# pooled sd for d
n1, n0 = len(rv1), len(rv0)
pooled_sd = np.sqrt(((n1-1)*var1 + (n0-1)*var0) / (n1+n0-2))
d = (mean1 - mean0) / pooled_sd if pooled_sd != 0 else np.nan
print('Means: rv1', mean1, 'rv0', mean0, 'd=', d)

# Mixed effects model with participant random intercept (if possible)
try:
    # standardize wpm for numerical stability
    subset['wpm_z'] = (subset['wpm'] - subset['wpm'].mean()) / subset['wpm'].std(ddof=0)
    md = smf.mixedlm('wpm_z ~ reader_view', subset, groups=subset['speed'])
    mdf = md.fit(reml=False)
    print('\nMixedLM summary:')
    print(mdf.summary())
except Exception as e:
    print('MixedLM failed:', e)

# robust regression controlling for text difficulty and word count
try:
    # include readability (uuid col) and num_words
    model = smf.ols('wpm ~ reader_view + num_words + uuid', data=subset).fit()
    print('\nOLS summary (controls num_words, readability):')
    print(model.summary())
except Exception as e:
    print('OLS failed:', e)
