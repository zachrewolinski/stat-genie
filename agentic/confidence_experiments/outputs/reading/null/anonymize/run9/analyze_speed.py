import pandas as pd
import numpy as np
from scipy import stats

# Load data

df = pd.read_csv('reading.csv')

# Compute reading speed (words per minute) using reading time excluding scrolling (feature5) in ms
# Avoid division by zero
speed_wpm = df['feature7'] * 60000 / df['feature5']
df = df.assign(speed_wpm=speed_wpm)

# Filter dyslexic individuals (feature17 == 1). feature12 indicates severity too, but use feature17 binary
sub = df[df['feature17'] == 1].copy()
print('Total rows dyslexic:', len(sub))
print('Unique participants dyslexic:', sub['feature1'].nunique())
print('Reader view counts:', sub['feature3'].value_counts())

# Remove any infinite or NaN speeds
sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=['speed_wpm'])

# Independent samples comparison
rv_on = sub[sub['feature3'] == 1]['speed_wpm']
rv_off = sub[sub['feature3'] == 0]['speed_wpm']
print('Speed summary on:', rv_on.describe())
print('Speed summary off:', rv_off.describe())

# Welch t-test
welch = stats.ttest_ind(rv_on, rv_off, equal_var=False, nan_policy='omit')
print('Welch t-test:', welch)

# Compute Cohen's d (Hedges g) for independent samples
n1, n0 = rv_on.shape[0], rv_off.shape[0]
mean1, mean0 = rv_on.mean(), rv_off.mean()
var1, var0 = rv_on.var(ddof=1), rv_off.var(ddof=1)
# pooled sd for Hedges g
pooled_sd = np.sqrt(((n1-1)*var1 + (n0-1)*var0) / (n1 + n0 - 2))
cohen_d = (mean1 - mean0) / pooled_sd if pooled_sd > 0 else np.nan
# Hedges g correction
J = 1 - (3 / (4*(n1+n0)-9)) if (n1+n0) > 2 else 1
hedges_g = cohen_d * J
print('Hedges g:', hedges_g)

# Participant-level paired analysis
# Aggregate by participant and condition
agg = sub.groupby(['feature1','feature3'])['speed_wpm'].mean().reset_index()
# pivot to wide
wide = agg.pivot(index='feature1', columns='feature3', values='speed_wpm')
# participants with both conditions
paired = wide.dropna()
print('Participants with both conditions:', paired.shape[0])

if paired.shape[0] > 1:
    paired_diff = paired[1] - paired[0]
    ttest_paired = stats.ttest_rel(paired[1], paired[0])
    print('Paired t-test:', ttest_paired)
    print('Paired mean diff:', paired_diff.mean())
    # Cohen's d for paired samples: mean diff / sd diff
    d_paired = paired_diff.mean() / paired_diff.std(ddof=1) if paired_diff.std(ddof=1) > 0 else np.nan
    print('Paired Cohen d:', d_paired)
else:
    print('Not enough paired participants for paired t-test')

# Non-parametric test (Mann-Whitney) as robustness
mw = stats.mannwhitneyu(rv_on, rv_off, alternative='two-sided')
print('Mann-Whitney U:', mw)

# Effect estimate via linear model with participant fixed effects if both conditions
# We'll do simple within-subject difference at participant level
if paired.shape[0] > 1:
    print('Participant-level mean speed on:', paired[1].mean())
    print('Participant-level mean speed off:', paired[0].mean())

