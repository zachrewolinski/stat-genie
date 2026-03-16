import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.formula.api as smf


df = pd.read_csv('reading.csv')

# Compute reading speed (words per minute) using time excluding scrolling (feature5)
# Avoid divide by zero by dropping non-positive times
speed = df['feature7'] / (df['feature5'] / 1000 / 60)

# Add to df
analysis_df = df.copy()
analysis_df['reading_speed_wpm'] = speed

# Focus on dyslexic participants
# feature17 indicates dyslexia (1 yes, 0 no)
analysis_df = analysis_df[analysis_df['feature17'] == 1].copy()

# Drop any non-positive or missing speeds
analysis_df = analysis_df[np.isfinite(analysis_df['reading_speed_wpm'])]
analysis_df = analysis_df[analysis_df['reading_speed_wpm'] > 0]

# Summaries by reader view
summary = analysis_df.groupby('feature3')['reading_speed_wpm'].agg(['count','mean','median','std'])
print('summary by reader view (0=off,1=on):')
print(summary)

# Participant-level paired comparison
pivot = analysis_df.pivot_table(index='feature1', columns='feature3', values='reading_speed_wpm', aggfunc='mean')
# Keep participants with both conditions
paired = pivot.dropna()
print('participants with both conditions', len(paired))

if len(paired) >= 2:
    diff = paired[1] - paired[0]
    t_stat, t_p = stats.ttest_rel(paired[1], paired[0])
    w_stat, w_p = stats.wilcoxon(paired[1], paired[0])
    print('paired t-test t, p', t_stat, t_p)
    print('wilcoxon stat, p', w_stat, w_p)
    print('mean diff', diff.mean(), 'median diff', diff.median())

# Unpaired test as fallback
on = analysis_df[analysis_df['feature3'] == 1]['reading_speed_wpm']
off = analysis_df[analysis_df['feature3'] == 0]['reading_speed_wpm']
print('unpaired t-test', stats.ttest_ind(on, off, equal_var=False))
print('mann-whitney', stats.mannwhitneyu(on, off, alternative='two-sided'))

# Mixed effects model with random intercept for participant
# Use log speed to reduce skew
analysis_df['log_speed'] = np.log(analysis_df['reading_speed_wpm'])

# Encode reader view as binary
analysis_df['reader_view'] = analysis_df['feature3']

# Use mixedlm; drop any infinite values
analysis_df = analysis_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_speed','reader_view','feature1'])

try:
    model = smf.mixedlm('log_speed ~ reader_view', analysis_df, groups=analysis_df['feature1'])
    result = model.fit(reml=False, method='lbfgs')
    print(result.summary())
except Exception as e:
    print('mixedlm failed', e)

