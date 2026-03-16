import pandas as pd
import numpy as np
import scipy.stats as stats


df = pd.read_csv('reading.csv')
# reading speed wpm
speed = df['feature7'] / (df['feature5'] / 1000 / 60)

df = df.assign(reading_speed_wpm=speed)

# dyslexic subset
sub = df[df['feature17'] == 1].copy()
sub = sub[np.isfinite(sub['reading_speed_wpm']) & (sub['reading_speed_wpm'] > 0)]

summary = sub.groupby('feature3')['reading_speed_wpm'].agg(['count','mean','median','std'])
print(summary)

pivot = sub.pivot_table(index='feature1', columns='feature3', values='reading_speed_wpm', aggfunc='mean')
paired = pivot.dropna()
print('paired n', len(paired))

if len(paired) >= 2:
    diff = paired[1] - paired[0]
    t_stat, t_p = stats.ttest_rel(paired[1], paired[0])
    w_stat, w_p = stats.wilcoxon(paired[1], paired[0])
    mean_diff = diff.mean()
    median_diff = diff.median()
    sd_diff = diff.std(ddof=1)
    cohen_d = mean_diff / sd_diff if sd_diff != 0 else np.nan
    # 95% CI for mean diff
    ci = stats.t.interval(0.95, len(diff)-1, loc=mean_diff, scale=sd_diff/np.sqrt(len(diff)))
    print('paired t', t_stat, t_p)
    print('wilcoxon', w_stat, w_p)
    print('mean_diff', mean_diff)
    print('median_diff', median_diff)
    print('cohen_d', cohen_d)
    print('ci95', ci)

# unpaired as reference
on = sub[sub['feature3'] == 1]['reading_speed_wpm']
off = sub[sub['feature3'] == 0]['reading_speed_wpm']
print('unpaired t', stats.ttest_ind(on, off, equal_var=False))
print('mannwhitney', stats.mannwhitneyu(on, off, alternative='two-sided'))

