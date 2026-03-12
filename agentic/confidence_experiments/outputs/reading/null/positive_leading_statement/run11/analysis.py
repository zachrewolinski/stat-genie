import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf

DATA = Path('reading.csv')

df = pd.read_csv(DATA)

# basic prep
# ensure speed positive

df = df.copy()

# dyslexia participants: dyslexia_bin == 1 OR dyslexia>0
if 'dyslexia_bin' in df.columns:
    dys_df = df[df['dyslexia_bin'] == 1].copy()
else:
    dys_df = df[df['dyslexia'] > 0].copy()

# remove missing or non-positive speeds
speed_col = 'speed'

dys_df = dys_df.replace([np.inf, -np.inf], np.nan)

dys_df = dys_df.dropna(subset=[speed_col, 'reader_view', 'uuid'])

dys_df = dys_df[dys_df[speed_col] > 0]

# log speed for modeling

dys_df['log_speed'] = np.log(dys_df[speed_col])

# check per participant condition counts
cond_counts = dys_df.groupby(['uuid','reader_view']).size().unstack(fill_value=0)

# participants with both conditions
both_mask = (cond_counts.get(0,0) > 0) & (cond_counts.get(1,0) > 0)

participants_both = cond_counts[both_mask].index

paired_df = dys_df[dys_df['uuid'].isin(participants_both)].copy()

# aggregate per participant per condition
agg = paired_df.groupby(['uuid','reader_view'])['speed'].mean().reset_index()
# pivot for paired test
pivot = agg.pivot(index='uuid', columns='reader_view', values='speed').dropna()

# paired t-test on log speed
pivot_log = np.log(pivot)

paired_t = stats.ttest_rel(pivot_log[1], pivot_log[0]) if set(pivot_log.columns)=={0,1} else None

# effect size (Cohen's d for paired: mean diff / sd diff)

diff = (pivot_log[1] - pivot_log[0])
cohen_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan

# nonparametric Wilcoxon on paired log speeds
try:
    wilcoxon_res = stats.wilcoxon(pivot_log[1], pivot_log[0])
except Exception as e:
    wilcoxon_res = e

# bootstrap CI for mean log-diff
rng = np.random.default_rng(42)
boot_means = []
if len(diff) > 1:
    for _ in range(2000):
        sample = rng.choice(diff, size=len(diff), replace=True)
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
else:
    ci_low, ci_high = np.nan, np.nan

# mixed-effects model with random intercepts for uuid and page_id (if possible)

# statsmodels MixedLM doesn't support random effects for multiple groups easily with formula.
# We'll include page_id as fixed effect and random intercept for uuid.

model = None
model_result = None
try:
    # include page_id fixed effect, plus num_words if present
    formula = 'log_speed ~ reader_view'
    if 'num_words' in dys_df.columns:
        formula += ' + num_words'
    if 'page_id' in dys_df.columns:
        formula += ' + C(page_id)'
    if 'device' in dys_df.columns:
        formula += ' + C(device)'
    # run mixedlm
    model = smf.mixedlm(formula, dys_df, groups=dys_df['uuid'])
    model_result = model.fit(reml=False, method='lbfgs', maxiter=200)
except Exception as e:
    model_result = e

# compute raw means
mean_speed = dys_df.groupby('reader_view')['speed'].mean()
median_speed = dys_df.groupby('reader_view')['speed'].median()

n_total = len(dys_df)

print('dyslexia_rows', n_total)
print('participants_both', len(participants_both))
print('paired_n', len(pivot))
print('mean_speed', mean_speed.to_dict())
print('median_speed', median_speed.to_dict())

if paired_t:
    print('paired_t_stat', paired_t.statistic)
    print('paired_p', paired_t.pvalue)
    print('paired_cohen_d', cohen_d)
    print('mean_log_diff', diff.mean())
    print('boot_ci_log_diff', (ci_low, ci_high))
    if hasattr(wilcoxon_res, 'statistic'):
        print('wilcoxon_stat', wilcoxon_res.statistic)
        print('wilcoxon_p', wilcoxon_res.pvalue)
    else:
        print('wilcoxon_failed', repr(wilcoxon_res))

print('mixedlm_summary')
if hasattr(model_result, 'summary'):
    print(model_result.summary())
else:
    print('mixedlm_failed', repr(model_result))
