import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind, DescrStatsW

# Load data
df = pd.read_csv('reading.csv')

# Use dyslexia_bin when available; fallback to dyslexia > 0
if 'dyslexia_bin' in df.columns:
    dys = df[df['dyslexia_bin'] == 1].copy()
else:
    dys = df[df['dyslexia'] > 0].copy()

# Basic cleaning
use_cols = ['uuid', 'reader_view', 'speed', 'page_id', 'num_words', 'Flesch_Kincaid',
            'device', 'age', 'gender', 'education', 'language', 'retake_trial']
for col in use_cols:
    if col not in dys.columns:
        dys[col] = np.nan

# Drop invalid speeds
dys = dys.replace([np.inf, -np.inf], np.nan)
dys = dys.dropna(subset=['reader_view', 'speed'])
dys = dys[dys['speed'] > 0]

# Summary stats by reader_view
summary = dys.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std']).reset_index()

# Welch t-test on log(speed)
log_speed = np.log(dys['speed'])
rv1 = log_speed[dys['reader_view'] == 1]
rv0 = log_speed[dys['reader_view'] == 0]

t_stat, p_val, dfree = ttest_ind(rv1, rv0, usevar='unequal')

# Within-subject comparison for users with both conditions
pivot = dys.pivot_table(index='uuid', columns='reader_view', values='speed', aggfunc='mean')
paired = pivot.dropna(subset=[0, 1]).copy()
paired['diff_log'] = np.log(paired[1]) - np.log(paired[0])
paired_stats = DescrStatsW(paired['diff_log'])
paired_t, paired_p, paired_df = paired_stats.ttest_mean(0)

# Regression with clustered SEs by participant
# Use log speed for skew reduction
reg_df = dys.dropna(subset=['page_id', 'num_words', 'Flesch_Kincaid', 'device', 'age',
                             'gender', 'education', 'language', 'retake_trial']).copy()
reg_df['log_speed'] = np.log(reg_df['speed'])

formula = (
    'log_speed ~ reader_view + C(page_id) + num_words + Flesch_Kincaid '
    '+ C(device) + age + C(gender) + C(education) + C(language) + retake_trial'
)

model = smf.ols(formula, data=reg_df)
fit = model.fit(cov_type='cluster', cov_kwds={'groups': reg_df['uuid']})
coef = fit.params.get('reader_view', np.nan)
pval = fit.pvalues.get('reader_view', np.nan)

# Effect size in percent for log model
pct_effect = (np.exp(coef) - 1) * 100 if pd.notna(coef) else np.nan

print('Dyslexia subset size:', len(dys))
print('Summary by reader_view (speed):')
print(summary.to_string(index=False))
print('\nWelch t-test on log(speed):')
print(f't={t_stat:.4f}, p={p_val:.4g}, df={dfree:.2f}')

print('\nWithin-subject (paired) log-speed difference for users with both conditions:')
print(f'n_pairs={len(paired)}, t={paired_t:.4f}, p={paired_p:.4g}, df={paired_df:.0f}')

print('\nRegression (log speed) with clustered SEs by uuid:')
print(f'coef(reader_view)={coef:.6f}, p={pval:.4g}, approx_effect={pct_effect:.2f}%')
