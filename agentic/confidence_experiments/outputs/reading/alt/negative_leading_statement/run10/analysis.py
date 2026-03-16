import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('reading.csv')

# Clean/ensure numeric
for col in ['reader_view','speed','dyslexia','dyslexia_bin','retake_trial','num_words','Flesch_Kincaid','age']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter dyslexic individuals: dyslexia_bin == 1 (or dyslexia >0)
# We'll use dyslexia_bin if available else dyslexia >0
if 'dyslexia_bin' in df.columns:
    dys = df[df['dyslexia_bin'] == 1].copy()
else:
    dys = df[df['dyslexia'] > 0].copy()

print('Total rows:', len(df))
print('Dyslexia rows:', len(dys))
print('Unique dyslexic participants:', dys['uuid'].nunique())

# Basic group stats
summary = dys.groupby('reader_view')['speed'].agg(['count','mean','median','std']).reset_index()
print('\nSpeed summary by reader_view (dyslexic):')
print(summary)

# Effect size: Cohen's d (independent)
rv0 = dys[dys['reader_view'] == 0]['speed'].dropna()
rv1 = dys[dys['reader_view'] == 1]['speed'].dropna()

# Two-sample t-test (Welch)
if len(rv0) > 1 and len(rv1) > 1:
    tstat, pval = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
    # Cohen's d for unequal sizes
    n1, n0 = len(rv1), len(rv0)
    s1, s0 = rv1.std(ddof=1), rv0.std(ddof=1)
    # pooled SD
    sp = np.sqrt(((n1-1)*s1**2 + (n0-1)*s0**2) / (n1+n0-2))
    d = (rv1.mean() - rv0.mean()) / sp if sp > 0 else np.nan
    print('\nWelch t-test reader_view=1 vs 0 (dyslexic):')
    print('t =', tstat, 'p =', pval, 'cohen d =', d)
else:
    print('\nNot enough data for t-test')

# Check within-subject availability (do participants have both conditions?)
counts_by_uuid = dys.groupby('uuid')['reader_view'].nunique()
print('\nParticipants with both reader_view conditions:', (counts_by_uuid >= 2).sum())

# Participant-level paired analysis if possible
paired_uuids = counts_by_uuid[counts_by_uuid >= 2].index
paired = dys[dys['uuid'].isin(paired_uuids)]
if len(paired) > 0:
    # average speed per participant per condition
    paired_means = paired.groupby(['uuid','reader_view'])['speed'].mean().unstack()
    paired_means = paired_means.dropna()
    if paired_means.shape[0] > 1:
        tstat_p, pval_p = stats.ttest_rel(paired_means[1], paired_means[0], nan_policy='omit')
        diff = paired_means[1] - paired_means[0]
        d_p = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
        print('\nPaired t-test on participant means (reader_view 1 vs 0):')
        print('n_pairs =', paired_means.shape[0], 't =', tstat_p, 'p =', pval_p, 'd =', d_p)
    else:
        print('\nNot enough paired participants for paired t-test')
else:
    print('\nNo paired participants')

# Mixed effects model: speed ~ reader_view + page_id + num_words + Flesch_Kincaid + device + age + gender + retake_trial
# Random intercept for uuid
# Drop rows with missing essential variables
model_cols = ['speed','reader_view','page_id','num_words','Flesch_Kincaid','device','age','gender','retake_trial','uuid']
model_df = dys[model_cols].dropna()
print('\nModel rows:', len(model_df))

if len(model_df) > 20:
    # MixedLM with random intercept for uuid
    # Use categorical for page_id, device, gender, retake_trial
    try:
        md = smf.mixedlm('speed ~ reader_view + C(page_id) + num_words + Flesch_Kincaid + C(device) + age + C(gender) + C(retake_trial)',
                         model_df, groups=model_df['uuid'])
        mdf = md.fit(reml=False)
        print('\nMixedLM results:')
        print(mdf.summary())
    except Exception as e:
        print('\nMixedLM failed:', e)
        # fallback OLS with cluster-robust SE
        try:
            ols = smf.ols('speed ~ reader_view + C(page_id) + num_words + Flesch_Kincaid + C(device) + age + C(gender) + C(retake_trial)',
                          data=model_df).fit(cov_type='cluster', cov_kwds={'groups': model_df['uuid']})
            print('\nOLS cluster-robust results:')
            print(ols.summary())
        except Exception as e2:
            print('OLS failed:', e2)
