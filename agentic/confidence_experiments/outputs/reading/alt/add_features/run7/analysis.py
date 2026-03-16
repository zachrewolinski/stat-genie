import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

path = 'reading.csv'

df = pd.read_csv(path)
print(df.head())
print(df.columns)
print(df.shape)

# Focus: reader_view effect on speed for dyslexia individuals
# dyslexia status: dyslexia_bin (1 dyslexia) or dyslexia (1 or 2). We'll use dyslexia_bin==1.

# Clean: ensure speed numeric >0

# Summary by dyslexia_bin and reader_view
for col in ['dyslexia_bin','dyslexia']:
    if col in df.columns:
        print(col, df[col].value_counts(dropna=False).head())

# Remove missing in key vars
key_cols = ['speed','reader_view','dyslexia_bin']

sub = df[key_cols].dropna()
print('sub rows', sub.shape)

# Only dyslexia individuals
sub_dys = sub[sub['dyslexia_bin']==1]
print('dys rows', sub_dys.shape)

# basic stats
stats = sub_dys.groupby('reader_view')['speed'].agg(['count','mean','median','std'])
print(stats)

# t-test
import scipy.stats as st
rv0 = sub_dys[sub_dys['reader_view']==0]['speed']
rv1 = sub_dys[sub_dys['reader_view']==1]['speed']
print('means', rv0.mean(), rv1.mean())
print('medians', rv0.median(), rv1.median())

# Welch t-test
if len(rv0)>1 and len(rv1)>1:
    tstat, pval = st.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
    print('welch t-test', tstat, pval)

# Mann-Whitney
if len(rv0)>0 and len(rv1)>0:
    ustat, pval_u = st.mannwhitneyu(rv1, rv0, alternative='two-sided')
    print('mannwhitney', ustat, pval_u)

# Regression with controls? maybe adjust for page_id, num_words, language? We'll do log(speed) maybe, because skewed.

# add log speed
sub2 = df[['speed','reader_view','dyslexia_bin','num_words','page_id','language','device','age','gender','education','retake_trial','correct_rate','Flesch_Kincaid']].copy()
sub2 = sub2.dropna(subset=['speed','reader_view','dyslexia_bin'])

# restrict dyslexia
sub2 = sub2[sub2['dyslexia_bin']==1]

# avoid non-positive speed
sub2 = sub2[sub2['speed']>0]
sub2['log_speed'] = np.log(sub2['speed'])

# simple regression: log_speed ~ reader_view
model1 = smf.ols('log_speed ~ reader_view', data=sub2).fit()
print(model1.summary())

# Add controls: page_id, num_words, language, device, age, gender, education, retake_trial, correct_rate, Flesch_Kincaid
# convert categorical via C()
formula = 'log_speed ~ reader_view + C(page_id) + num_words + C(language) + C(device) + age + C(gender) + C(education) + retake_trial + correct_rate + Flesch_Kincaid'
model2 = smf.ols(formula, data=sub2).fit()
print(model2.summary())

# Also consider mixed? But fine.

# Effect size (Cohen d)
if len(rv0)>1 and len(rv1)>1:
    n0, n1 = len(rv0), len(rv1)
    s0, s1 = rv0.std(ddof=1), rv1.std(ddof=1)
    pooled = np.sqrt(((n0-1)*s0**2 + (n1-1)*s1**2)/(n0+n1-2))
    d = (rv1.mean() - rv0.mean())/pooled
    print('cohen d', d)

