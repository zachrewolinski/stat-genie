import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('reading.csv')

# correlations with feature20
cols = ['feature4','feature5','feature6','feature7']
for c in cols:
    corr = df[[c,'feature20']].corr().iloc[0,1]
    print('corr', c, 'feature20', corr)

# maybe feature20 = words per minute based on feature4 or 5? compute wpm using feature4/5
for base in ['feature4','feature5']:
    df[f'wpm_{base}'] = df['feature7'] / (df[base]/60000.0)
    corr = df[[f'wpm_{base}','feature20']].corr().iloc[0,1]
    print('corr feature20 vs wpm', base, corr)

# maybe feature20 is ms per word?
for base in ['feature4','feature5']:
    df[f'ms_per_word_{base}'] = df[base] / df['feature7']
    corr = df[[f'ms_per_word_{base}','feature20']].corr().iloc[0,1]
    print('corr feature20 vs ms_per_word', base, corr)

# check monotonic relationships via spearman
for base in ['feature4','feature5']:
    sp = stats.spearmanr(df[f'wpm_{base}'], df['feature20']).correlation
    print('spearman feature20 vs wpm', base, sp)
    sp2 = stats.spearmanr(df[f'ms_per_word_{base}'], df['feature20']).correlation
    print('spearman feature20 vs ms_per_word', base, sp2)

# inspect quantiles of feature20 and ms_per_word
print('feature20 quantiles', df['feature20'].quantile([0.05,0.5,0.95]).to_dict())
print('ms_per_word_feature5 quantiles', df['ms_per_word_feature5'].quantile([0.05,0.5,0.95]).to_dict())
