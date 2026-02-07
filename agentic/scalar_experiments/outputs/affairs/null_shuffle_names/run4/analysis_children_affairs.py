import pandas as pd
import numpy as np


df = pd.read_csv('affairs.csv')

# Map columns based on value patterns (null_shuffle_names).
# 'religiousness' is yes/no -> children indicator.
children = df['religiousness'].map({'yes': 1, 'no': 0})
# 'age' has values {0,1,2,3,7,12} -> affairs frequency categories.
affairs = df['age']

# Basic summaries
summary = df.assign(children=children, affairs=affairs).groupby('children')['affairs'].agg(['count','mean','median'])

# Proportion with any affairs
any_affairs = df.assign(children=children, affairs=affairs).groupby('children')['affairs'].apply(lambda s: (s > 0).mean())

# Difference in means and standardized effect size (Cohen's d)
mean_yes = affairs[children == 1].mean()
mean_no = affairs[children == 0].mean()
std_yes = affairs[children == 1].std(ddof=1)
std_no = affairs[children == 0].std(ddof=1)

n_yes = (children == 1).sum()
n_no = (children == 0).sum()

pooled_sd = np.sqrt(((n_yes-1)*std_yes**2 + (n_no-1)*std_no**2) / (n_yes+n_no-2))
cohen_d = (mean_yes - mean_no) / pooled_sd if pooled_sd > 0 else np.nan

print('Summary by children (0=no,1=yes):')
print(summary)
print('\nProportion with any affairs (>0):')
print(any_affairs)
print('\nMean affairs yes/no:', mean_yes, mean_no)
print('Cohen d (yes-no):', cohen_d)

# Also compute difference in any-affairs proportion
prop_yes = any_affairs.loc[1]
prop_no = any_affairs.loc[0]
print('Diff in any-affairs proportion (yes-no):', prop_yes - prop_no)
