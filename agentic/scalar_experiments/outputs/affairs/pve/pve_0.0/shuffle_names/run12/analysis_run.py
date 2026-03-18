import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data

df = pd.read_csv('affairs.csv')

# Map variables based on metadata misalignment
# children indicator appears in column 'religiousness' (yes/no)
# affairs engagement appears in column 'age'

children = df['religiousness'].map({'yes':1, 'no':0})
affairs = df['age']

# Drop missing just in case
mask = children.notna() & affairs.notna()
children = children[mask]
affairs = affairs[mask]

# Group stats
summary = affairs.groupby(children).agg(['count','mean','std','median'])
summary.index = summary.index.map({0:'no_children',1:'children'})
print('Group summary:\n', summary)

# Welch t-test
no = affairs[children==0]
yes = affairs[children==1]

t_stat, p_val = stats.ttest_ind(yes, no, equal_var=False, nan_policy='omit')
print('\nWelch t-test (children vs no children): t=%.4f p=%.6f' % (t_stat, p_val))

# Mann-Whitney U (two-sided)
# Use alternative='two-sided'
U, p_mw = stats.mannwhitneyu(yes, no, alternative='two-sided')
print('Mann-Whitney U: U=%.1f p=%.6f' % (U, p_mw))

# Effect size Cohen's d (children - no)
mean_diff = yes.mean() - no.mean()
pooled_std = np.sqrt(((yes.std(ddof=1)**2 + no.std(ddof=1)**2) / 2))
cohens_d = mean_diff / pooled_std
print('Mean diff (children - no):', mean_diff)
print('Cohen d:', cohens_d)

# Regression: affairs ~ children
X = sm.add_constant(children)
model = sm.OLS(affairs, X).fit()
print('\nOLS regression affairs ~ children')
print(model.summary())

