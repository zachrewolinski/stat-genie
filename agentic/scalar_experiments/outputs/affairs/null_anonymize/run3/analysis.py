import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
path = 'affairs.csv'
df = pd.read_csv(path)

# Map variables
# feature2 = affairs frequency (numeric)
# feature6 = children (yes/no)
# feature3 = gender (female/male)

# Basic cleaning

df = df.copy()

# encode
child_map = {'yes': 1, 'no': 0}
gender_map = {'male': 1, 'female': 0}

df['child'] = df['feature6'].map(child_map)
df['male'] = df['feature3'].map(gender_map)

# Outcome
Y = df['feature2']

# Group stats
mean_yes = Y[df['child'] == 1].mean()
mean_no = Y[df['child'] == 0].mean()
med_yes = Y[df['child'] == 1].median()
med_no = Y[df['child'] == 0].median()

# t-test (Welch)
ttest = stats.ttest_ind(Y[df['child']==1], Y[df['child']==0], equal_var=False, nan_policy='omit')

# Mann-Whitney U
mw = stats.mannwhitneyu(Y[df['child']==1], Y[df['child']==0], alternative='two-sided')

# OLS with controls
X = df[['child','male','feature4','feature5','feature7','feature8','feature9','feature10']]
X = sm.add_constant(X)
ols = sm.OLS(Y, X).fit(cov_type='HC3')

# Poisson (for count-ish)
poisson = sm.GLM(Y, X, family=sm.families.Poisson()).fit(cov_type='HC3')

results = {
    'n_total': int(len(df)),
    'n_child_yes': int(df['child'].sum()),
    'n_child_no': int((df['child']==0).sum()),
    'mean_affairs_child_yes': float(mean_yes),
    'mean_affairs_child_no': float(mean_no),
    'median_affairs_child_yes': float(med_yes),
    'median_affairs_child_no': float(med_no),
    'ttest_stat': float(ttest.statistic),
    'ttest_p': float(ttest.pvalue),
    'mw_stat': float(mw.statistic),
    'mw_p': float(mw.pvalue),
    'ols_child_coef': float(ols.params['child']),
    'ols_child_p': float(ols.pvalues['child']),
    'ols_child_ci': [float(c) for c in ols.conf_int().loc['child'].tolist()],
    'poisson_child_coef': float(poisson.params['child']),
    'poisson_child_p': float(poisson.pvalues['child']),
    'poisson_child_ci': [float(c) for c in poisson.conf_int().loc['child'].tolist()],
}

print(json.dumps(results, indent=2))
