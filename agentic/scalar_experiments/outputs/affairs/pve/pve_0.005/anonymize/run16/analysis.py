import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = 'affairs.csv'

df = pd.read_csv(DATA_PATH)

# rename for clarity
# feature2: affair frequency (numeric)
# feature6: children yes/no

# Ensure children indicator
# handle any capitalization/whitespace
children = df['feature6'].astype(str).str.strip().str.lower()

df = df.copy()

df['children_yes'] = (children == 'yes').astype(int)

affairs = df['feature2']

# Basic group stats
stats_by_group = df.groupby('children_yes')['feature2'].agg(['count','mean','median','std']).to_dict()

# Difference in means t-test (Welch)
arr_yes = df.loc[df['children_yes']==1, 'feature2']
arr_no = df.loc[df['children_yes']==0, 'feature2']

ttest = stats.ttest_ind(arr_yes, arr_no, equal_var=False, nan_policy='omit')

# Nonparametric Mann-Whitney (two-sided)
try:
    mwu = stats.mannwhitneyu(arr_yes, arr_no, alternative='two-sided')
    mwu_stat = mwu.statistic
    mwu_p = mwu.pvalue
except Exception:
    mwu_stat = None
    mwu_p = None

# Linear regression (unadjusted)
ols_unadj = smf.ols('feature2 ~ children_yes', data=df).fit(cov_type='HC3')

# Adjusted regression for common confounders: age, years married, gender, religiousness, education, occupation, marriage rating
# Use feature3 (gender), feature4 (age), feature5 (years married), feature7, feature8, feature9, feature10
# Include categorical gender via C(feature3)
ols_adj = smf.ols('feature2 ~ children_yes + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10', data=df).fit(cov_type='HC3')

result = {
    'group_stats': stats_by_group,
    'ttest_stat': float(ttest.statistic),
    'ttest_p': float(ttest.pvalue),
    'mwu_stat': None if mwu_stat is None else float(mwu_stat),
    'mwu_p': None if mwu_p is None else float(mwu_p),
    'ols_unadj_coef_children': float(ols_unadj.params.get('children_yes', np.nan)),
    'ols_unadj_p_children': float(ols_unadj.pvalues.get('children_yes', np.nan)),
    'ols_adj_coef_children': float(ols_adj.params.get('children_yes', np.nan)),
    'ols_adj_p_children': float(ols_adj.pvalues.get('children_yes', np.nan)),
    'n': int(df.shape[0])
}

print(json.dumps(result, indent=2))
