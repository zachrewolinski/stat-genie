import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('affairs.csv')

# Map children variable
# feature6: yes/no
# Create indicator: 1 if yes, 0 if no
if df['feature6'].dtype.name == 'category':
    children = df['feature6']
else:
    children = df['feature6'].astype(str)

df['children_yes'] = children.str.lower().map({'yes':1, 'no':0})

# Outcome variable
outcome = df['feature2']

# Group stats
summary = df.groupby('children_yes')['feature2'].agg(['count','mean','std','median'])

# Welch t-test
no = df.loc[df['children_yes']==0, 'feature2']
yes = df.loc[df['children_yes']==1, 'feature2']

ttest = stats.ttest_ind(yes, no, equal_var=False, nan_policy='omit')

# Mann-Whitney U test (non-parametric)
try:
    mwu = stats.mannwhitneyu(yes, no, alternative='two-sided')
except Exception as e:
    mwu = None

# Effect size (Cohen's d)
mean_yes = yes.mean()
mean_no = no.mean()
std_yes = yes.std(ddof=1)
std_no = no.std(ddof=1)
# pooled std (Welch)
pooled = np.sqrt(((std_yes**2) + (std_no**2)) / 2)
cohen_d = (mean_yes - mean_no) / pooled if pooled != 0 else np.nan

# OLS regression with controls
# Encode gender (feature3) as category
# Use formula; children_yes as main predictor
formula = 'feature2 ~ children_yes + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10'
model = smf.ols(formula, data=df).fit()

# Extract coefficient and p-value for children_yes
coef = model.params.get('children_yes', np.nan)
pval = model.pvalues.get('children_yes', np.nan)

results = {
    'summary': summary.to_dict(),
    'ttest_stat': ttest.statistic,
    'ttest_pvalue': ttest.pvalue,
    'mwu_stat': None if mwu is None else mwu.statistic,
    'mwu_pvalue': None if mwu is None else mwu.pvalue,
    'mean_yes': mean_yes,
    'mean_no': mean_no,
    'diff_yes_minus_no': mean_yes - mean_no,
    'cohen_d': cohen_d,
    'reg_coef_children_yes': coef,
    'reg_pvalue_children_yes': pval,
    'reg_r2': model.rsquared,
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
