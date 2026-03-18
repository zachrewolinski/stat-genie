import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('affairs.csv')

# Rename for clarity

df = df.rename(columns={'feature2': 'affairs', 'feature6': 'children'})

df['children'] = df['children'].astype(str)

# Summary by children
summary = df.groupby('children')['affairs'].agg(['count', 'mean', 'median', 'std'])

# Mann-Whitney U test (nonparametric)

grp_yes = df.loc[df['children'] == 'yes', 'affairs']
grp_no = df.loc[df['children'] == 'no', 'affairs']

u_stat, u_p = stats.mannwhitneyu(grp_yes, grp_no, alternative='two-sided')

# t-test (Welch) for mean difference

t_stat, t_p = stats.ttest_ind(grp_yes, grp_no, equal_var=False)

# Cohen's d

def cohens_d(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    s_pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    return (np.mean(a) - np.mean(b)) / np.sqrt(s_pooled)


d = cohens_d(grp_yes, grp_no)

# Binary outcome: any affairs > 0

df['any_affair'] = (df['affairs'] > 0).astype(int)

# 2x2 table and chi-square

ct = pd.crosstab(df['children'], df['any_affair'])
chi2, chi_p, _, _ = stats.chi2_contingency(ct)

# Logistic regression for odds ratio (children yes vs no)

df['children_yes'] = (df['children'] == 'yes').astype(int)
logit = sm.Logit(df['any_affair'], sm.add_constant(df['children_yes'])).fit(disp=0)

params = logit.params
conf = logit.conf_int()

or_val = np.exp(params['children_yes'])
or_ci = np.exp(conf.loc['children_yes'])

# Linear regression (affairs ~ children) for mean difference

ols = smf.ols('affairs ~ children_yes', data=df).fit()

results = {
    'summary': summary.to_dict(),
    'mannwhitney_u': {'u_stat': float(u_stat), 'p_value': float(u_p)},
    'ttest': {'t_stat': float(t_stat), 'p_value': float(t_p)},
    'cohens_d': float(d),
    'chi2': {'chi2': float(chi2), 'p_value': float(chi_p)},
    'logit': {
        'coef_children_yes': float(params['children_yes']),
        'p_value': float(logit.pvalues['children_yes']),
        'odds_ratio': float(or_val),
        'or_ci_low': float(or_ci[0]),
        'or_ci_high': float(or_ci[1])
    },
    'ols': {
        'coef_children_yes': float(ols.params['children_yes']),
        'p_value': float(ols.pvalues['children_yes'])
    }
}

pd.set_option('display.max_columns', None)
print('SUMMARY')
print(summary)
print('\nRESULTS')
for k, v in results.items():
    if k == 'summary':
        continue
    print(k, v)
