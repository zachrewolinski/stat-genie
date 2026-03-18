import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('affairs.csv')

# Basic checks
print('columns', df.columns.tolist())

# Rename for convenience
# feature2: affairs frequency, feature6: children yes/no

# Ensure types
# children indicator
children = df['feature6'].astype(str).str.lower()

# Some rows might have capitalization; normalize to yes/no

# any affair indicator
any_affair = df['feature2'] > 0

# group stats
summary = df.assign(children=children, any_affair=any_affair).groupby('children').agg(
    n=('feature2', 'size'),
    mean_affairs=('feature2', 'mean'),
    median_affairs=('feature2', 'median'),
    prop_any=('any_affair', 'mean')
)
print('\nGroup summary:\n', summary)

# t-test (Welch)
vals_yes = df.loc[children == 'yes', 'feature2']
vals_no = df.loc[children == 'no', 'feature2']

# Some groups might be empty; guard
if len(vals_yes) > 1 and len(vals_no) > 1:
    tstat, pval = stats.ttest_ind(vals_yes, vals_no, equal_var=False)
    print('\nWelch t-test: t=', tstat, 'p=', pval)

    # Mann-Whitney U
    ustat, upval = stats.mannwhitneyu(vals_yes, vals_no, alternative='two-sided')
    print('Mann-Whitney U: U=', ustat, 'p=', upval)

    # Cohen's d (using pooled SD)
    mean_diff = vals_yes.mean() - vals_no.mean()
    s1 = vals_yes.var(ddof=1)
    s2 = vals_no.var(ddof=1)
    n1, n2 = len(vals_yes), len(vals_no)
    pooled_sd = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1 + n2 - 2))
    d = mean_diff / pooled_sd if pooled_sd > 0 else np.nan
    print('Cohen d (yes-no):', d)

    # Proportion test (chi-square)
    tab = pd.crosstab(children, any_affair)
    chi2, chi_p, dof, expected = stats.chi2_contingency(tab)
    print('Chi-square any_affair vs children: chi2=', chi2, 'p=', chi_p)

# Logistic regression (binary any_affair)
# Use children plus controls (exclude feature1)

# Prepare dataset
reg_df = df.copy()
reg_df['any_affair'] = any_affair.astype(int)
reg_df['children'] = children

# formula with controls; treat gender and children as categorical
formula = 'any_affair ~ C(children) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10'
try:
    logit_model = smf.logit(formula=formula, data=reg_df).fit(disp=False)
    print('\nLogit model summary (children):')
    print(logit_model.summary().tables[1])

    # Extract odds ratio for children yes vs no
    params = logit_model.params
    conf = logit_model.conf_int()
    if 'C(children)[T.yes]' in params.index:
        coef = params['C(children)[T.yes]']
        or_ = np.exp(coef)
        conf_or = np.exp(conf.loc['C(children)[T.yes]'])
        p = logit_model.pvalues['C(children)[T.yes]']
        print('Odds ratio children yes vs no:', or_, '95% CI', tuple(conf_or), 'p=', p)

except Exception as e:
    print('Logit model failed:', e)
