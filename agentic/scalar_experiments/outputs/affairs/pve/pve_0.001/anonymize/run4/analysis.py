import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('affairs.csv')

# Rename for clarity
cols = {
    'feature1': 'id',
    'feature2': 'affairs_freq',
    'feature3': 'gender',
    'feature4': 'age',
    'feature5': 'years_married',
    'feature6': 'children',
    'feature7': 'religiosity',
    'feature8': 'education',
    'feature9': 'occupation',
    'feature10': 'marriage_rating',
}

df = df.rename(columns=cols)

# Basic groups
children_yes = df[df['children'] == 'yes']
children_no = df[df['children'] == 'no']

summary = {
    'n_yes': len(children_yes),
    'n_no': len(children_no),
    'mean_yes': children_yes['affairs_freq'].mean(),
    'mean_no': children_no['affairs_freq'].mean(),
    'median_yes': children_yes['affairs_freq'].median(),
    'median_no': children_no['affairs_freq'].median(),
    'prop_any_yes': (children_yes['affairs_freq'] > 0).mean(),
    'prop_any_no': (children_no['affairs_freq'] > 0).mean(),
}

# Welch t-test on means
welch = stats.ttest_ind(children_yes['affairs_freq'], children_no['affairs_freq'], equal_var=False)

# Mann-Whitney U (two-sided)
mann = stats.mannwhitneyu(children_yes['affairs_freq'], children_no['affairs_freq'], alternative='two-sided')

# Effect size (Cohen's d) for difference in means
mean_diff = summary['mean_yes'] - summary['mean_no']

# Pooled SD for Cohen's d (using standard formula with unequal n)
ny = summary['n_yes']
no = summary['n_no']
sy = children_yes['affairs_freq'].std(ddof=1)
so = children_no['affairs_freq'].std(ddof=1)
sp = np.sqrt(((ny-1)*sy**2 + (no-1)*so**2) / (ny+no-2))
cohens_d = mean_diff / sp if sp != 0 else np.nan

# Chi-square test on any-affair proportions
cont_table = pd.crosstab(df['children'], df['affairs_freq'] > 0)
chi2, chi2_p, _, _ = stats.chi2_contingency(cont_table)

# Logistic regression for any affair with controls
# Encode children yes/no and gender; use C() for categorical
# Use robust SEs (HC1) to reduce sensitivity to heteroskedasticity

df['any_affair'] = (df['affairs_freq'] > 0).astype(int)

logit_model = smf.logit(
    'any_affair ~ C(children) + C(gender) + age + years_married + religiosity + education + occupation + marriage_rating',
    data=df
).fit(disp=False, cov_type='HC1')

# Extract children effect (children=yes compared to no) from logit
# statsmodels uses C(children)[T.yes]
coef = logit_model.params.get('C(children)[T.yes]')
se = logit_model.bse.get('C(children)[T.yes]')
# Wald z and p
z = coef / se if se is not None else np.nan
p = 2 * (1 - stats.norm.cdf(abs(z))) if se is not None else np.nan
# Odds ratio and 95% CI
or_val = np.exp(coef)
ci_low = np.exp(coef - 1.96*se)
ci_high = np.exp(coef + 1.96*se)

# OLS on affair frequency with controls (as robustness)
ols_model = smf.ols(
    'affairs_freq ~ C(children) + C(gender) + age + years_married + religiosity + education + occupation + marriage_rating',
    data=df
).fit(cov_type='HC1')

ols_coef = ols_model.params.get('C(children)[T.yes]')
ols_p = ols_model.pvalues.get('C(children)[T.yes]')

print('SUMMARY', summary)
print('WELCH', welch)
print('MANN', mann)
print('MEAN_DIFF', mean_diff, 'COHENS_D', cohens_d)
print('CHI2', chi2, 'P', chi2_p)
print('LOGIT_COEF', coef, 'SE', se, 'Z', z, 'P', p, 'OR', or_val, 'CI', (ci_low, ci_high))
print('OLS_COEF', ols_coef, 'P', ols_p)
