import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('affairs.csv')

# Basic cleaning: ensure categorical variables are strings
for col in ['feature3', 'feature6']:
    df[col] = df[col].astype(str)

# Outcome and groups
outcome = 'feature2'
children = 'feature6'

# Group stats
summary = df.groupby(children)[outcome].agg(['count', 'mean', 'median', 'std'])

# Welch t-test on mean outcome
no_vals = df.loc[df[children] == 'no', outcome]
yes_vals = df.loc[df[children] == 'yes', outcome]

ttest = stats.ttest_ind(yes_vals, no_vals, equal_var=False, nan_policy='omit')

# Mann-Whitney U (non-parametric)
try:
    mwu = stats.mannwhitneyu(yes_vals, no_vals, alternative='two-sided')
except ValueError:
    mwu = None

# Binary outcome: any engagement (feature2 > 0)
df['any_affair'] = (df[outcome] > 0).astype(int)

prop_any = df.groupby(children)['any_affair'].agg(['mean', 'count'])

# Chi-square test for independence
contingency = pd.crosstab(df[children], df['any_affair'])
chi2 = stats.chi2_contingency(contingency)

# OLS with controls
ols_formula = (
    f"{outcome} ~ C({children}) + C(feature3) + feature4 + feature5 + "
    "feature7 + feature8 + feature9 + feature10"
)
ols_model = smf.ols(ols_formula, data=df).fit(cov_type='HC3')

# Logistic regression with controls
logit_formula = (
    f"any_affair ~ C({children}) + C(feature3) + feature4 + feature5 + "
    "feature7 + feature8 + feature9 + feature10"
)
logit_model = smf.logit(logit_formula, data=df).fit(disp=False)

# Extract coefficient for children
ols_coef = ols_model.params.get(f"C({children})[T.yes]", np.nan)
ols_p = ols_model.pvalues.get(f"C({children})[T.yes]", np.nan)

logit_coef = logit_model.params.get(f"C({children})[T.yes]", np.nan)
logit_p = logit_model.pvalues.get(f"C({children})[T.yes]", np.nan)

# Odds ratio for logistic
logit_or = np.exp(logit_coef) if np.isfinite(logit_coef) else np.nan

print('Group summary (feature2 by children):')
print(summary)
print('\nWelch t-test:', ttest)
if mwu is not None:
    print('Mann-Whitney U:', mwu)

print('\nProportion any affair by children:')
print(prop_any)
print('\nChi-square test:', chi2)

print('\nOLS children coef (yes vs no):', ols_coef, 'p=', ols_p)
print('Logit children coef (yes vs no):', logit_coef, 'p=', logit_p, 'OR=', logit_or)

# Save key stats to csv for inspection if needed
summary.to_csv('summary_children.csv')
prop_any.to_csv('prop_any_children.csv')
