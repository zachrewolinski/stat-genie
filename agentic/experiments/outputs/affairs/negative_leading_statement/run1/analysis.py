import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
DF_PATH = "affairs.csv"
df = pd.read_csv(DF_PATH)

# Basic cleaning
# Ensure children is categorical with yes/no
# Some datasets may have trailing spaces; normalize
if df['children'].dtype == object:
    df['children'] = df['children'].str.strip().str.lower()

# Create binary indicators
# affairs_any: 1 if any extramarital affairs in past year
# children_yes: 1 if has children

df['affairs_any'] = (df['affairs'] > 0).astype(int)

df['children_yes'] = (df['children'] == 'yes').astype(int)

# Descriptive statistics by children
summary = df.groupby('children')[['affairs', 'affairs_any']].agg(['mean', 'median', 'count'])
print("Descriptive stats by children status:\n", summary, "\n")

# Two-sample t-test on affairs counts (Welch)
children_yes = df.loc[df['children_yes'] == 1, 'affairs']
children_no = df.loc[df['children_yes'] == 0, 'affairs']

# If either group is empty, skip tests
if len(children_yes) > 1 and len(children_no) > 1:
    tstat, pval = stats.ttest_ind(children_yes, children_no, equal_var=False)
    print(f"Welch t-test on affairs counts: t={tstat:.3f}, p={pval:.4f}")

    # Mann-Whitney U test (nonparametric)
    ustat, pval_u = stats.mannwhitneyu(children_yes, children_no, alternative='two-sided')
    print(f"Mann-Whitney U test on affairs counts: U={ustat:.1f}, p={pval_u:.4f}\n")

# Logistic regression on any affairs
# Control variables to adjust for confounding
# Use C() for categorical fields
logit_formula = "affairs_any ~ children_yes + C(gender) + age + yearsmarried + religiousness + education + occupation + rating"
logit_model = smf.logit(logit_formula, data=df).fit(disp=False)
print("Logit model summary (affairs_any):")
print(logit_model.summary())
print()

# OLS regression on affairs count (not ideal for counts, but interpretable)
ols_formula = "affairs ~ children_yes + C(gender) + age + yearsmarried + religiousness + education + occupation + rating"
ols_model = smf.ols(ols_formula, data=df).fit()
print("OLS model summary (affairs count):")
print(ols_model.summary())
print()

# Poisson regression for counts
poisson_model = smf.poisson(ols_formula, data=df).fit(disp=False)
print("Poisson model summary (affairs count):")
print(poisson_model.summary())
print()

# Extract key effect sizes
# For logit, report odds ratio and p-value for children_yes
logit_coef = logit_model.params['children_yes']
logit_p = logit_model.pvalues['children_yes']
logit_or = np.exp(logit_coef)

# For OLS and Poisson
ols_coef = ols_model.params['children_yes']
ols_p = ols_model.pvalues['children_yes']

poisson_coef = poisson_model.params['children_yes']
poisson_p = poisson_model.pvalues['children_yes']
poisson_ir = np.exp(poisson_coef)

print("Key effect estimates for children_yes:")
print(f"Logit (any affair): coef={logit_coef:.4f}, OR={logit_or:.3f}, p={logit_p:.4f}")
print(f"OLS (affairs count): coef={ols_coef:.4f}, p={ols_p:.4f}")
print(f"Poisson (affairs count): coef={poisson_coef:.4f}, IRR={poisson_ir:.3f}, p={poisson_p:.4f}")
