import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
df = pd.read_csv("affairs.csv")

# Basic cleaning / encoding
# children: yes/no
# affairs is numeric

# Summary stats by children
summary = df.groupby('children')['affairs'].agg(['count','mean','median']).rename(columns={'count':'n'})

# Proportion with any affairs
summary_any = df.assign(any_affair=df['affairs']>0).groupby('children')['any_affair'].mean().to_frame('prop_any')

# t-test for mean difference
children_yes = df.loc[df['children']=='yes','affairs']
children_no = df.loc[df['children']=='no','affairs']

ttest = stats.ttest_ind(children_yes, children_no, equal_var=False, nan_policy='omit')

# Effect size: difference in means and Cohen's d (unequal n, pooled SD)
mean_diff = children_yes.mean() - children_no.mean()

# pooled SD
n1, n2 = len(children_yes), len(children_no)
var1, var2 = children_yes.var(ddof=1), children_no.var(ddof=1)
pooled_sd = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
cohen_d = mean_diff / pooled_sd if pooled_sd > 0 else np.nan

# OLS regression controlling for other covariates
# Use robust SE
formula = "affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating"
ols = smf.ols(formula, data=df).fit(cov_type='HC3')

# Logistic regression for any affair
logit = smf.logit(
    "any_affair ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating",
    data=df.assign(any_affair=(df['affairs'] > 0).astype(int)),
).fit(disp=False)

# Marginal effect of children on probability (average marginal effect)
try:
    marg = logit.get_margeff(at='overall').summary_frame()
except Exception:
    marg = None

# Output results
print("Summary by children (affairs):")
print(summary)
print("\nProportion any affair:")
print(summary_any)
print("\nMean difference (yes - no):", mean_diff)
print("Cohen's d:", cohen_d)
print("T-test:", ttest)

print("\nOLS coefficients (children):")
# in formula, baseline is children=no, so coefficient for C(children)[T.yes]
print(ols.params.filter(like='C(children)').to_string())
print(ols.bse.filter(like='C(children)').to_string())
print(ols.pvalues.filter(like='C(children)').to_string())

print("\nLogit coefficients (children):")
print(logit.params.filter(like='C(children)').to_string())
print(logit.bse.filter(like='C(children)').to_string())
print(logit.pvalues.filter(like='C(children)').to_string())

if marg is not None:
    print("\nLogit marginal effects (overall):")
    print(marg.loc[marg.index.str.contains('C\(children\)')])
