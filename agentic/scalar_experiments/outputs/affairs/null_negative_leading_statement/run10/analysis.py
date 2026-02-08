import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "affairs.csv"
df = pd.read_csv(path)

# Basic stats by children
summary = df.groupby('children').agg(
    n=('affairs','size'),
    mean_affairs=('affairs','mean'),
    median_affairs=('affairs','median'),
    any_affair=('affairs', lambda x: (x>0).mean())
)
print("Summary by children:")
print(summary)

# Difference in means
means = summary['mean_affairs']
if set(means.index) >= {'yes','no'}:
    diff = means['yes'] - means['no']
    print(f"\nMean affairs difference (yes - no): {diff:.4f}")

# OLS on affairs with controls
# Using categorical for gender, children
ols = smf.ols('affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating', data=df).fit()
print("\nOLS coefficients (affairs):")
print(ols.summary().tables[1])

# Logistic for any affair
# Use GLM binomial

df['any_affair'] = (df['affairs'] > 0).astype(int)
logit = smf.glm('any_affair ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating', data=df, family=sm.families.Binomial()).fit()
print("\nLogit/GLM coefficients (any affair):")
print(logit.summary().tables[1])

# Effect size: odds ratio for children yes (baseline no)
coef_children_yes = logit.params.get('C(children)[T.yes]', np.nan)
if pd.notna(coef_children_yes):
    or_children_yes = np.exp(coef_children_yes)
    print(f"\nOdds ratio for children=yes vs no: {or_children_yes:.3f}")

# T-test for mean affairs difference
from scipy import stats

aff_yes = df.loc[df['children']=='yes','affairs']
aff_no = df.loc[df['children']=='no','affairs']
if len(aff_yes)>1 and len(aff_no)>1:
    tstat, pval = stats.ttest_ind(aff_yes, aff_no, equal_var=False)
    print(f"\nWelch t-test (mean affairs yes vs no): t={tstat:.3f}, p={pval:.4f}")

# Proportion test for any affair
from statsmodels.stats.proportion import proportions_ztest

count = np.array([ (df.loc[df['children']=='yes','any_affair']).sum(), (df.loc[df['children']=='no','any_affair']).sum() ])
obs = np.array([ (df['children']=='yes').sum(), (df['children']=='no').sum() ])
if obs.min()>0:
    stat, pval = proportions_ztest(count, obs)
    print(f"\nProportion z-test (any affair yes vs no): z={stat:.3f}, p={pval:.4f}")
