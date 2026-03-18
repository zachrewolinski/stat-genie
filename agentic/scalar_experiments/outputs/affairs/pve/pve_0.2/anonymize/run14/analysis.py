import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# Load data
path = "affairs.csv"
df = pd.read_csv(path)

# Map columns for clarity
# feature2: affairs frequency; feature6: children yes/no; feature3: gender

# Basic group stats
# Ensure children categories
print("children value counts:")
print(df['feature6'].value_counts(dropna=False))

# Outcome summaries
for child_val in ['yes','no']:
    sub = df[df['feature6'] == child_val]
    print("\nchild=", child_val, "n=", len(sub))
    print("mean affairs", sub['feature2'].mean())
    print("median affairs", sub['feature2'].median())
    print("share any affair", (sub['feature2']>0).mean())

# Two-sample tests
# t-test on mean affairs
children_yes = df[df['feature6']=='yes']['feature2']
children_no = df[df['feature6']=='no']['feature2']

# Welch t-test
wt = stats.ttest_ind(children_yes, children_no, equal_var=False, nan_policy='omit')
print("\nWelch t-test mean affairs: stat", wt.statistic, "p", wt.pvalue)

# Mann-Whitney U test
mw = stats.mannwhitneyu(children_yes, children_no, alternative='two-sided')
print("Mann-Whitney U: stat", mw.statistic, "p", mw.pvalue)

# Logistic regression for any affair
# Create binary outcome
_df = df.copy()
_df['any_affair'] = (_df['feature2'] > 0).astype(int)

# Unadjusted logit
model_unadj = smf.logit('any_affair ~ C(feature6)', data=_df).fit(disp=False)
print("\nLogit unadjusted")
print(model_unadj.summary())

# Adjusted logit with covariates
# include gender (feature3), age (feature4), years married (feature5), religiousness (feature7), education (feature8), occupation (feature9), marriage rating (feature10)
model_adj = smf.logit('any_affair ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10', data=_df).fit(disp=False)
print("\nLogit adjusted")
print(model_adj.summary())

# Poisson regression for count-ish outcome
# Use robust SE (HC3) to mitigate overdisp
poisson_unadj = smf.glm('feature2 ~ C(feature6)', data=_df, family=sm.families.Poisson()).fit(cov_type='HC3')
print("\nPoisson unadjusted")
print(poisson_unadj.summary())

poisson_adj = smf.glm('feature2 ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10', data=_df, family=sm.families.Poisson()).fit(cov_type='HC3')
print("\nPoisson adjusted")
print(poisson_adj.summary())

# Compute effect sizes
# difference in means
mean_yes = children_yes.mean()
mean_no = children_no.mean()
print("\nmean_diff (yes - no):", mean_yes - mean_no)

# Cohen's d (using pooled SD)
# Use nan-safe
ny = children_yes.dropna()
no = children_no.dropna()
pooled_sd = np.sqrt(((ny.var(ddof=1))*(len(ny)-1) + (no.var(ddof=1))*(len(no)-1)) / (len(ny)+len(no)-2))
cohen_d = (ny.mean() - no.mean()) / pooled_sd
print("Cohen's d:", cohen_d)
