import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "affairs.csv"
df = pd.read_csv(path)

# Define variables
# feature2: frequency of extramarital affairs
# feature6: children yes/no

df = df.copy()

df["children"] = df["feature6"].map({"yes": 1, "no": 0})

df["affairs"] = df["feature2"]

df["affair_any"] = (df["affairs"] > 0).astype(int)

# Descriptive stats
group_stats = df.groupby("children")["affairs"].agg(["count", "mean", "median", "std"])

# Welch t-test
child_yes = df.loc[df["children"] == 1, "affairs"].to_numpy()
child_no = df.loc[df["children"] == 0, "affairs"].to_numpy()

t_stat, t_p = stats.ttest_ind(child_yes, child_no, equal_var=False)

# Mann-Whitney U test
u_stat, u_p = stats.mannwhitneyu(child_yes, child_no, alternative="two-sided")

# Effect size: difference in means (yes - no)
mean_diff = child_yes.mean() - child_no.mean()

# Logistic regression for any affair with controls
# Controls: gender (feature3), age (feature4), years married (feature5), religiousness (feature7),
# education (feature8), occupation (feature9), marriage rating (feature10)

# Encode gender as binary
# feature3: female/male
# Use C() in formula

formula_logit = "affair_any ~ children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10"

logit_model = smf.logit(formula_logit, data=df).fit(disp=False)

# Use default covariance for logit (robust not consistently available in this environment)
logit_results = logit_model

# OLS on affairs with controls (robust SE)
formula_ols = "affairs ~ children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10"
ols_model = smf.ols(formula_ols, data=df).fit(cov_type="HC3")

# Print results
print("Group stats (children=1 yes, 0 no):\n", group_stats)
print("\nMean difference (yes - no):", mean_diff)
print("Welch t-test: t=%.4f p=%.4g" % (t_stat, t_p))
print("Mann-Whitney U: U=%.4f p=%.4g" % (u_stat, u_p))

# Logit summary for children coefficient
logit_coef = logit_results.params["children"]
logit_se = logit_results.bse["children"]
logit_p = logit_results.pvalues["children"]
logit_or = np.exp(logit_coef)

print("\nLogit (affair_any) children coef=%.4f se=%.4f p=%.4g OR=%.4f" % (logit_coef, logit_se, logit_p, logit_or))

# OLS summary for children coefficient
ols_coef = ols_model.params["children"]
ols_se = ols_model.bse["children"]
ols_p = ols_model.pvalues["children"]
print("OLS (affairs) children coef=%.4f se=%.4f p=%.4g" % (ols_coef, ols_se, ols_p))

# Also compute group difference in affair_any rate
rate_yes = df.loc[df["children"] == 1, "affair_any"].mean()
rate_no = df.loc[df["children"] == 0, "affair_any"].mean()

# Two-proportion z-test
count = np.array([
    df.loc[df["children"] == 1, "affair_any"].sum(),
    df.loc[df["children"] == 0, "affair_any"].sum()
])

nobs = np.array([
    (df["children"] == 1).sum(),
    (df["children"] == 0).sum()
])

stat, pval = sm.stats.proportions_ztest(count, nobs)

print("\nAffair_any rate: children yes=%.4f, no=%.4f" % (rate_yes, rate_no))
print("Two-proportion z-test: z=%.4f p=%.4g" % (stat, pval))
