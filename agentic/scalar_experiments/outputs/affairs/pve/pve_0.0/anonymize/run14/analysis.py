import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv("affairs.csv")

# Basic cleanup
# feature6 is children yes/no
# feature2 is affair frequency

# Create binary any-affair indicator
# Some values in feature2 might be negative? description indicates counts, but data could have noise
# We'll treat >0 as any affairs

# Ensure numeric

df["feature2"] = pd.to_numeric(df["feature2"], errors="coerce")

# Drop rows with missing key fields
key_df = df.dropna(subset=["feature2", "feature6"]).copy()

key_df["any_affair"] = (key_df["feature2"] > 0).astype(int)

# Group stats
summary = key_df.groupby("feature6").agg(
    n=("feature2", "size"),
    mean_affairs=("feature2", "mean"),
    median_affairs=("feature2", "median"),
    prop_any=("any_affair", "mean"),
)

# T-test for mean difference
children_vals = key_df.loc[key_df["feature6"] == "yes", "feature2"].values
no_children_vals = key_df.loc[key_df["feature6"] == "no", "feature2"].values

# Welch t-test
if len(children_vals) > 1 and len(no_children_vals) > 1:
    t_stat, t_p = stats.ttest_ind(children_vals, no_children_vals, equal_var=False, nan_policy="omit")
else:
    t_stat, t_p = np.nan, np.nan

# Mann-Whitney U test for distribution shift
if len(children_vals) > 0 and len(no_children_vals) > 0:
    try:
        u_stat, u_p = stats.mannwhitneyu(children_vals, no_children_vals, alternative="two-sided")
    except ValueError:
        u_stat, u_p = np.nan, np.nan
else:
    u_stat, u_p = np.nan, np.nan

# Logistic regression for any affair vs children
# Control for gender, age, years married, religiousness, education, occupation, marriage rating
# feature3 categorical, feature6 categorical

# Build formula
formula = "any_affair ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10"

logit_res = None
try:
    logit_model = smf.logit(formula, data=key_df)
    logit_res = logit_model.fit(disp=False)
except Exception as e:
    logit_res = e

# OLS for affair frequency with same controls
ols_res = None
try:
    ols_model = smf.ols("feature2 ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10", data=key_df)
    ols_res = ols_model.fit()
except Exception as e:
    ols_res = e

print("Summary by children status:\n", summary)
print("\nWelch t-test: t=", t_stat, "p=", t_p)
print("Mann-Whitney U: U=", u_stat, "p=", u_p)

if isinstance(logit_res, Exception):
    print("Logit failed:", logit_res)
else:
    print("\nLogit summary (children effect):")
    # Coefficient for children yes vs no
    params = logit_res.params
    bse = logit_res.bse
    pvals = logit_res.pvalues
    for term in params.index:
        if "C(feature6)" in term:
            print(term, "coef=", params[term], "se=", bse[term], "p=", pvals[term])

if isinstance(ols_res, Exception):
    print("OLS failed:", ols_res)
else:
    print("\nOLS summary (children effect):")
    params = ols_res.params
    bse = ols_res.bse
    pvals = ols_res.pvalues
    for term in params.index:
        if "C(feature6)" in term:
            print(term, "coef=", params[term], "se=", bse[term], "p=", pvals[term])

