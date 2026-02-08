import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv("affairs.csv")

# Basic cleanup
# Children column: expect 'yes'/'no'
if df["children"].dtype != object:
    df["children"] = df["children"].astype(str)

# Create indicators
df["children_yes"] = (df["children"].str.lower() == "yes").astype(int)
df["affair_any"] = (df["affairs"] > 0).astype(int)

# Summary stats
summary = df.groupby("children_yes").agg(
    n=("affairs", "size"),
    mean_affairs=("affairs", "mean"),
    mean_affair_any=("affair_any", "mean"),
)

# Effect sizes
mean_diff_affairs = summary.loc[1, "mean_affairs"] - summary.loc[0, "mean_affairs"]
mean_diff_any = summary.loc[1, "mean_affair_any"] - summary.loc[0, "mean_affair_any"]

# Logistic regression for affair_any
# Controls: age, yearsmarried, religiousness, education, occupation, rating, gender
# Encode gender: female=1, male=0
df["gender_female"] = (df["gender"].str.lower() == "female").astype(int)

X_cols = [
    "children_yes",
    "age",
    "yearsmarried",
    "religiousness",
    "education",
    "occupation",
    "rating",
    "gender_female",
]
X = df[X_cols]
X = sm.add_constant(X)
logit_model = sm.Logit(df["affair_any"], X).fit(disp=0)

# Linear regression on affairs count
ols_model = sm.OLS(df["affairs"], X).fit()

print("SUMMARY BY CHILDREN (0=no,1=yes)")
print(summary)
print("\nMean difference (children yes - no) in affairs:", mean_diff_affairs)
print("Mean difference (children yes - no) in affair_any:", mean_diff_any)

print("\nLOGIT COEF for children_yes:", logit_model.params["children_yes"])
print("LOGIT p-value for children_yes:", logit_model.pvalues["children_yes"])
print("LOGIT odds ratio:", np.exp(logit_model.params["children_yes"]))

print("\nOLS COEF for children_yes:", ols_model.params["children_yes"])
print("OLS p-value for children_yes:", ols_model.pvalues["children_yes"])
print("OLS R2:", ols_model.rsquared)
