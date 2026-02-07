import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
file_path = "affairs.csv"
df = pd.read_csv(file_path)

# Map columns
# feature2: frequency of extramarital sex in past year
# feature6: children (yes/no)

# Clean
# Ensure feature6 is binary indicator
children = df["feature6"].astype(str).str.lower().map({"yes": 1, "no": 0})

# Outcomes
freq = df["feature2"].astype(float)
any_affair = (freq > 0).astype(int)

# Basic group stats
summary = df.groupby("feature6")["feature2"].agg(["count", "mean", "median", "std"])

# Proportion with any affair by children
prop_any = df.assign(any_affair=any_affair).groupby("feature6")["any_affair"].mean()

# t-test for mean difference
freq_yes = freq[children == 1]
freq_no = freq[children == 0]
t_stat, p_value = stats.ttest_ind(freq_yes, freq_no, equal_var=False, nan_policy="omit")

# Mann-Whitney U
u_stat, u_p = stats.mannwhitneyu(freq_yes, freq_no, alternative="two-sided")

# Cohen's d
n1, n0 = freq_yes.size, freq_no.size
s1, s0 = freq_yes.std(ddof=1), freq_no.std(ddof=1)
pooled = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
cohen_d = (freq_yes.mean() - freq_no.mean()) / pooled if pooled != 0 else np.nan

# Regression: logistic on any affair with controls
# Controls (feature3 gender, feature4 age, feature5 years married, feature7 religiousness,
# feature8 education, feature9 occupation, feature10 marriage rating)
reg_df = df.copy()
reg_df["children"] = children
reg_df["any_affair"] = any_affair

# Handle categorical gender
reg_df["gender"] = reg_df["feature3"].astype("category")

# Build logistic regression
formula = (
    "any_affair ~ children + C(gender) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10"
)
logit_model = smf.logit(formula=formula, data=reg_df).fit(disp=False)

# Linear regression on frequency (OLS) for interpretability
ols_formula = (
    "feature2 ~ children + C(gender) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10"
)
ols_model = smf.ols(formula=ols_formula, data=reg_df).fit()

# Output results
print("Group summary (feature2 by children):")
print(summary)
print("\nProportion with any affair by children:")
print(prop_any)
print("\nT-test (mean difference yes-no):")
print({"t_stat": t_stat, "p_value": p_value})
print("\nMann-Whitney U:")
print({"u_stat": u_stat, "p_value": u_p})
print("\nCohen's d (yes-no):", cohen_d)

print("\nLogit regression (any affair):")
print(logit_model.summary().tables[1])

print("\nOLS regression (frequency):")
print(ols_model.summary().tables[1])
