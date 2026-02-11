import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = "affairs.csv"
df = pd.read_csv(path)

# Basic cleaning
# feature2 is frequency of affairs; treat as numeric
# feature6 is children (yes/no)

df = df.copy()

# Create binary has_affair

df["has_affair"] = (df["feature2"] > 0).astype(int)

# Group stats

groups = df.groupby("feature6")
mean_affairs = groups["feature2"].mean()
median_affairs = groups["feature2"].median()
prop_affairs = groups["has_affair"].mean()
counts = groups["feature2"].count()

# Two-sample t-test (Welch)

yes = df[df["feature6"] == "yes"]["feature2"]
no = df[df["feature6"] == "no"]["feature2"]

ttest = stats.ttest_ind(yes, no, equal_var=False, nan_policy="omit")

# Mann-Whitney U (non-parametric)
# Use alternative='two-sided' to keep general; we will check sign separately
mw = stats.mannwhitneyu(yes, no, alternative="two-sided")

# Proportion test (chi-square) for any affair
contingency = pd.crosstab(df["feature6"], df["has_affair"])
chi2, chi2_p, _, _ = stats.chi2_contingency(contingency)

# Logistic regression with controls
# Controls based on available metadata
# feature3 gender, feature4 age, feature5 years married, feature7 religiosity,
# feature8 education, feature9 occupation, feature10 marriage rating
# Use categorical for gender

logit_model = smf.logit(
    "has_affair ~ C(feature3) + feature4 + feature5 + C(feature6) + feature7 + feature8 + feature9 + feature10",
    data=df,
).fit(disp=False)

# Extract coefficient and p-value for children=yes (relative to no)
# statsmodels encodes C(feature6)[T.yes]
coef = logit_model.params.get("C(feature6)[T.yes]", np.nan)
pval = logit_model.pvalues.get("C(feature6)[T.yes]", np.nan)

# OLS on log1p of affairs for robustness
ols_model = smf.ols(
    "np.log1p(feature2) ~ C(feature3) + feature4 + feature5 + C(feature6) + feature7 + feature8 + feature9 + feature10",
    data=df,
).fit()
ols_coef = ols_model.params.get("C(feature6)[T.yes]", np.nan)
ols_pval = ols_model.pvalues.get("C(feature6)[T.yes]", np.nan)

results = {
    "mean_affairs": mean_affairs.to_dict(),
    "median_affairs": median_affairs.to_dict(),
    "prop_affairs": prop_affairs.to_dict(),
    "counts": counts.to_dict(),
    "ttest": {"stat": ttest.statistic, "pvalue": ttest.pvalue},
    "mannwhitney": {"stat": mw.statistic, "pvalue": mw.pvalue},
    "chi2": {"stat": chi2, "pvalue": chi2_p},
    "logit_children_yes": {"coef": coef, "pvalue": pval},
    "ols_log1p_children_yes": {"coef": ols_coef, "pvalue": ols_pval},
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
