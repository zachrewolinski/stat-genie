import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
df = pd.read_csv("affairs.csv")

# Rename columns for readability
df = df.rename(columns={
    "feature1": "id",
    "feature2": "affairs",
    "feature3": "gender",
    "feature4": "age",
    "feature5": "years_married",
    "feature6": "children",
    "feature7": "religiousness",
    "feature8": "education",
    "feature9": "occupation",
    "feature10": "marriage_rating",
})

# Clean / encode
df["children"] = df["children"].astype(str).str.lower().str.strip()
df["has_children"] = (df["children"] == "yes").astype(int)

# Basic group stats
group_stats = df.groupby("has_children")["affairs"].agg(["count", "mean", "median", "std"])

# Effect size (Cohen's d)
grp0 = df.loc[df.has_children == 0, "affairs"]
grp1 = df.loc[df.has_children == 1, "affairs"]
pooled_sd = np.sqrt(((grp0.var(ddof=1) * (len(grp0)-1)) + (grp1.var(ddof=1) * (len(grp1)-1))) / (len(grp0)+len(grp1)-2))
cohen_d = (grp1.mean() - grp0.mean()) / pooled_sd

# Two-sample tests
ttest = stats.ttest_ind(grp1, grp0, equal_var=False, nan_policy='omit')
mannwhitney = stats.mannwhitneyu(grp1, grp0, alternative='two-sided')

# OLS regression with controls
# Use robust (HC3) standard errors
formula = "affairs ~ has_children + C(gender) + age + years_married + religiousness + education + occupation + marriage_rating"
model = smf.ols(formula, data=df).fit(cov_type="HC3")

# Output key results
print("Group stats (has_children=0 no, 1 yes):")
print(group_stats)
print("\nCohen's d (children vs no children):", cohen_d)
print("\nWelch t-test:", ttest)
print("Mann-Whitney U:", mannwhitney)
print("\nOLS with controls (robust SE):")
print(model.summary().tables[1])

# Save key numbers for manual use
results = {
    "mean_no_children": grp0.mean(),
    "mean_children": grp1.mean(),
    "median_no_children": grp0.median(),
    "median_children": grp1.median(),
    "cohen_d": cohen_d,
    "ttest_stat": float(ttest.statistic),
    "ttest_p": float(ttest.pvalue),
    "mw_stat": float(mannwhitney.statistic),
    "mw_p": float(mannwhitney.pvalue),
    "ols_coef_children": float(model.params["has_children"]),
    "ols_p_children": float(model.pvalues["has_children"]),
}
print("\nKey results:")
for k, v in results.items():
    print(f"{k}: {v}")
