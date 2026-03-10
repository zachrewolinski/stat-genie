import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
path = "panda_nuts.csv"

df = pd.read_csv(path)

# Rename columns for clarity
col_map = {
    "feature1": "id",
    "feature2": "age",
    "feature3": "sex",
    "feature4": "hammer",
    "feature5": "nuts_opened",
    "feature6": "duration_s",
    "feature7": "help"
}

df = df.rename(columns=col_map)

# Compute efficiency: nuts opened per second
# Add a small guard for zero durations (not expected per metadata)
if (df["duration_s"] <= 0).any():
    df = df[df["duration_s"] > 0].copy()

df["efficiency"] = df["nuts_opened"] / df["duration_s"]

# Basic stats
print("Rows:", len(df))
print(df[["age", "sex", "help", "nuts_opened", "duration_s", "efficiency"]].describe(include="all"))

# Fit OLS with categorical factors
model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit()
print("\nOLS summary:\n", model.summary())

# Type II ANOVA for overall factor significance
anova = sm.stats.anova_lm(model, typ=2)
print("\nType II ANOVA:\n", anova)

# Also check nonparametric correlations for age
from scipy import stats

pearson = stats.pearsonr(df["age"], df["efficiency"])
spearman = stats.spearmanr(df["age"], df["efficiency"])
print("\nAge-efficiency Pearson r, p:", pearson)
print("Age-efficiency Spearman rho, p:", spearman)

# Group comparisons for sex and help (t-test and Mann-Whitney)
for var in ["sex", "help"]:
    groups = [g["efficiency"].values for _, g in df.groupby(var)]
    if len(groups) == 2:
        t = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        u = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        print(f"\n{var} t-test:", t)
        print(f"{var} Mann-Whitney U:", u)
    else:
        print(f"\n{var} has {len(groups)} groups; skipping two-group tests")
