import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Compute efficiency: nuts opened per second
# Avoid division by zero (none expected but guard)
df = df.copy()
df["efficiency"] = df["feature5"] / df["feature6"].replace(0, np.nan)

# Basic summary
print("Rows:", len(df))
print(df[["feature2","feature3","feature7","feature5","feature6","efficiency"]].describe(include='all'))
print("Missing efficiency:", df["efficiency"].isna().sum())

# OLS regression with categorical sex and help
model = smf.ols("efficiency ~ feature2 + C(feature3) + C(feature7)", data=df).fit()
print(model.summary())

# ANOVA table for predictors (type II)
try:
    anova = sm.stats.anova_lm(model, typ=2)
    print("ANOVA (Type II):")
    print(anova)
except Exception as e:
    print("ANOVA failed:", e)

# Nonparametric check: Spearman correlation between age and efficiency
from scipy.stats import spearmanr

rho, pval = spearmanr(df["feature2"], df["efficiency"], nan_policy='omit')
print("Spearman age-efficiency rho:", rho, "p:", pval)

# Group comparisons for sex and help (t-test and Mann-Whitney)
from scipy.stats import ttest_ind, mannwhitneyu

# Sex groups
sex_vals = df["feature3"].dropna().unique()
if len(sex_vals) == 2:
    g1 = df[df["feature3"] == sex_vals[0]]["efficiency"].dropna()
    g2 = df[df["feature3"] == sex_vals[1]]["efficiency"].dropna()
    tstat, tp = ttest_ind(g1, g2, equal_var=False)
    ustat, up = mannwhitneyu(g1, g2, alternative='two-sided')
    print("Sex groups:", sex_vals)
    print("t-test p:", tp, "Mann-Whitney p:", up)

# Help groups
help_vals = df["feature7"].dropna().unique()
if len(help_vals) == 2:
    h1 = df[df["feature7"] == help_vals[0]]["efficiency"].dropna()
    h2 = df[df["feature7"] == help_vals[1]]["efficiency"].dropna()
    tstat, tp = ttest_ind(h1, h2, equal_var=False)
    ustat, up = mannwhitneyu(h1, h2, alternative='two-sided')
    print("Help groups:", help_vals)
    print("t-test p:", tp, "Mann-Whitney p:", up)

# Effect sizes (Cohen's d) helper

def cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = len(a)
    nb = len(b)
    if na < 2 or nb < 2:
        return np.nan
    sa = a.var(ddof=1)
    sb = b.var(ddof=1)
    s = np.sqrt(((na-1)*sa + (nb-1)*sb) / (na+nb-2))
    if s == 0:
        return np.nan
    return (a.mean() - b.mean()) / s

if len(sex_vals) == 2:
    d = cohens_d(g1, g2)
    print("Cohen's d (sex):", d)
if len(help_vals) == 2:
    d = cohens_d(h1, h2)
    print("Cohen's d (help):", d)

# Save basic stats for inspection
print("Mean efficiency:", df["efficiency"].mean())
print("Std efficiency:", df["efficiency"].std())
