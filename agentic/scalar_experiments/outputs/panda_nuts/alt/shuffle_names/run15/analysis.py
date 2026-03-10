import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv("panda_nuts.csv")

# Map columns to meanings based on info.json metadata
# age_years -> column 'hammer'
# sex -> column 'nuts_opened' (values m/f)
# help_received -> column 'seconds' (values y/N)
# nuts_opened_count -> column 'help'
# duration_seconds -> column 'chimpanzee'

# Create efficiency as nuts opened per second

df = df.copy()
df["age_years"] = df["hammer"].astype(float)
df["sex_mf"] = df["nuts_opened"].astype(str)
df["help_received"] = df["seconds"].astype(str)
df["nuts_opened_count"] = df["help"].astype(float)
df["duration_seconds"] = df["chimpanzee"].astype(float)

# avoid divide by zero

df = df[df["duration_seconds"] > 0].copy()
df["efficiency"] = df["nuts_opened_count"] / df["duration_seconds"]

# Standardize categorical values

df["sex_mf"] = df["sex_mf"].str.lower().str.strip()
df["help_received"] = df["help_received"].str.lower().str.strip()

# Fit OLS with robust SE
model = smf.ols("efficiency ~ age_years + C(sex_mf) + C(help_received)", data=df).fit(cov_type="HC3")

print("N", len(df))
print(model.summary())

# Also test overall model (ANOVA) and effect sizes (partial eta squared) using statsmodels
# Compute ANOVA table (type II)
try:
    import statsmodels.stats.anova as anova
    anova_table = anova.anova_lm(model, typ=2)
    print("\nANOVA type II:")
    print(anova_table)
except Exception as e:
    print("ANOVA failed", e)

# Simple group means for context
print("\nGroup means by sex:")
print(df.groupby("sex_mf")["efficiency"].mean())
print("\nGroup means by help_received:")
print(df.groupby("help_received")["efficiency"].mean())

# Correlation with age
print("\nCorrelation age-efficiency:")
print(df["age_years"].corr(df["efficiency"]))
