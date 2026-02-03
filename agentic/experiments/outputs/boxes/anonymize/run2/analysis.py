import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = "boxes.csv"
df = pd.read_csv(path)

# Rename for clarity
# feature1: outcome (1=unchosen, 2=majority, 3=minority)
# feature3: age
# feature5: site ID

df = df.copy()
df["site"] = df["feature5"].astype("category")
df["age"] = df["feature3"].astype(float)

# Social choice: chose either majority or minority

df["social"] = (df["feature1"] != 1).astype(int)
# Majority choice among social choices

df["majority"] = (df["feature1"] == 2).astype(int)

# Helper: likelihood ratio test

def lr_test(full, reduced):
    lr_stat = 2 * (full.llf - reduced.llf)
    df_diff = full.df_model - reduced.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value

results = {}

# Model A: social reliance ~ age + site
model_social_full = smf.logit("social ~ age + C(site)", data=df).fit(disp=False)
model_social_age_only = smf.logit("social ~ age", data=df).fit(disp=False)
model_social_site_only = smf.logit("social ~ C(site)", data=df).fit(disp=False)

lr_age_social = lr_test(model_social_full, model_social_site_only)
lr_site_social = lr_test(model_social_full, model_social_age_only)

results["social_lr_age"] = lr_age_social
results["social_lr_site"] = lr_site_social

# Model B: majority preference among social choices
social_df = df[df["social"] == 1].copy()
model_majority_full = smf.logit("majority ~ age + C(site)", data=social_df).fit(disp=False)
model_majority_age_only = smf.logit("majority ~ age", data=social_df).fit(disp=False)
model_majority_site_only = smf.logit("majority ~ C(site)", data=social_df).fit(disp=False)

lr_age_majority = lr_test(model_majority_full, model_majority_site_only)
lr_site_majority = lr_test(model_majority_full, model_majority_age_only)

results["majority_lr_age"] = lr_age_majority
results["majority_lr_site"] = lr_site_majority

# Descriptives by age group and site
bins = [3, 6, 9, 12, 14.5]
labels = ["4-6", "7-9", "10-12", "13-14"]
df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)

# Proportions overall by age group
age_group_stats = df.groupby("age_group").apply(
    lambda g: pd.Series({
        "n": len(g),
        "social_rate": g["social"].mean(),
        "majority_rate": g.loc[g["social"] == 1, "majority"].mean()
    })
).reset_index()

# Proportions by site
site_stats = df.groupby("site").apply(
    lambda g: pd.Series({
        "n": len(g),
        "social_rate": g["social"].mean(),
        "majority_rate": g.loc[g["social"] == 1, "majority"].mean()
    })
).reset_index()

# Save outputs for review
age_group_stats.to_csv("age_group_stats.csv", index=False)
site_stats.to_csv("site_stats.csv", index=False)

print("LR tests (stat, df, p):")
print("social ~ age + site: age effect", results["social_lr_age"])
print("social ~ age + site: site effect", results["social_lr_site"])
print("majority ~ age + site: age effect", results["majority_lr_age"])
print("majority ~ age + site: site effect", results["majority_lr_site"])
print("\nAge group stats:\n", age_group_stats)
print("\nSite stats:\n", site_stats)
