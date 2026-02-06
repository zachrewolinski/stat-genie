import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv("boxes.csv")

# Define outcomes
_df["social"] = (_df["y"] != 1).astype(int)  # 1=chose demonstrated (majority or minority)
_df["majority"] = (_df["y"] == 2).astype(int)

# Helper for likelihood-ratio test

def lr_test(model_full, model_reduced):
    lr_stat = 2 * (model_full.llf - model_reduced.llf)
    df_diff = model_full.df_model - model_reduced.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value

results = {}

# Social reliance model
social_full = smf.logit("social ~ age + C(culture)", data=_df).fit(disp=0)
social_age_only = smf.logit("social ~ age", data=_df).fit(disp=0)
social_culture_only = smf.logit("social ~ C(culture)", data=_df).fit(disp=0)

results["social_age_lr"] = lr_test(social_full, social_culture_only)
results["social_culture_lr"] = lr_test(social_full, social_age_only)

# Interaction test for developmental differences by culture
social_interaction = smf.logit("social ~ age * C(culture)", data=_df).fit(disp=0)
results["social_interaction_lr"] = lr_test(social_interaction, social_full)

# Majority preference among social choosers
_df_social = _df[_df["social"] == 1].copy()
maj_full = smf.logit("majority ~ age + C(culture)", data=_df_social).fit(disp=0)
maj_age_only = smf.logit("majority ~ age", data=_df_social).fit(disp=0)
maj_culture_only = smf.logit("majority ~ C(culture)", data=_df_social).fit(disp=0)

results["maj_age_lr"] = lr_test(maj_full, maj_culture_only)
results["maj_culture_lr"] = lr_test(maj_full, maj_age_only)

maj_interaction = smf.logit("majority ~ age * C(culture)", data=_df_social).fit(disp=0)
results["maj_interaction_lr"] = lr_test(maj_interaction, maj_full)

# Descriptive summaries
social_by_culture = _df.groupby("culture")["social"].mean()
maj_by_culture = _df_social.groupby("culture")["majority"].mean()

# Age bins for descriptive trend
age_bins = pd.cut(_df["age"], bins=[3,5,8,11,14], labels=["4-5","6-8","9-11","12-14"])
_df["age_bin"] = age_bins
_df_social["age_bin"] = pd.cut(_df_social["age"], bins=[3,5,8,11,14], labels=["4-5","6-8","9-11","12-14"])
social_by_age = _df.groupby("age_bin")["social"].mean()
maj_by_age = _df_social.groupby("age_bin")["majority"].mean()

# Print key stats
print("N total:", len(_df))
print("N social:", len(_df_social))
print("Social reliance LRT age (vs culture-only):", results["social_age_lr"])
print("Social reliance LRT culture (vs age-only):", results["social_culture_lr"])
print("Social reliance LRT interaction:", results["social_interaction_lr"])
print("Majority preference LRT age (vs culture-only):", results["maj_age_lr"])
print("Majority preference LRT culture (vs age-only):", results["maj_culture_lr"])
print("Majority preference LRT interaction:", results["maj_interaction_lr"])

print("\nSocial reliance by culture (mean):")
print(social_by_culture)
print("\nMajority preference by culture (mean, social only):")
print(maj_by_culture)
print("\nSocial reliance by age bin (mean):")
print(social_by_age)
print("\nMajority preference by age bin (mean, social only):")
print(maj_by_age)
