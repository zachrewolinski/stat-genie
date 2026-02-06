import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2

# Load data
DF_PATH = "boxes.csv"
df = pd.read_csv(DF_PATH)

# Keep relevant columns
cols = ["y", "age", "culture"]
df = df[cols].copy()

# Drop rows with missing key data
before_rows = len(df)
df = df.dropna(subset=cols)

after_rows = len(df)

# Ensure types
# y should be integers 1,2,3
# culture is a site id, treat as categorical
# age is numeric
# Coerce y to int if possible
try:
    df["y"] = df["y"].astype(int)
except ValueError:
    df["y"] = pd.to_numeric(df["y"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)

# Derived outcomes
# social_choice: chose any demonstrated option (majority or minority)
# majority_choice: chose majority among demonstrated options

df["social_choice"] = (df["y"] != 1).astype(int)

# Define helper for likelihood ratio test

def lr_test(model_full, model_restricted):
    lr = 2 * (model_full.llf - model_restricted.llf)
    df_diff = model_full.df_model - model_restricted.df_model
    p_value = chi2.sf(lr, df_diff)
    return lr, int(df_diff), p_value

# Model 1: social choice ~ age + culture
model_social_full = smf.logit("social_choice ~ age + C(culture)", data=df).fit(disp=0)
model_social_age = smf.logit("social_choice ~ age", data=df).fit(disp=0)
model_social_culture = smf.logit("social_choice ~ C(culture)", data=df).fit(disp=0)

lr_culture_social = lr_test(model_social_full, model_social_age)
lr_age_social = lr_test(model_social_full, model_social_culture)

# Model 2: majority preference among demonstrated options
# Filter to those who chose majority or minority

df_social = df[df["y"].isin([2, 3])].copy()
df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

model_maj_full = smf.logit("majority_choice ~ age + C(culture)", data=df_social).fit(disp=0)
model_maj_age = smf.logit("majority_choice ~ age", data=df_social).fit(disp=0)
model_maj_culture = smf.logit("majority_choice ~ C(culture)", data=df_social).fit(disp=0)

lr_culture_maj = lr_test(model_maj_full, model_maj_age)
lr_age_maj = lr_test(model_maj_full, model_maj_culture)

# Descriptive rates by culture and age quantiles
age_bins = pd.qcut(df["age"], q=4, duplicates="drop")
rate_by_age = df.groupby(age_bins)["social_choice"].mean()
rate_majority_by_age = df_social.groupby(pd.qcut(df_social["age"], q=4, duplicates="drop"))["majority_choice"].mean()

rate_by_culture = df.groupby("culture")["social_choice"].mean().sort_index()
rate_majority_by_culture = df_social.groupby("culture")["majority_choice"].mean().sort_index()

# Print results
print("Rows before drop:", before_rows)
print("Rows after drop:", after_rows)
print("Age range:", df["age"].min(), df["age"].max())
print("\nSocial choice model (y!=1):")
print(model_social_full.summary())
print("LR test culture effect (full vs age-only):", lr_culture_social)
print("LR test age effect (full vs culture-only):", lr_age_social)

print("\nMajority choice model (y==2 among y in {2,3}):")
print(model_maj_full.summary())
print("LR test culture effect (full vs age-only):", lr_culture_maj)
print("LR test age effect (full vs culture-only):", lr_age_maj)

print("\nRates by age quartile (social choice):")
print(rate_by_age)
print("\nRates by age quartile (majority choice):")
print(rate_majority_by_age)

print("\nRates by culture (social choice):")
print(rate_by_culture)
print("\nRates by culture (majority choice):")
print(rate_majority_by_culture)
