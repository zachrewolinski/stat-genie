import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# Load data
DATA_PATH = "boxes.csv"
df = pd.read_csv(DATA_PATH)

# Create indicators
# Social reliance: chose majority or minority (vs unchosen option)
df['social_choice'] = df['y'].isin([2, 3]).astype(int)
# Majority preference among social choices
social_df = df[df['social_choice'] == 1].copy()
social_df['majority_choice'] = (social_df['y'] == 2).astype(int)

# Treat culture as categorical

df['culture'] = df['culture'].astype('category')
social_df['culture'] = social_df['culture'].astype('category')

# Logistic regression for social reliance
model_social = smf.logit('social_choice ~ age + C(culture)', data=df).fit(disp=False)

# Logistic regression for majority preference among social choices
model_majority = smf.logit('majority_choice ~ age + C(culture)', data=social_df).fit(disp=False)

# Also test interaction with age for culture (developmental differences by culture)
model_social_int = smf.logit('social_choice ~ age * C(culture)', data=df).fit(disp=False)
model_majority_int = smf.logit('majority_choice ~ age * C(culture)', data=social_df).fit(disp=False)

# Likelihood ratio tests for interaction vs main effects
lr_social = 2 * (model_social_int.llf - model_social.llf)
lr_majority = 2 * (model_majority_int.llf - model_majority.llf)

df_social = model_social_int.df_model - model_social.df_model
df_majority = model_majority_int.df_model - model_majority.df_model

p_social = stats.chi2.sf(lr_social, df_social)
p_majority = stats.chi2.sf(lr_majority, df_majority)

# Summary stats by culture and age bins for interpretability
age_bins = pd.cut(df['age'], bins=[3.5, 6, 9, 12, 14.5], labels=['4-6', '7-9', '10-12', '13-14'])
df['age_bin'] = age_bins
social_rate_by = df.groupby(['culture', 'age_bin'])['social_choice'].mean().unstack()
majority_rate_by = social_df.assign(age_bin=pd.cut(social_df['age'], bins=[3.5, 6, 9, 12, 14.5], labels=['4-6', '7-9', '10-12', '13-14']))\
    .groupby(['culture', 'age_bin'])['majority_choice'].mean().unstack()

# Print key results
print("=== Social reliance (choose majority/minority vs unchosen) ===")
print(model_social.summary())
print("\nInteraction LR test (age*culture):")
print(f"LR stat={lr_social:.3f}, df={df_social}, p={p_social:.4g}")

print("\n=== Majority preference among social choices ===")
print(model_majority.summary())
print("\nInteraction LR test (age*culture):")
print(f"LR stat={lr_majority:.3f}, df={df_majority}, p={p_majority:.4g}")

print("\n=== Social choice rate by culture and age bin ===")
print(social_rate_by)
print("\n=== Majority choice rate by culture and age bin (social choices only) ===")
print(majority_rate_by)
