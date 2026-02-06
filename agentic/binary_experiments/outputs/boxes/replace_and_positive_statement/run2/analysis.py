import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as st

# Load data
DF_PATH = "boxes.csv"
df = pd.read_csv(DF_PATH)

# Derived outcomes
# social_choice: chose an option demonstrated by someone (majority or minority)
# majority_choice: among those who chose demonstrated options, did they choose majority?
df["social_choice"] = (df["y"] != 1).astype(int)
df["majority_choice"] = (df["y"] == 2).astype(int)

# Age groups for descriptive summaries
bins = [3, 6, 9, 12, 15]
labels = ["4-6", "7-9", "10-12", "13-14"]
df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

# Descriptive rates by culture and age group
social_rates = (
    df.groupby(["culture", "age_group"], dropna=False)["social_choice"]
    .mean()
    .unstack()
)
majority_rates = (
    df[df["social_choice"] == 1]
    .groupby(["culture", "age_group"], dropna=False)["majority_choice"]
    .mean()
    .unstack()
)

print("Social choice rates (P(chose demonstrated option)) by culture x age group:")
print(social_rates.round(3))
print("\nMajority choice rates (P(majority | chose demonstrated)) by culture x age group:")
print(majority_rates.round(3))


def lr_test(model_full, model_reduced):
    lr_stat = 2 * (model_full.llf - model_reduced.llf)
    df = int(model_full.df_model - model_reduced.df_model)
    p = st.chi2.sf(lr_stat, df)
    return lr_stat, df, p


# Logistic regression models for social_choice
m0_social = smf.logit(
    "social_choice ~ age + gender + majority_first", data=df
).fit(disp=False)

m1_social = smf.logit(
    "social_choice ~ age + C(culture) + gender + majority_first", data=df
).fit(disp=False)

m2_social = smf.logit(
    "social_choice ~ age + C(culture) + age:C(culture) + gender + majority_first",
    data=df,
).fit(disp=False)

lr_culture_social = lr_test(m1_social, m0_social)
lr_interaction_social = lr_test(m2_social, m1_social)

print("\nLogit: social_choice ~ age + culture (+ controls)")
print("LR test for culture (m1 vs m0): stat=%.3f, df=%d, p=%.4g" % lr_culture_social)
print("LR test for age x culture interaction (m2 vs m1): stat=%.3f, df=%d, p=%.4g" % lr_interaction_social)

# Logistic regression models for majority_choice among social choosers
social_df = df[df["social_choice"] == 1].copy()

m0_maj = smf.logit(
    "majority_choice ~ age + gender + majority_first", data=social_df
).fit(disp=False)

m1_maj = smf.logit(
    "majority_choice ~ age + C(culture) + gender + majority_first", data=social_df
).fit(disp=False)

m2_maj = smf.logit(
    "majority_choice ~ age + C(culture) + age:C(culture) + gender + majority_first",
    data=social_df,
).fit(disp=False)

lr_culture_maj = lr_test(m1_maj, m0_maj)
lr_interaction_maj = lr_test(m2_maj, m1_maj)

print("\nLogit: majority_choice | social_choice ~ age + culture (+ controls)")
print("LR test for culture (m1 vs m0): stat=%.3f, df=%d, p=%.4g" % lr_culture_maj)
print("LR test for age x culture interaction (m2 vs m1): stat=%.3f, df=%d, p=%.4g" % lr_interaction_maj)

# Also report main effects of age in the models with culture (for developmental stage effect)
print("\nAge effect (coefficient, p-value) in culture-adjusted models:")
print("social_choice: coef=%.3f, p=%.4g" % (m1_social.params["age"], m1_social.pvalues["age"]))
print("majority_choice: coef=%.3f, p=%.4g" % (m1_maj.params["age"], m1_maj.pvalues["age"]))
