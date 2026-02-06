import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

# Load data
_df = pd.read_csv("boxes.csv")

# Derived variables
_df = _df.assign(
    social=(_df["y"] != 1).astype(int),
    majority=(_df["y"] == 2).astype(int),
)

# Model 1: reliance on social info (choose majority or minority vs unchosen)
model_social_full = smf.logit("social ~ age + C(culture)", data=_df).fit(disp=False)
model_social_age = smf.logit("social ~ age", data=_df).fit(disp=False)
model_social_null = smf.logit("social ~ 1", data=_df).fit(disp=False)

# Likelihood ratio tests
lr_social_culture = 2 * (model_social_full.llf - model_social_age.llf)
df_social_culture = model_social_full.df_model - model_social_age.df_model
p_social_culture = chi2.sf(lr_social_culture, df_social_culture)

lr_social_age = 2 * (model_social_age.llf - model_social_null.llf)
df_social_age = model_social_age.df_model - model_social_null.df_model
p_social_age = chi2.sf(lr_social_age, df_social_age)

# Model 2: preference for majority cues among social choices
_df_social = _df[_df["y"] != 1].copy()
model_majority_full = smf.logit("majority ~ age + C(culture)", data=_df_social).fit(disp=False)
model_majority_age = smf.logit("majority ~ age", data=_df_social).fit(disp=False)
model_majority_null = smf.logit("majority ~ 1", data=_df_social).fit(disp=False)

lr_majority_culture = 2 * (model_majority_full.llf - model_majority_age.llf)
df_majority_culture = model_majority_full.df_model - model_majority_age.df_model
p_majority_culture = chi2.sf(lr_majority_culture, df_majority_culture)

lr_majority_age = 2 * (model_majority_age.llf - model_majority_null.llf)
df_majority_age = model_majority_age.df_model - model_majority_null.df_model
p_majority_age = chi2.sf(lr_majority_age, df_majority_age)

# Summaries
print("=== Reliance on social information (social vs unchosen) ===")
print(model_social_full.summary())
print("LR test culture effect (vs age-only):", lr_social_culture, "df=", df_social_culture, "p=", p_social_culture)
print("LR test age effect (vs null):", lr_social_age, "df=", df_social_age, "p=", p_social_age)

print("\n=== Majority preference among social choices ===")
print(model_majority_full.summary())
print("LR test culture effect (vs age-only):", lr_majority_culture, "df=", df_majority_culture, "p=", p_majority_culture)
print("LR test age effect (vs null):", lr_majority_age, "df=", df_majority_age, "p=", p_majority_age)

# Save key results for conclusion
results = {
    "p_social_culture": p_social_culture,
    "p_social_age": p_social_age,
    "p_majority_culture": p_majority_culture,
    "p_majority_age": p_majority_age,
}
print("\nKey p-values:", results)
