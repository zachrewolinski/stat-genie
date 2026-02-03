import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# Load data
DATA_PATH = "boxes.csv"
df = pd.read_csv(DATA_PATH)

# Define outcomes
# Reliance on social information: choosing majority or minority vs unchosen
# y: 1=unchosen, 2=majority, 3=minority

df["social_reliance"] = (df["y"] != 1).astype(int)

# Preference for majority cues among those who used social information
social_df = df[df["y"].isin([2, 3])].copy()
social_df["majority_choice"] = (social_df["y"] == 2).astype(int)

# Model 1: Does social reliance vary by age and culture?
model_social = smf.glm(
    formula="social_reliance ~ age + C(culture)",
    data=df,
    family=sm.families.Binomial()
).fit()

# Model 2: Does majority preference vary by age and culture (conditional on using social info)?
model_majority = smf.glm(
    formula="majority_choice ~ age + C(culture)",
    data=social_df,
    family=sm.families.Binomial()
).fit()

# Summaries
print("=== Social reliance model ===")
print(model_social.summary())
print("\n=== Majority preference model ===")
print(model_majority.summary())

# Extract key p-values
social_p_age = model_social.pvalues.get("age", float("nan"))
# Any culture effects (overall) via likelihood ratio test vs model without culture
model_social_no_culture = smf.glm(
    formula="social_reliance ~ age",
    data=df,
    family=sm.families.Binomial()
).fit()

lr_stat_social = 2 * (model_social.llf - model_social_no_culture.llf)
df_diff_social = model_social.df_model - model_social_no_culture.df_model
social_p_culture = stats.chi2.sf(lr_stat_social, df_diff_social)

majority_p_age = model_majority.pvalues.get("age", float("nan"))
model_majority_no_culture = smf.glm(
    formula="majority_choice ~ age",
    data=social_df,
    family=sm.families.Binomial()
).fit()

lr_stat_majority = 2 * (model_majority.llf - model_majority_no_culture.llf)
df_diff_majority = model_majority.df_model - model_majority_no_culture.df_model
majority_p_culture = stats.chi2.sf(lr_stat_majority, df_diff_majority)

print("\n=== Key p-values ===")
print(f"Social reliance: age p={social_p_age:.4g}, culture LRT p={social_p_culture:.4g}")
print(f"Majority preference: age p={majority_p_age:.4g}, culture LRT p={majority_p_culture:.4g}")
