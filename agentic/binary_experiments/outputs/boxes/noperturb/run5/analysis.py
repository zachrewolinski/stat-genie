import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("boxes.csv")

# Derived outcomes
_df["social_reliance"] = (_df["y"] != 1).astype(int)  # 1 if chose demonstrated option (majority or minority)
_df["majority_choice"] = (_df["y"] == 2).astype(int)
_df["minority_choice"] = (_df["y"] == 3).astype(int)

# Restrict to demonstrated choices for majority preference (exclude unchosen option)
_demo = _df[_df["y"].isin([2, 3])].copy()
_demo["majority_over_minority"] = (_demo["y"] == 2).astype(int)

# Model 1: reliance on social information vs age and culture (with interaction)
model_social = smf.glm(
    "social_reliance ~ age * C(culture)",
    data=_df,
    family=sm.families.Binomial()
).fit()

# Model 2: majority preference among demonstrated choices vs age and culture (with interaction)
model_majority = smf.glm(
    "majority_over_minority ~ age * C(culture)",
    data=_demo,
    family=sm.families.Binomial()
).fit()

# Summaries
print("=== Social Reliance Model (Binomial GLM) ===")
print(model_social.summary())
print("\n=== Majority Preference Model (Binomial GLM) ===")
print(model_majority.summary())

# Simple descriptive statistics
summary = (
    _df.groupby(["culture", "age"])["y"]
    .value_counts(normalize=True)
    .rename("proportion")
    .reset_index()
)

print("\n=== Proportions by culture and age (y=1 unchosen, y=2 majority, y=3 minority) ===")
print(summary.head(30))

# Aggregate proportions by culture
culture_summary = (
    _df.groupby("culture")["y"]
    .value_counts(normalize=True)
    .rename("proportion")
    .reset_index()
)
print("\n=== Proportions by culture ===")
print(culture_summary)

# Aggregate proportions by age
age_summary = (
    _df.groupby("age")["y"]
    .value_counts(normalize=True)
    .rename("proportion")
    .reset_index()
)
print("\n=== Proportions by age ===")
print(age_summary)
