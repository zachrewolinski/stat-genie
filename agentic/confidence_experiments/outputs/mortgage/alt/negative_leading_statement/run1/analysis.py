import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
file_path = "mortgage.csv"
df = pd.read_csv(file_path)

# Drop index-like column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Basic counts
n_total = len(df)

# Acceptance rate by gender
# female: 1 if applicant is female, 0 if male
# accept: 1 if accepted, 0 if denied
rate_by_gender = df.groupby("female")["accept"].mean().rename({0: "male", 1: "female"})
counts_by_gender = df.groupby("female")["accept"].agg(["count", "sum"]).rename(index={0: "male", 1: "female"})

# Contingency table for chi-square test (female vs accept)
contingency = pd.crosstab(df["female"], df["accept"])
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

# Logistic regression
# We'll model accept as function of female + controls
controls = [
    "black",
    "housing_expense_ratio",
    "self_employed",
    "married",
    "mortgage_credit",
    "consumer_credit",
    "bad_history",
    "PI_ratio",
    "loan_to_value",
    "denied_PMI",
]

# Remove rows with missing values in relevant columns
model_cols = ["accept", "female"] + controls
model_df = df[model_cols].dropna()

formula = "accept ~ female + " + " + ".join(controls)
logit_model = smf.logit(formula, data=model_df).fit(disp=False)

# Extract female coefficient and odds ratio
coef_female = logit_model.params.get("female", np.nan)
se_female = logit_model.bse.get("female", np.nan)
p_female = logit_model.pvalues.get("female", np.nan)

odds_ratio = np.exp(coef_female)
conf_int = logit_model.conf_int().loc["female"]
conf_int_odds = np.exp(conf_int)

# Print results for interpretation
print("Total rows:", n_total)
print("Acceptance rates by gender:")
print(rate_by_gender)
print("Counts by gender (count, sum accepted):")
print(counts_by_gender)
print("Chi-square test p-value:", p_chi2)
print("Logit female coef:", coef_female)
print("Logit female SE:", se_female)
print("Logit female p-value:", p_female)
print("Logit female odds ratio:", odds_ratio)
print("Logit female OR 95% CI:", conf_int_odds.values)
print("Logit N used:", len(model_df))
