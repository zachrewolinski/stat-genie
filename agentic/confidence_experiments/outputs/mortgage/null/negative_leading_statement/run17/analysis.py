import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = "mortgage.csv"

df = pd.read_csv(path)

# Basic cleaning: drop unnamed index column if exists
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Ensure binary vars are numeric

# Define outcome: accept (1 accepted, 0 denied)
if "accept" in df.columns:
    df["accept"] = pd.to_numeric(df["accept"], errors="coerce")

# female variable
if "female" in df.columns:
    df["female"] = pd.to_numeric(df["female"], errors="coerce")

# Remove rows with missing key vars
key_cols = ["accept", "female"]

df_clean = df.dropna(subset=key_cols).copy()

# 1) Descriptive stats: approval rate by gender

approval_by_gender_raw = df_clean.groupby("female")["accept"].agg(["mean", "count"])
approval_by_gender = approval_by_gender_raw.rename(index={0: "male", 1: "female"})

# 2) Chi-square test of independence for accept vs female
contingency = pd.crosstab(df_clean["female"], df_clean["accept"])
chi2, p_chi, dof, exp = stats.chi2_contingency(contingency)

# 3) Logistic regression - unadjusted
# accept ~ female

model_unadj = smf.logit("accept ~ female", data=df_clean).fit(disp=False)

# 4) Logistic regression - adjusted for available covariates
# select a reasonable set of controls: credit & financial variables
covariates = [
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

# keep covariates present
covariates = [c for c in covariates if c in df_clean.columns]

# drop rows with missing covariates
all_cols = ["accept", "female"] + covariates

df_adj = df_clean.dropna(subset=all_cols).copy()

formula = "accept ~ female"
if covariates:
    formula += " + " + " + ".join(covariates)

model_adj = smf.logit(formula, data=df_adj).fit(disp=False)

# Extract female coefficient and p-value
unadj_coef = model_unadj.params.get("female", np.nan)
unadj_p = model_unadj.pvalues.get("female", np.nan)

adj_coef = model_adj.params.get("female", np.nan)
adj_p = model_adj.pvalues.get("female", np.nan)

# Convert to odds ratios
unadj_or = float(np.exp(unadj_coef))
adj_or = float(np.exp(adj_coef))

# Effect sizes: difference in approval rates
if 0 in approval_by_gender_raw.index and 1 in approval_by_gender_raw.index:
    approval_diff = approval_by_gender_raw.loc[1, "mean"] - approval_by_gender_raw.loc[0, "mean"]
else:
    approval_diff = np.nan

# Save results to json-like print for manual use

print("APPROVAL_BY_GENDER")
print(approval_by_gender)
print("\nCONTINGENCY")
print(contingency)
print(f"\nCHI2: {chi2:.4f}, p={p_chi:.6g}")
print("\nUNADJ_LOGIT")
print(model_unadj.summary().tables[1])
print(f"Unadj OR: {unadj_or:.4f}, p={unadj_p:.6g}")

print("\nADJ_LOGIT")
print(model_adj.summary().tables[1])
print(f"Adj OR: {adj_or:.4f}, p={adj_p:.6g}")
print(f"\nApproval rate diff (female - male): {approval_diff:.6f}")
print(f"\nN unadj: {len(df_clean)}, N adj: {len(df_adj)}")
