import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv("mortgage.csv")

# Ensure binary numeric
for col in ["female", "accept", "deny", "black", "self_employed", "married", "bad_history", "denied_PMI"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Basic counts
n = len(df)

# Approval rates by gender
approval_by_gender = df.groupby("female")["accept"].mean()
counts = df.groupby("female")["accept"].agg(["count", "sum"])

# Chi-square test of independence
contingency = pd.crosstab(df["female"], df["accept"])
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

# Logistic regression: unadjusted and adjusted
results = {}

model1 = smf.logit("accept ~ female", data=df).fit(disp=False)
results["unadjusted"] = model1

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

formula = "accept ~ female + " + " + ".join(covariates)
model2 = smf.logit(formula, data=df).fit(disp=False)
results["adjusted"] = model2


def summarize_model(model):
    coef = model.params["female"]
    se = model.bse["female"]
    p = model.pvalues["female"]
    ci_low, ci_high = model.conf_int().loc["female"]
    or_est = np.exp(coef)
    or_low, or_high = np.exp([ci_low, ci_high])
    return {
        "coef": coef,
        "se": se,
        "p": p,
        "or": or_est,
        "or_ci": (or_low, or_high),
    }

summary_unadj = summarize_model(model1)
summary_adj = summarize_model(model2)

print("N", n)
print("Approval rates by female (0=male,1=female):")
print(approval_by_gender)
print("Counts by female:")
print(counts)
print("Chi-square test female vs accept: chi2=%.4f p=%.6f" % (chi2, p_chi2))
print("Unadjusted logit female coef=%.4f p=%.6f OR=%.4f CI=[%.4f, %.4f]" % (
    summary_unadj["coef"], summary_unadj["p"], summary_unadj["or"], summary_unadj["or_ci"][0], summary_unadj["or_ci"][1]
))
print("Adjusted logit female coef=%.4f p=%.6f OR=%.4f CI=[%.4f, %.4f]" % (
    summary_adj["coef"], summary_adj["p"], summary_adj["or"], summary_adj["or_ci"][0], summary_adj["or_ci"][1]
))
