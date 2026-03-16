import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

print("shape", df.shape)
print("columns", list(df.columns))

# Identify outcome
if "accept" in df.columns:
    accept = df["accept"].copy()
elif "deny" in df.columns:
    accept = 1 - df["deny"].copy()
else:
    raise ValueError("No accept or deny column found")

# Ensure binary numeric
accept = pd.to_numeric(accept, errors="coerce")

if "female" not in df.columns:
    raise ValueError("No female column found")

female = pd.to_numeric(df["female"], errors="coerce")

# Candidate controls that are mortgage-relevant
candidate_controls = [
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
controls = [c for c in candidate_controls if c in df.columns]

use_cols = ["female"] + controls

tmp = df[use_cols].copy()

tmp["accept"] = accept

# Drop missing
analysis_df = tmp.dropna(subset=["accept", "female"] + controls)

# Ensure binary indicator in {0,1}
analysis_df = analysis_df[(analysis_df["accept"].isin([0, 1])) & (analysis_df["female"].isin([0, 1]))]

print("analysis_rows", len(analysis_df))

# Contingency table
ct = pd.crosstab(analysis_df["female"], analysis_df["accept"])
print("contingency\n", ct)

# Rates by gender
rates = analysis_df.groupby("female")["accept"].mean()
print("approval_rate_by_gender", rates.to_dict())
if 0 in rates.index and 1 in rates.index:
    diff = rates.loc[1] - rates.loc[0]
else:
    diff = np.nan
print("female_minus_male_rate_diff", diff)

# Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(ct)
print("chi2", chi2, "p", p)

# Unadjusted logistic regression
X1 = sm.add_constant(analysis_df[["female"]])
y = analysis_df["accept"]
model1 = sm.GLM(y, X1, family=sm.families.Binomial())
res1 = model1.fit(cov_type="HC1")
coef1 = res1.params["female"]
se1 = res1.bse["female"]
ci1 = res1.conf_int().loc["female"].tolist()
print("unadjusted_logit_coef", coef1, "se", se1, "p", res1.pvalues["female"], "ci", ci1)
print("unadjusted_odds_ratio", np.exp(coef1), "ci", [np.exp(ci1[0]), np.exp(ci1[1])])

# Adjusted logistic regression (if controls exist)
if controls:
    X2 = sm.add_constant(analysis_df[["female"] + controls])
    model2 = sm.GLM(y, X2, family=sm.families.Binomial())
    res2 = model2.fit(cov_type="HC1")
    coef2 = res2.params["female"]
    se2 = res2.bse["female"]
    ci2 = res2.conf_int().loc["female"].tolist()
    print("adjusted_logit_coef", coef2, "se", se2, "p", res2.pvalues["female"], "ci", ci2)
    print("adjusted_odds_ratio", np.exp(coef2), "ci", [np.exp(ci2[0]), np.exp(ci2[1])])
    print("controls_used", controls)
else:
    res2 = None

# Save a compact JSON summary for convenience
summary = {
    "n": int(len(analysis_df)),
    "approval_rate_female": float(rates.get(1, np.nan)),
    "approval_rate_male": float(rates.get(0, np.nan)),
    "rate_diff_female_minus_male": float(diff),
    "chi2_p": float(p),
    "unadjusted": {
        "coef": float(coef1),
        "p": float(res1.pvalues["female"]),
        "odds_ratio": float(np.exp(coef1)),
        "ci_low": float(np.exp(ci1[0])),
        "ci_high": float(np.exp(ci1[1])),
    },
}
if res2 is not None:
    summary["adjusted"] = {
        "coef": float(coef2),
        "p": float(res2.pvalues["female"]),
        "odds_ratio": float(np.exp(coef2)),
        "ci_low": float(np.exp(ci2[0])),
        "ci_high": float(np.exp(ci2[1])),
        "controls": controls,
    }

with open("analysis_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("wrote analysis_summary.json")
