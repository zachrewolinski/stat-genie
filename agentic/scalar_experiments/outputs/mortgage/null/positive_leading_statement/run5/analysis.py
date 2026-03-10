import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)
# Drop index-like column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Basic counts
n_total = len(df)

# Approval rates by gender
rate_female = df.loc[df["female"] == 1, "accept"].mean()
rate_male = df.loc[df["female"] == 0, "accept"].mean()
rate_diff = rate_female - rate_male

# Contingency table and chi-square test
cont = pd.crosstab(df["female"], df["accept"])
chi2, p_chi2, dof, expected = stats.chi2_contingency(cont)

# Logistic regression: unadjusted
model1 = smf.logit("accept ~ female", data=df).fit(disp=False)
coef1 = model1.params["female"]
p1 = model1.pvalues["female"]
or1 = float(np.exp(coef1))
ci1 = model1.conf_int().loc["female"]
or1_ci = (float(np.exp(ci1[0])), float(np.exp(ci1[1])))

# Logistic regression: adjusted with applicant/loan characteristics
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
formula = "accept ~ female + " + " + ".join(controls)
model2 = smf.logit(formula, data=df).fit(disp=False, maxiter=200)
coef2 = model2.params["female"]
p2 = model2.pvalues["female"]
or2 = float(np.exp(coef2))
ci2 = model2.conf_int().loc["female"]
or2_ci = (float(np.exp(ci2[0])), float(np.exp(ci2[1])))

# Pack results for downstream reasoning (printed as JSON)
results = {
    "n_total": int(n_total),
    "rate_female": float(rate_female),
    "rate_male": float(rate_male),
    "rate_diff": float(rate_diff),
    "chi2_p": float(p_chi2),
    "logit_unadjusted": {
        "coef_female": float(coef1),
        "p_female": float(p1),
        "or_female": float(or1),
        "or_female_ci": [float(or1_ci[0]), float(or1_ci[1])],
    },
    "logit_adjusted": {
        "coef_female": float(coef2),
        "p_female": float(p2),
        "or_female": float(or2),
        "or_female_ci": [float(or2_ci[0]), float(or2_ci[1])],
    },
}

print(json.dumps(results, indent=2))
