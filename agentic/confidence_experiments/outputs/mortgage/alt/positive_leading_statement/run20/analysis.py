import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleanup
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Ensure expected binary columns are numeric 0/1
binary_cols = ["female", "black", "self_employed", "married", "bad_history", "deny", "denied_PMI", "accept"]
for col in binary_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing key variables
key_cols = ["female", "deny", "accept"]
missing_before = df[key_cols].isna().any(axis=1).sum()

df = df.dropna(subset=key_cols)

# Descriptive stats: denial and acceptance rates by gender
rates = (
    df.groupby("female")["deny", "accept"].mean()
    .rename(index={0: "male", 1: "female"})
)
counts = df.groupby("female").size().rename(index={0: "male", 1: "female"})

# Two-proportion z-test for denial rates
male = df[df["female"] == 0]
female = df[df["female"] == 1]

n_male = len(male)
n_female = len(female)

deny_male = male["deny"].mean()
deny_female = female["deny"].mean()

# Pooled proportion for z-test
p_pool = (male["deny"].sum() + female["deny"].sum()) / (n_male + n_female)
se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_male + 1 / n_female))
if se_pool > 0:
    z_stat = (deny_female - deny_male) / se_pool
    p_value_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))
else:
    z_stat = np.nan
    p_value_z = np.nan

# Chi-square test of independence
cont_table = pd.crosstab(df["female"], df["deny"])
chi2, chi2_p, chi2_dof, _ = stats.chi2_contingency(cont_table)

# Logistic regression: deny ~ female (unadjusted)
model_unadj = smf.logit("deny ~ female", data=df).fit(disp=0)

# Logistic regression: deny ~ female + controls
control_cols = [
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
controls = [c for c in control_cols if c in df.columns]

formula_adj = "deny ~ female"
if controls:
    formula_adj += " + " + " + ".join(controls)

model_adj = smf.logit(formula_adj, data=df).fit(disp=0)

# Alternative adjusted model excluding denied_PMI (potential post-application outcome)
controls_no_pmi = [c for c in controls if c != "denied_PMI"]
formula_adj_no_pmi = "deny ~ female"
if controls_no_pmi:
    formula_adj_no_pmi += " + " + " + ".join(controls_no_pmi)
model_adj_no_pmi = smf.logit(formula_adj_no_pmi, data=df).fit(disp=0)

# Extract coefficients and odds ratios for female
coef_unadj = model_unadj.params["female"]
se_unadj = model_unadj.bse["female"]
p_unadj = model_unadj.pvalues["female"]
or_unadj = np.exp(coef_unadj)

coef_adj = model_adj.params["female"]
se_adj = model_adj.bse["female"]
p_adj = model_adj.pvalues["female"]
or_adj = np.exp(coef_adj)

# 95% CI for odds ratios
ci_unadj = model_unadj.conf_int().loc["female"].to_numpy()
ci_adj = model_adj.conf_int().loc["female"].to_numpy()
ci_or_unadj = np.exp(ci_unadj)
ci_or_adj = np.exp(ci_adj)

coef_adj_no_pmi = model_adj_no_pmi.params["female"]
p_adj_no_pmi = model_adj_no_pmi.pvalues["female"]
or_adj_no_pmi = np.exp(coef_adj_no_pmi)
ci_adj_no_pmi = model_adj_no_pmi.conf_int().loc["female"].to_numpy()
ci_or_adj_no_pmi = np.exp(ci_adj_no_pmi)

results = {
    "n_total": int(len(df)),
    "n_male": int(n_male),
    "n_female": int(n_female),
    "deny_rate_male": float(deny_male),
    "deny_rate_female": float(deny_female),
    "accept_rate_male": float(male["accept"].mean()),
    "accept_rate_female": float(female["accept"].mean()),
    "z_stat": float(z_stat),
    "p_value_z": float(p_value_z),
    "chi2_p": float(chi2_p),
    "or_unadj": float(or_unadj),
    "p_unadj": float(p_unadj),
    "ci_or_unadj_low": float(ci_or_unadj[0]),
    "ci_or_unadj_high": float(ci_or_unadj[1]),
    "or_adj": float(or_adj),
    "p_adj": float(p_adj),
    "ci_or_adj_low": float(ci_or_adj[0]),
    "ci_or_adj_high": float(ci_or_adj[1]),
    "or_adj_no_pmi": float(or_adj_no_pmi),
    "p_adj_no_pmi": float(p_adj_no_pmi),
    "ci_or_adj_no_pmi_low": float(ci_or_adj_no_pmi[0]),
    "ci_or_adj_no_pmi_high": float(ci_or_adj_no_pmi[1]),
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
