import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats

# Load data
path = "mortgage.csv"
df = pd.read_csv(path)

# Clean columns
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Define variables
outcome = "accept"  # 1 if accepted
female = "female"   # 1 if female

# Drop rows with missing values in key columns
key_cols = [outcome, female]
df_key = df.dropna(subset=key_cols).copy()

# Basic counts
n_total = len(df_key)

# Acceptance rates by gender
rates = df_key.groupby(female)[outcome].mean()
counts = df_key.groupby(female)[outcome].agg(["count", "sum"])

# Proportion test (female vs male)
# female=1, male=0
count_accept = np.array([counts.loc[1, "sum"], counts.loc[0, "sum"]])
count_total = np.array([counts.loc[1, "count"], counts.loc[0, "count"]])
stat, pval_z = proportions_ztest(count_accept, count_total, alternative='two-sided')

# Chi-square test
contingency = pd.crosstab(df_key[female], df_key[outcome])
chi2, pval_chi, dof, expected = stats.chi2_contingency(contingency)

# Unadjusted logistic regression
X_unadj = sm.add_constant(df_key[[female]])
y = df_key[outcome]
logit_unadj = sm.Logit(y, X_unadj).fit(disp=False)

# Adjusted logistic regression with controls
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

# Ensure controls exist
control_cols = [c for c in control_cols if c in df.columns]

adj_cols = [female] + control_cols

df_adj = df.dropna(subset=adj_cols + [outcome]).copy()
X_adj = sm.add_constant(df_adj[adj_cols])
y_adj = df_adj[outcome]

logit_adj = sm.Logit(y_adj, X_adj).fit(disp=False)

# Extract female effect
coef_unadj = logit_unadj.params[female]
se_unadj = logit_unadj.bse[female]
p_unadj = logit_unadj.pvalues[female]

coef_adj = logit_adj.params[female]
se_adj = logit_adj.bse[female]
p_adj = logit_adj.pvalues[female]

# Odds ratios and 95% CI
or_unadj = float(np.exp(coef_unadj))
or_adj = float(np.exp(coef_adj))

ci_unadj = np.exp([coef_unadj - 1.96*se_unadj, coef_unadj + 1.96*se_unadj])
ci_adj = np.exp([coef_adj - 1.96*se_adj, coef_adj + 1.96*se_adj])

# Build summary
summary = {
    "n_total": int(n_total),
    "accept_rate_female": float(rates.loc[1]),
    "accept_rate_male": float(rates.loc[0]),
    "accept_rate_diff_female_minus_male": float(rates.loc[1] - rates.loc[0]),
    "ztest_pvalue": float(pval_z),
    "chi2_pvalue": float(pval_chi),
    "unadjusted": {
        "coef_female": float(coef_unadj),
        "odds_ratio": float(or_unadj),
        "ci95": [float(ci_unadj[0]), float(ci_unadj[1])],
        "pvalue": float(p_unadj),
    },
    "adjusted": {
        "coef_female": float(coef_adj),
        "odds_ratio": float(or_adj),
        "ci95": [float(ci_adj[0]), float(ci_adj[1])],
        "pvalue": float(p_adj),
        "n_used": int(len(df_adj)),
    }
}

with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
