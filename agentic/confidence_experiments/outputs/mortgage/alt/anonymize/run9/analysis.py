import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)
# Clean inf -> NaN
_df = df.replace([np.inf, -np.inf], np.nan)

# Map columns
female = _df["feature2"]  # 1 female, 0 male
approved = _df["feature14"]  # 1 accepted, 0 denied

# Basic counts
n_total = len(_df)

# Contingency table and rates (dropna only for these two columns)
ct_df = _df[["feature2", "feature14"]].dropna()
ct = pd.crosstab(ct_df["feature2"], ct_df["feature14"])
for col in [0, 1]:
    if col not in ct.columns:
        ct[col] = 0
ct = ct[[0, 1]]

rates = ct[1] / ct.sum(axis=1)

# Chi-square test of independence
chi2, p_chi2, dof, expected = stats.chi2_contingency(ct.values)

# Unadjusted logistic regression (approved ~ female)
logit_unadj = sm.Logit(ct_df["feature14"], sm.add_constant(ct_df["feature2"])).fit(disp=False)

# Adjusted logistic regression controlling for other observed covariates
control_cols = [
    "feature3",  # black
    "feature4",  # housing expense ratio
    "feature5",  # self-employed
    "feature6",  # married
    "feature7",  # mortgage credit score
    "feature8",  # consumer credit score
    "feature9",  # bad credit history
    "feature10", # debt-to-income
    "feature12", # loan-to-value
    "feature13", # PMI denied
]

adj_cols = ["feature14", "feature2"] + control_cols
adj_df = _df[adj_cols].dropna()
logit_adj = sm.Logit(adj_df["feature14"], sm.add_constant(adj_df[["feature2"] + control_cols])).fit(disp=False)

# Extract coefficients and odds ratios
coef_unadj = float(logit_unadj.params["feature2"])
se_unadj = float(logit_unadj.bse["feature2"])
p_unadj = float(logit_unadj.pvalues["feature2"])

coef_adj = float(logit_adj.params["feature2"])
se_adj = float(logit_adj.bse["feature2"])
p_adj = float(logit_adj.pvalues["feature2"])

or_unadj = float(np.exp(coef_unadj))
or_adj = float(np.exp(coef_adj))

ci_unadj = logit_unadj.conf_int().loc["feature2"].values
ci_adj = logit_adj.conf_int().loc["feature2"].values
or_ci_unadj = np.exp(ci_unadj)
or_ci_adj = np.exp(ci_adj)

print("n_total", n_total)
print("n_ct", len(ct_df))
print("n_adj", len(adj_df))
print("contingency_table\n", ct)
print("approval_rates", rates.to_dict())
print("chi2", chi2, "p", p_chi2)
print("unadjusted_logit coef", coef_unadj, "se", se_unadj, "p", p_unadj, "or", or_unadj, "or_ci", or_ci_unadj)
print("adjusted_logit coef", coef_adj, "se", se_adj, "p", p_adj, "or", or_adj, "or_ci", or_ci_adj)
