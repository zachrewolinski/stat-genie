import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
csv_path = "mortgage.csv"

df = pd.read_csv(csv_path)

# Key variables based on info.json
# feature2: 1 if female, 0 if male
# feature14: 1 if accepted, 0 if denied

gender = df["feature2"]
approval = df["feature14"]

# Basic counts
cont_table = pd.crosstab(gender, approval)
# Ensure ordering: rows 0=male,1=female; cols 0=denied,1=accepted
cont_table = cont_table.reindex(index=[0.0,1.0], columns=[0,1], fill_value=0)

# Approval rates by gender
male_rate = cont_table.loc[0.0,1] / cont_table.loc[0.0].sum()
female_rate = cont_table.loc[1.0,1] / cont_table.loc[1.0].sum()
rate_diff = female_rate - male_rate

# Chi-square test for independence
chi2, p_chi, dof, expected = stats.chi2_contingency(cont_table.values)

# Two-proportion z-test (approx) for difference in approval rates
# Use pooled proportion
n_m = cont_table.loc[0.0].sum()
n_f = cont_table.loc[1.0].sum()
prop_pool = (cont_table.loc[0.0,1] + cont_table.loc[1.0,1]) / (n_m + n_f)
se_pool = np.sqrt(prop_pool * (1 - prop_pool) * (1/n_m + 1/n_f))
if se_pool == 0:
    z_stat = np.nan
    p_z = np.nan
else:
    z_stat = rate_diff / se_pool
    p_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))

# Logistic regression: approval ~ gender (female)
X1 = sm.add_constant(gender)
logit1 = sm.Logit(approval, X1, missing='drop')
res1 = logit1.fit(disp=False)

# Logistic regression with controls
control_cols = [
    "feature3",  # Black
    "feature4",  # housing expense ratio
    "feature5",  # self-employed
    "feature6",  # married
    "feature7",  # mortgage credit score
    "feature8",  # consumer credit score
    "feature9",  # bad credit
    "feature10", # debt ratio
    "feature12", # loan-to-value
    "feature13", # PMI denied
]

X2 = df[["feature2"] + control_cols].copy()
X2 = sm.add_constant(X2)
logit2 = sm.Logit(approval, X2, missing='drop')
res2 = logit2.fit(disp=False)

# Extract gender effect
coef1 = res1.params["feature2"]
pval1 = res1.pvalues["feature2"]
OR1 = np.exp(coef1)

coef2 = res2.params["feature2"]
pval2 = res2.pvalues["feature2"]
OR2 = np.exp(coef2)
ci2 = res2.conf_int().loc["feature2"].tolist()
OR2_ci = [float(np.exp(ci2[0])), float(np.exp(ci2[1]))]

# Collect results
results = {
    "n_total": int(df.shape[0]),
    "n_male": int(n_m),
    "n_female": int(n_f),
    "male_approval_rate": male_rate,
    "female_approval_rate": female_rate,
    "approval_rate_diff_female_minus_male": rate_diff,
    "chi2_pvalue": p_chi,
    "ztest_pvalue": p_z,
    "logit_unadjusted_coef": coef1,
    "logit_unadjusted_or": OR1,
    "logit_unadjusted_pvalue": pval1,
    "logit_adjusted_coef": coef2,
    "logit_adjusted_or": OR2,
    "logit_adjusted_pvalue": pval2,
    "logit_adjusted_or_ci95": OR2_ci,
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
