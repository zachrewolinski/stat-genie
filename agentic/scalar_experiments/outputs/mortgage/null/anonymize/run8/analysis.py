import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

# Identify columns
female_col = "feature2"  # 1 if female, 0 if male
accept_col = "feature14"  # 1 if accepted, 0 if denied

# Basic counts and acceptance rates
ct = pd.crosstab(df[female_col], df[accept_col])
# Ensure columns order 0,1
ct = ct.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

# Chi-square test of independence
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)

# Acceptance rates by gender
accept_rate_male = ct.loc[0, 1] / ct.loc[0].sum() if ct.loc[0].sum() > 0 else np.nan
accept_rate_female = ct.loc[1, 1] / ct.loc[1].sum() if ct.loc[1].sum() > 0 else np.nan
rate_diff = accept_rate_female - accept_rate_male

# Two-proportion z-test for rate difference
count = np.array([ct.loc[1, 1], ct.loc[0, 1]])
nobs = np.array([ct.loc[1].sum(), ct.loc[0].sum()])
prop_test = sm.stats.proportions_ztest(count, nobs, alternative="two-sided")

# 95% CI for difference in proportions (Wald)
prop1 = accept_rate_female
prop0 = accept_rate_male
se_diff = np.sqrt(prop1 * (1 - prop1) / nobs[0] + prop0 * (1 - prop0) / nobs[1])
ci_low = rate_diff - 1.96 * se_diff
ci_high = rate_diff + 1.96 * se_diff

# Logistic regression controlling for other variables
# Exclude target and obviously redundant/ID-like fields
candidate_features = [
    "feature2",  # female
    "feature3",  # black
    "feature4",  # housing expense ratio
    "feature5",  # self-employed
    "feature6",  # married
    "feature7",  # mortgage credit score
    "feature8",  # consumer credit score
    "feature9",  # bad credit history
    "feature10", # debt-to-income ratio
    "feature12", # loan-to-value ratio
    "feature13", # denied PMI
]

X = df[candidate_features].copy()
y = df[accept_col].copy()

# Drop rows with missing or infinite values
mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
X = X.loc[mask].copy()
y = y.loc[mask].copy()

# Add intercept
X = sm.add_constant(X, has_constant="add")

logit_model = sm.Logit(y, X)
logit_result = logit_model.fit(disp=False)

coef = logit_result.params[female_col]
se = logit_result.bse[female_col]
# Odds ratio and 95% CI
or_female = float(np.exp(coef))
or_ci_low = float(np.exp(coef - 1.96 * se))
or_ci_high = float(np.exp(coef + 1.96 * se))
p_logit = float(logit_result.pvalues[female_col])

results = {
    "contingency_table": ct.to_dict(),
    "chi2": float(chi2),
    "chi2_p": float(p_chi),
    "accept_rate_male": float(accept_rate_male),
    "accept_rate_female": float(accept_rate_female),
    "rate_diff_female_minus_male": float(rate_diff),
    "diff_ci_95": [float(ci_low), float(ci_high)],
    "prop_z_p": float(prop_test[1]),
    "logit_odds_ratio_female": or_female,
    "logit_or_ci_95": [or_ci_low, or_ci_high],
    "logit_p_female": p_logit,
    "logit_n": int(logit_result.nobs),
}

print(json.dumps(results, indent=2))
