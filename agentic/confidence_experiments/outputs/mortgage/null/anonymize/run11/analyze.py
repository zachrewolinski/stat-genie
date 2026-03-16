import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2_contingency

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

# Outcome: approval (1=accepted, 0=denied)
if "feature14" not in df.columns:
    raise ValueError("Expected feature14 for approval status")

df = df.copy()

# Sanity check: feature11 appears to be denial. Ensure consistency.
if "feature11" in df.columns:
    # feature11 is 1 if denied, 0 if accepted
    # Check if feature11 is complement of feature14
    mismatch = (df["feature11"] + df["feature14"]).value_counts().to_dict()
else:
    mismatch = None

# Unadjusted approval rates by gender
female = df["feature2"]
approval = df["feature14"]

ct = pd.crosstab(female, approval)
# Ensure both levels exist
chi2, p_chi2, dof, expected = chi2_contingency(ct)

# Logistic regression with controls
# Exclude feature1 (likely id), feature11/feature14 (outcome), keep others
predictor_cols = [
    "feature2",
    "feature3",
    "feature4",
    "feature5",
    "feature6",
    "feature7",
    "feature8",
    "feature9",
    "feature10",
    "feature12",
    "feature13",
]

model_df = df[predictor_cols + ["feature14"]].dropna()

X = model_df[predictor_cols]
X = sm.add_constant(X, has_constant="add")
y = model_df["feature14"]

logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=False)

coef = result.params["feature2"]
se = result.bse["feature2"]
p_value = result.pvalues["feature2"]

# Odds ratio and 95% CI
odds_ratio = float(np.exp(coef))
ci_low = float(np.exp(coef - 1.96 * se))
ci_high = float(np.exp(coef + 1.96 * se))

# Unadjusted approval rates
approval_rate_female = ct.loc[1, 1] / ct.loc[1].sum() if 1 in ct.index else None
approval_rate_male = ct.loc[0, 1] / ct.loc[0].sum() if 0 in ct.index else None
rate_diff = approval_rate_female - approval_rate_male

summary = {
    "n_rows": int(df.shape[0]),
    "ct": ct.to_dict(),
    "chi2_p": float(p_chi2),
    "approval_rate_female": float(approval_rate_female),
    "approval_rate_male": float(approval_rate_male),
    "rate_diff": float(rate_diff),
    "logit_coef_female": float(coef),
    "logit_p_female": float(p_value),
    "odds_ratio_female": float(odds_ratio),
    "odds_ratio_ci_low": ci_low,
    "odds_ratio_ci_high": ci_high,
    "mismatch_check": mismatch,
}

print(json.dumps(summary, indent=2))
