import json
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.api as sm

DATA_PATH = "mortgage.csv"

# Load data
_df = pd.read_csv(DATA_PATH)

# Define variables
gender = _df["feature2"]  # 1 female, 0 male
approved = _df["feature14"]  # 1 accepted, 0 denied

df = _df.copy()

# Drop rows with missing in relevant columns
base_cols = ["feature2", "feature14"]
base_df = df.dropna(subset=base_cols)

# Contingency table
ct = pd.crosstab(base_df["feature2"], base_df["feature14"])  # rows: gender, cols: approved
ct = ct.reindex(index=[0, 1], columns=[0, 1])

chi2, pval, dof, expected = chi2_contingency(ct)

# Rates
male_df = base_df[base_df["feature2"] == 0]
female_df = base_df[base_df["feature2"] == 1]

male_approval = (male_df["feature14"] == 1).mean()
female_approval = (female_df["feature14"] == 1).mean()

risk_diff = female_approval - male_approval

# Odds ratio (female vs male) with CI using Wald on log OR
# 2x2 table: rows gender (male=0, female=1), cols approved(0/1)
# a=female approved, b=female denied, c=male approved, d=male denied
try:
    a = ct.loc[1, 1]
    b = ct.loc[1, 0]
    c = ct.loc[0, 1]
    d = ct.loc[0, 0]
    if min(a, b, c, d) == 0:
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5
    or_female = (a / b) / (c / d)
    se_log_or = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    log_or = np.log(or_female)
    ci_low = np.exp(log_or - 1.96 * se_log_or)
    ci_high = np.exp(log_or + 1.96 * se_log_or)
except Exception:
    or_female = np.nan
    ci_low = np.nan
    ci_high = np.nan

# Adjusted logistic regression
# Exclude feature1 (likely id) and feature11 (denied, redundant)
control_cols = [
    "feature3", "feature4", "feature5", "feature6", "feature7",
    "feature8", "feature9", "feature10", "feature12", "feature13",
]

model_df = df[["feature14", "feature2"] + control_cols].dropna()
X = model_df[["feature2"] + control_cols]
X = sm.add_constant(X)
y = model_df["feature14"]

logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=False)

coef_gender = float(result.params.get("feature2"))
se_gender = float(result.bse.get("feature2"))
p_gender = float(result.pvalues.get("feature2"))

# Adjusted odds ratio and CI
adj_or = float(np.exp(coef_gender))
adj_ci_low = float(np.exp(coef_gender - 1.96 * se_gender))
adj_ci_high = float(np.exp(coef_gender + 1.96 * se_gender))

# Average marginal effect: difference in predicted approval if gender set to 1 vs 0
X1 = X.copy()
X0 = X.copy()
X1["feature2"] = 1
X0["feature2"] = 0
pred1 = result.predict(X1)
pred0 = result.predict(X0)
me_gender = float((pred1 - pred0).mean())
me_pp = me_gender * 100.0

# Decide response score
# Heuristic: if adjusted p < 0.05, "Yes" with magnitude based on effect size
# else "No" with score < 50, closer to 50 if unadjusted shows some difference.
if p_gender < 0.05:
    abs_pp = abs(me_pp)
    if abs_pp >= 10:
        score = 80
    elif abs_pp >= 5:
        score = 70
    elif abs_pp >= 2:
        score = 60
    else:
        score = 55
else:
    if pval < 0.05 and abs(risk_diff) >= 0.05:
        score = 45
    elif pval < 0.05 and abs(risk_diff) >= 0.02:
        score = 40
    else:
        score = 30

# Build explanation
explanation = (
    "I compared approval rates by gender and ran an adjusted logistic regression. "
    f"Unadjusted approval rate: males={male_approval:.3f}, females={female_approval:.3f} "
    f"(risk difference={risk_diff:.3f}; chi-square p={pval:.3g}). "
    "In the multivariable logit model controlling for credit and application factors "
    "(race, housing expense ratio, self-employment, marital status, mortgage and consumer credit scores, "
    "bad credit history, debt-to-income ratio, loan-to-value ratio, and PMI denial), "
    f"the gender coefficient had p={p_gender:.3g} with adjusted odds ratio {adj_or:.3f} "
    f"(95% CI {adj_ci_low:.3f}–{adj_ci_high:.3f}). "
    f"The average marginal effect of being female on approval probability was {me_gender:.4f} "
    f"(~{me_pp:.2f} percentage points). "
    "Given the adjusted results, I interpret this as "
    f"{'evidence of a gender effect on approval, but very small in magnitude' if p_gender < 0.05 else 'no robust evidence of a gender effect on approval'}.")

output = {"response": int(score), "explanation": explanation}

with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump(output, f)

# Print key stats for inspection
print(json.dumps({
    "n": int(len(df)),
    "male_approval": float(male_approval),
    "female_approval": float(female_approval),
    "risk_diff": float(risk_diff),
    "chi2_p": float(pval),
    "or_female": float(or_female),
    "or_ci": [float(ci_low), float(ci_high)],
    "logit_p_gender": float(p_gender),
    "adj_or_gender": float(adj_or),
    "adj_or_ci": [float(adj_ci_low), float(adj_ci_high)],
    "marginal_effect_gender": float(me_gender),
    "score": int(score)
}, indent=2))
