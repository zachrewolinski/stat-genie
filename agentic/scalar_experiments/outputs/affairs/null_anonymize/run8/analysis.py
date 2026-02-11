import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
DF_PATH = "affairs.csv"
df = pd.read_csv(DF_PATH)

# Identify columns
children_col = "feature6"  # yes/no
outcome_col = "feature2"   # affairs count (coded)

# Prepare data
# Map children yes/no to 1/0
children = df[children_col].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
if children.isna().any():
    raise ValueError("Unexpected values in feature6 (children) column.")

outcome = pd.to_numeric(df[outcome_col], errors="coerce")

# Group summaries
means = outcome.groupby(children).mean()
medians = outcome.groupby(children).median()
counts = outcome.groupby(children).count()

mean_no_children = float(means.loc[0])
mean_children = float(means.loc[1])
median_no_children = float(medians.loc[0])
median_children = float(medians.loc[1])

# Two-sample tests
vals_no_children = outcome[children == 0].dropna()
vals_children = outcome[children == 1].dropna()

# Welch t-test
welch_t = stats.ttest_ind(vals_no_children, vals_children, equal_var=False, nan_policy="omit")

# Mann-Whitney U (nonparametric)
try:
    mwu = stats.mannwhitneyu(vals_no_children, vals_children, alternative="greater")
except Exception:
    mwu = None

# Regression: Poisson with robust SEs, controlling for covariates
# Build design matrix
# feature3 is gender (categorical). Others are numeric.
# Use all features except outcome and children col.

covariate_cols = [
    "feature3",  # gender
    "feature4",
    "feature5",
    "feature7",
    "feature8",
    "feature9",
    "feature10",
]

X = df[covariate_cols].copy()

# One-hot encode gender
X = pd.get_dummies(X, columns=["feature3"], drop_first=True)

# Add children indicator
X["children"] = children

# Add constant
X = sm.add_constant(X, has_constant="add")

y = outcome

# Fit Poisson model
poisson_model = sm.GLM(y, X, family=sm.families.Poisson())
poisson_res = poisson_model.fit(cov_type="HC3")

coef_children = float(poisson_res.params["children"])
pval_children = float(poisson_res.pvalues["children"])

# Determine direction: negative coefficient suggests fewer affairs with children
# Evidence primarily from regression p-value

if coef_children < 0 and pval_children < 0.05:
    response = "Yes"
elif coef_children > 0 and pval_children < 0.05:
    response = "No"
else:
    # Not statistically significant; default to No
    response = "No"

# Map p-value and direction to 0-100 scale
# Start at 50, move toward 100 for significant negative effect,
# toward 0 for significant positive effect, and toward 50 for non-significant

def scale_from_p(p, direction):
    # direction: -1 for negative effect, +1 for positive effect
    if p < 0.001:
        strength = 0.95
    elif p < 0.01:
        strength = 0.85
    elif p < 0.05:
        strength = 0.70
    elif p < 0.10:
        strength = 0.55
    else:
        strength = 0.50
    # direction shifts around 50
    return int(round(50 + (strength - 0.5) * 100 * (-1 if direction < 0 else 1)))

if coef_children < 0:
    scale = scale_from_p(pval_children, direction=-1)
else:
    scale = scale_from_p(pval_children, direction=1)

# Build explanation
explanation = (
    "Compared affair-frequency scores by presence of children and ran a Poisson regression "
    "controlling for gender, age, years married, religiosity, education, occupation, and marriage rating. "
    f"Mean affair score without children was {mean_no_children:.2f} (median {median_no_children:.2f}, n={int(counts.loc[0])}); "
    f"with children it was {mean_children:.2f} (median {median_children:.2f}, n={int(counts.loc[1])}). "
    f"Welch t-test p={welch_t.pvalue:.4g}; "
)
if mwu is not None:
    explanation += f"Mann-Whitney U (one-sided, no-children > children) p={mwu.pvalue:.4g}. "
else:
    explanation += "Mann-Whitney U test failed. "

explanation += (
    f"Poisson regression coefficient for children was {coef_children:.3f} with robust p={pval_children:.4g}. "
    "A negative coefficient indicates fewer affairs when children are present. "
)

if response == "Yes":
    explanation += "The negative and statistically significant association supports a decrease in affairs for those with children."
else:
    explanation += "The association is not statistically significant (or is positive), so there is insufficient evidence that children decrease affairs."

# Write conclusion
conclusion = {
    "response": response,
    "scale": int(scale),
    "explanation": explanation,
}

with open("conclusion.txt", "w") as f:
    json.dump(conclusion, f)
