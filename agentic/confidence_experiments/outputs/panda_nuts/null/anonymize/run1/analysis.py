import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

DATA_PATH = Path(__file__).with_name("panda_nuts.csv")

# Load data

df = pd.read_csv(DATA_PATH)

# Compute efficiency: nuts opened per second
# Guard against zero duration if any (though min is 2.5 per metadata)

df = df.copy()
df["efficiency"] = df["feature5"] / df["feature6"]

# Fit OLS with categorical predictors for sex and help
model = smf.ols("efficiency ~ feature2 + C(feature3) + C(feature7)", data=df).fit(cov_type="HC3")

# Extract results
params = model.params
pvalues = model.pvalues
conf_int = model.conf_int()

# Determine significance for each predictor
alpha = 0.05
sig = {k: (pvalues[k] < alpha) for k in pvalues.index}

# Build human-readable effects for categorical levels
# Reference levels are the first alphabetically by statsmodels; report coefficients accordingly.

# Decision logic:
# If any of age, sex, help show significant effects, answer Yes with strength based on effect sizes and overall R^2.
# Otherwise answer No.

# Compute standardized effect for age to gauge magnitude (beta * sd_x / sd_y)

y_sd = df["efficiency"].std(ddof=0)
age_sd = df["feature2"].std(ddof=0)
std_beta_age = None
if "feature2" in params and y_sd > 0 and age_sd > 0:
    std_beta_age = params["feature2"] * age_sd / y_sd

# Overall model fit
r2 = model.rsquared

# Identify any significant predictors among age, sex, help
predictor_keys = ["feature2", "C(feature3)[T.m]", "C(feature7)[T.y]"]
# Depending on levels, keys may differ; capture any term with feature3 or feature7
sig_predictors = []
for term in pvalues.index:
    if term == "feature2" or term.startswith("C(feature3)") or term.startswith("C(feature7)"):
        if pvalues[term] < alpha:
            sig_predictors.append(term)

# Decide response score
if sig_predictors:
    # Scale score by strength: base 65, increase with R2 and number of significant predictors
    score = 65
    score += int(min(20, r2 * 100 / 2))  # up to +20
    score += min(10, 5 * len(sig_predictors))
    score = max(60, min(95, score))
    answer = "Yes"
else:
    # No evidence: base 35, decrease with high p-values and low R2
    score = 35
    score -= int(min(15, (0.2 - r2) * 100 / 2)) if r2 < 0.2 else 0
    score = max(5, min(45, score))
    answer = "No"

# Build explanation
lines = []
lines.append(
    f"Outcome: nut-cracking efficiency defined as nuts opened per second (feature5/feature6)."
)
lines.append(
    f"Model: OLS efficiency ~ age (feature2) + sex (feature3) + help received (feature7) with HC3 robust SEs."
)
lines.append(
    f"Sample size: {len(df)} observations. Model R^2 = {r2:.3f}."
)

# Add predictor details
for term in pvalues.index:
    if term == "Intercept":
        continue
    if term == "feature2" or term.startswith("C(feature3)") or term.startswith("C(feature7)"):
        coef = params[term]
        pval = pvalues[term]
        ci_low, ci_high = conf_int.loc[term]
        lines.append(
            f"{term}: coef={coef:.4f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}], p={pval:.4f}."
        )

if std_beta_age is not None:
    lines.append(
        f"Standardized age effect (approx): {std_beta_age:.3f} SD change in efficiency per 1 SD age."
    )

if sig_predictors:
    lines.append(
        f"Significant predictors at alpha=0.05: {', '.join(sig_predictors)}. This indicates evidence that at least one of age, sex, or help influences efficiency."
    )
else:
    lines.append(
        "No predictors (age, sex, help) were statistically significant at alpha=0.05; evidence does not support an influence on efficiency in this sample."
    )

explanation = " ".join(lines)

# Write conclusion.txt
conclusion = {"response": int(score), "explanation": explanation}

out_path = Path(__file__).with_name("conclusion.txt")
out_path.write_text(json.dumps(conclusion))
