import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
csv_path = "teachingratings.csv"
df = pd.read_csv(csv_path)

# Basic cleaning: drop rows with missing key variables
key_vars = ["eval", "beauty"]
df_clean = df.dropna(subset=key_vars).copy()

# Pearson correlation
corr, corr_p = stats.pearsonr(df_clean["beauty"], df_clean["eval"])

# Simple OLS
model_simple = smf.ols("eval ~ beauty", data=df_clean).fit(cov_type="HC3")

# Multiple OLS with controls
# Use available covariates; treat categorical variables as factors
formula_controls = (
    "eval ~ beauty + age + C(gender) + C(minority) + C(native) + "
    "C(tenure) + C(division) + C(credits) + students + allstudents"
)
model_controls = smf.ols(formula_controls, data=df_clean).fit(cov_type="HC3")

# Extract results
simple_coef = model_simple.params.get("beauty", float("nan"))
simple_p = model_simple.pvalues.get("beauty", float("nan"))

controls_coef = model_controls.params.get("beauty", float("nan"))
controls_p = model_controls.pvalues.get("beauty", float("nan"))

n = int(df_clean.shape[0])

# Decide response scale
# Heuristic:
# - If beauty coefficient is significant (p<0.05) in both simple and controls -> Yes, strength based on effect size
# - If only in simple but not controls -> weak evidence, closer to neutral
# - If not significant -> No

# Effect size: scale coefficient by eval SD to get standardized-like effect
# (beauty is roughly standardized; we use eval SD for interpretability)
eval_sd = df_clean["eval"].std(ddof=1)
std_effect = controls_coef / eval_sd if eval_sd and np.isfinite(controls_coef) else np.nan

response = None

if (simple_p < 0.05) and (controls_p < 0.05):
    # positive or negative matters; if negative significant, still "Yes" (affect) but direction negative
    # Strength based on |std_effect|
    abs_eff = abs(std_effect)
    if abs_eff >= 0.3:
        response = 75
    elif abs_eff >= 0.2:
        response = 68
    elif abs_eff >= 0.1:
        response = 60
    else:
        response = 55
elif (simple_p < 0.05) and (controls_p >= 0.05):
    response = 45
else:
    response = 30

# Build explanation
explanation = (
    f"Analyzed {n} courses. Beauty and evaluation scores show a Pearson correlation of "
    f"r = {corr:.3f} (p = {corr_p:.3g}). In a simple regression (eval ~ beauty), "
    f"the beauty coefficient is {simple_coef:.3f} (p = {simple_p:.3g}). "
    f"With controls for age, gender, minority, native status, tenure, division, credits, "
    f"and class size (students and allstudents), the beauty coefficient is {controls_coef:.3f} "
    f"(p = {controls_p:.3g}). The standardized magnitude relative to eval SD is about {std_effect:.3f}. "
)

# Add interpretation
if (simple_p < 0.05) and (controls_p < 0.05):
    direction = "positive" if controls_coef > 0 else "negative"
    explanation += (
        f"Because the beauty effect remains statistically significant after controls and is {direction}, "
        "there is evidence that instructor beauty affects student instructional ratings, though the effect "
        "size appears modest."
    )
elif (simple_p < 0.05) and (controls_p >= 0.05):
    explanation += (
        "The association is significant in the unadjusted model but not after adding controls, so evidence "
        "for an independent beauty effect is weak."
    )
else:
    explanation += (
        "Beauty is not statistically significant once controls are considered, indicating little evidence that "
        "beauty independently affects instructional ratings."
    )

# Write conclusion
output = {
    "response": int(response),
    "explanation": explanation
}

with open("conclusion.txt", "w") as f:
    json.dump(output, f)
