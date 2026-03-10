import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
csv_path = "teachingratings.csv"
df = pd.read_csv(csv_path)

# Keep relevant columns for analysis
# Ensure expected columns exist
required_cols = [
    "eval",
    "beauty",
    "age",
    "gender",
    "credits",
    "division",
    "native",
    "tenure",
    "students",
    "allstudents",
    "minority",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Drop rows with missing key variables
analysis_df = df[required_cols].dropna().copy()

# Basic stats
n = len(analysis_df)

# Correlation
corr = analysis_df["beauty"].corr(analysis_df["eval"])

# Unadjusted model
model_simple = smf.ols("eval ~ beauty", data=analysis_df).fit(cov_type="HC3")

# Adjusted model with common controls
model_adj = smf.ols(
    "eval ~ beauty + age + C(gender) + C(credits) + C(division) + C(native) + C(tenure) + C(minority) + students + allstudents",
    data=analysis_df,
).fit(cov_type="HC3")

# Extract key stats
coef_simple = model_simple.params.get("beauty", np.nan)
p_simple = model_simple.pvalues.get("beauty", np.nan)

coef_adj = model_adj.params.get("beauty", np.nan)
p_adj = model_adj.pvalues.get("beauty", np.nan)

# Effect size in SD units (standardized beta) for adjusted model
# standardize beauty and eval
z_df = analysis_df.copy()
z_df["beauty_z"] = (z_df["beauty"] - z_df["beauty"].mean()) / z_df["beauty"].std(ddof=0)
z_df["eval_z"] = (z_df["eval"] - z_df["eval"].mean()) / z_df["eval"].std(ddof=0)
model_adj_std = smf.ols(
    "eval_z ~ beauty_z + age + C(gender) + C(credits) + C(division) + C(native) + C(tenure) + C(minority) + students + allstudents",
    data=z_df,
).fit(cov_type="HC3")
std_beta = model_adj_std.params.get("beauty_z", np.nan)

# Decide response scale
# Heuristic: if p_adj < 0.05 and coef_adj > 0, lean Yes; strength by effect size and p-value
if np.isfinite(p_adj) and p_adj < 0.05 and coef_adj > 0:
    # base around 65-85 depending on effect size and p-value
    strength = min(1.0, abs(std_beta) / 0.3)  # 0.3 ~ moderate
    response = int(round(65 + 30 * strength))
elif np.isfinite(p_adj) and p_adj < 0.05 and coef_adj < 0:
    # significant negative relationship (opposite)
    strength = min(1.0, abs(std_beta) / 0.3)
    response = int(round(35 - 30 * strength))
else:
    # not significant
    response = 50

# Build explanation
explanation = (
    f"Analysis used {n} courses with non-missing eval and beauty. "
    f"The simple correlation between beauty and eval is {corr:.3f}. "
    f"In an unadjusted OLS model (eval ~ beauty), the beauty coefficient is {coef_simple:.3f} (HC3 p={p_simple:.4f}). "
    f"In an adjusted model controlling for age, gender, credits, division, native language, tenure status, minority status, "
    f"students, and class size, the beauty coefficient is {coef_adj:.3f} (HC3 p={p_adj:.4f}). "
    f"The standardized effect in the adjusted model is {std_beta:.3f} SD of eval per 1 SD of beauty. "
)

# Label conclusion based on response
if response >= 60:
    conclusion = "Overall, the evidence supports a positive relationship between instructor beauty and teaching evaluations."
elif response <= 40:
    conclusion = "Overall, the evidence does not support a positive relationship; if anything, it suggests the opposite."
else:
    conclusion = "Overall, the evidence is insufficient to conclude that instructor beauty affects teaching evaluations."

explanation += conclusion

# Write conclusion.txt
output = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(output, f)

# Print brief output for debugging
print(json.dumps(output, indent=2))
