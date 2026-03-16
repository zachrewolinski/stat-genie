import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
DATA_PATH = "panda_nuts.csv"

df = pd.read_csv(DATA_PATH)

# Rename columns for clarity
rename_map = {
    "feature1": "id",
    "feature2": "age",
    "feature3": "sex",
    "feature4": "hammer",
    "feature5": "nuts_opened",
    "feature6": "duration_seconds",
    "feature7": "help",
}

df = df.rename(columns=rename_map)

# Basic cleaning
# Ensure categorical columns are strings
for col in ["sex", "hammer", "help"]:
    if col in df.columns:
        df[col] = df[col].astype(str)

# Compute efficiency (nuts per second) and per minute for interpretation
# Guard against any zero durations
if (df["duration_seconds"] <= 0).any():
    df = df.loc[df["duration_seconds"] > 0].copy()

df["efficiency"] = df["nuts_opened"] / df["duration_seconds"]

df["efficiency_per_min"] = df["efficiency"] * 60.0

# Fit OLS model with robust (HC3) standard errors
model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit(cov_type="HC3")

# Extract key stats
params = model.params
pvalues = model.pvalues
r2 = model.rsquared
n = int(model.nobs)

# Summaries for interpretation
mean_eff = df["efficiency_per_min"].mean()
std_eff = df["efficiency_per_min"].std(ddof=1)

# Group means for categorical predictors (per-minute scale for readability)
sex_means = df.groupby("sex", dropna=False)["efficiency_per_min"].mean().to_dict()
help_means = df.groupby("help", dropna=False)["efficiency_per_min"].mean().to_dict()

# Age effect in per-minute units
age_coef_per_min = params.get("age", np.nan) * 60.0
age_p = pvalues.get("age", np.nan)

# Sex and help coefficients: interpret relative to reference category
sex_effects = {k: v * 60.0 for k, v in params.items() if k.startswith("C(sex)")}
sex_pvals = {k: v for k, v in pvalues.items() if k.startswith("C(sex)")}

help_effects = {k: v * 60.0 for k, v in params.items() if k.startswith("C(help)")}
help_pvals = {k: v for k, v in pvalues.items() if k.startswith("C(help)")}

# Determine evidence strength
sig_effects = {
    "age": age_p < 0.05,
    "sex": any(p < 0.05 for p in sex_pvals.values()) if sex_pvals else False,
    "help": any(p < 0.05 for p in help_pvals.values()) if help_pvals else False,
}

num_sig = sum(sig_effects.values())

# Construct Likert response
# Heuristic: if none significant -> low (30). if one -> moderate (60), if >=2 -> higher (75)
if num_sig == 0:
    response = 30
elif num_sig == 1:
    response = 60
else:
    response = 75

# Build explanation
lines = []
lines.append(
    f"Analyzed {n} sessions. Nut-cracking efficiency was defined as nuts opened per second; results are reported per minute for interpretability."
)
lines.append(
    f"Overall mean efficiency was {mean_eff:.2f} nuts/min (SD {std_eff:.2f})."
)

lines.append(
    f"OLS with robust (HC3) SEs: efficiency ~ age + sex + help, R²={r2:.3f}."
)

lines.append(
    f"Age effect: {age_coef_per_min:.2f} nuts/min per year (p={age_p:.3f})."
)

# Sex interpretation
if sex_means:
    sex_mean_str = ", ".join([f"{k}={v:.2f}" for k, v in sex_means.items()])
    lines.append(f"Mean efficiency by sex (nuts/min): {sex_mean_str}.")

if sex_effects:
    for k, v in sex_effects.items():
        p = sex_pvals.get(k, np.nan)
        lines.append(f"Sex effect {k} vs reference: {v:.2f} nuts/min (p={p:.3f}).")

# Help interpretation
if help_means:
    help_mean_str = ", ".join([f"{k}={v:.2f}" for k, v in help_means.items()])
    lines.append(f"Mean efficiency by help (nuts/min): {help_mean_str}.")

if help_effects:
    for k, v in help_effects.items():
        p = help_pvals.get(k, np.nan)
        lines.append(f"Help effect {k} vs reference: {v:.2f} nuts/min (p={p:.3f}).")

# Final interpretation statement
if num_sig == 0:
    lines.append(
        "None of age, sex, or receiving help showed statistically significant associations with efficiency at α=0.05, so evidence does not support an influence in this dataset."
    )
elif num_sig == 1:
    which = [k for k, v in sig_effects.items() if v]
    lines.append(
        f"Only {which[0]} showed a statistically significant association with efficiency at α=0.05; evidence for the others was not significant."
    )
else:
    which = [k for k, v in sig_effects.items() if v]
    lines.append(
        f"Multiple predictors (" + ", ".join(which) + ") showed statistically significant associations with efficiency at α=0.05."
    )

explanation = " ".join(lines)

output = {"response": int(response), "explanation": explanation}

with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump(output, f)
