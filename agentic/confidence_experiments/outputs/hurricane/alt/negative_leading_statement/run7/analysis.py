import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "hurricane.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleaning
for col in ["alldeaths", "masfem", "wind", "min", "category", "ndam", "ndam15", "gender_mf", "year"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Use log1p of deaths to reduce skew (common in hurricane fatality data)
df["log_deaths"] = np.log1p(df["alldeaths"])

results = {}

# Bivariate correlation between femininity and deaths
corr = df[["masfem", "alldeaths"]].corr().iloc[0, 1]
results["corr_masfem_alldeaths"] = corr

# Bivariate regression: log_deaths ~ masfem
X = sm.add_constant(df["masfem"])
model_simple = sm.OLS(df["log_deaths"], X, missing="drop").fit()
results["simple_coef_masfem"] = model_simple.params.get("masfem", np.nan)
results["simple_p_masfem"] = model_simple.pvalues.get("masfem", np.nan)

# Controlled regression: log_deaths ~ masfem + wind + min + category + ndam15 + year
# Avoid multicollinearity with ndam vs ndam15; use ndam15 when available.
controls = ["masfem", "wind", "min", "category", "ndam15", "year"]
Xc = sm.add_constant(df[controls])
model_controls = sm.OLS(df["log_deaths"], Xc, missing="drop").fit()
results["controls_coef_masfem"] = model_controls.params.get("masfem", np.nan)
results["controls_p_masfem"] = model_controls.pvalues.get("masfem", np.nan)
results["controls_n"] = int(model_controls.nobs)

# Alternative control set without year (sensitivity)
controls2 = ["masfem", "wind", "min", "category", "ndam15"]
Xc2 = sm.add_constant(df[controls2])
model_controls2 = sm.OLS(df["log_deaths"], Xc2, missing="drop").fit()
results["controls2_coef_masfem"] = model_controls2.params.get("masfem", np.nan)
results["controls2_p_masfem"] = model_controls2.pvalues.get("masfem", np.nan)
results["controls2_n"] = int(model_controls2.nobs)

# Save results for inspection
with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
